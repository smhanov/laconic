package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/smhanov/laconic"
	"github.com/smhanov/laconic/fetch"
	"github.com/smhanov/laconic/search"
	"github.com/smhanov/llmhub"
	_ "github.com/smhanov/llmhub/providers/anthropic"
	_ "github.com/smhanov/llmhub/providers/gemini"
	_ "github.com/smhanov/llmhub/providers/ollama"
	_ "github.com/smhanov/llmhub/providers/openai"
)

// ---------------------------------------------------------------------------
// llmhub backend
// ---------------------------------------------------------------------------

// HubLLM adapts llmhub.Client to laconic.LLMProvider.
type HubLLM struct {
	Client   *llmhub.Client
	Provider string
	Model    string
	Debug    bool
}

func (h *HubLLM) Generate(ctx context.Context, systemPrompt, userPrompt string) (laconic.LLMResponse, error) {
	if h.Debug {
		log.Printf("\n=== LLM Request (%s/%s) ===\n[SYSTEM]\n%s\n\n[USER]\n%s\n=======================", h.Provider, h.Model, systemPrompt, userPrompt)
	}

	prompt := []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text(systemPrompt)),
		llmhub.NewUserMessage(llmhub.Text(userPrompt)),
	}

	resp, err := h.Client.Generate(ctx, prompt)
	if err != nil {
		return laconic.LLMResponse{}, err
	}

	text := strings.TrimSpace(resp.Text())
	if h.Debug {
		log.Printf("\n=== LLM Response ===\n%s\n====================\n", text)
	}

	return laconic.LLMResponse{Text: text, Cost: resp.Usage.Cost}, nil
}

// ---------------------------------------------------------------------------
// varMap is a repeatable flag that collects key=value pairs.
// ---------------------------------------------------------------------------

type varMap map[string]string

func (v *varMap) String() string { return fmt.Sprintf("%v", map[string]string(*v)) }

func (v *varMap) Set(s string) error {
	parts := strings.SplitN(s, "=", 2)
	if len(parts) != 2 || parts[0] == "" {
		return fmt.Errorf("expected KEY=VALUE, got %q", s)
	}
	(*v)[parts[0]] = parts[1]
	return nil
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	provider := flag.String("provider", "ollama", "LLM provider: ollama, openai, anthropic, gemini")
	backend := flag.String("backend", "", "Alias for -provider (deprecated)")
	endpoint := flag.String("endpoint", "", "Provider endpoint URL override (e.g. https://api.openai.com or OpenAI-compatible base URL)")
	apiKey := flag.String("api-key", "", "API key for authenticated endpoints (e.g. OpenAI)")
	model := flag.String("model", "", "Model name to use (required)")
	promptFile := flag.String("prompt", "", "Path to prompt file (required)")
	maxIterations := flag.Int("max-iterations", 5, "Maximum search iterations")
	strategy := flag.String("strategy", "scratchpad", "Strategy to use: scratchpad or graph-reader")
	graphSteps := flag.Int("graph-max-steps", 8, "Maximum steps for graph-reader strategy")
	searchProvider := flag.String("search", "duckduckgo", "Search provider: duckduckgo or brave")
	braveKey := flag.String("brave-key", "", "Brave Search API key (required when -search=brave)")
	knowledgeFile := flag.String("knowledge", "", "Path to a file for reading/writing collected knowledge (enables follow-up questions)")
	debug := flag.Bool("debug", false, "Print full LLM prompts and responses")
	vars := make(varMap)
	flag.Var(&vars, "var", "Set a template variable: -var KEY=VALUE (repeatable). Replaces {{KEY}} in prompt file.")

	flag.Parse()

	if *model == "" {
		log.Fatal("Error: -model is required")
	}
	if *promptFile == "" {
		log.Fatal("Error: -prompt is required")
	}

	// Read prompt from file
	promptData, err := os.ReadFile(*promptFile)
	if err != nil {
		log.Fatalf("Error reading prompt file: %v", err)
	}
	question := strings.TrimSpace(string(promptData))

	if question == "" {
		log.Fatal("Error: prompt file is empty")
	}

	// Replace {{KEY}} placeholders with values supplied via -var flags.
	for k, v := range vars {
		question = strings.ReplaceAll(question, "{{"+k+"}}", v)
	}

	selectedProvider := strings.TrimSpace(strings.ToLower(*provider))
	if selectedProvider == "" {
		selectedProvider = strings.TrimSpace(strings.ToLower(*backend))
	}
	if selectedProvider == "" {
		selectedProvider = "ollama"
	}

	effectiveAPIKey := strings.TrimSpace(*apiKey)

	llmOpts := []llmhub.Option{
		llmhub.WithModel(*model),
		llmhub.WithHTTPClient(&http.Client{Timeout: 10 * time.Minute}),
	}
	if strings.TrimSpace(*endpoint) != "" {
		llmOpts = append(llmOpts, llmhub.WithBaseURL(strings.TrimSpace(*endpoint)))
	}

	client, err := llmhub.New(selectedProvider, effectiveAPIKey, llmOpts...)
	if err != nil {
		log.Fatalf("Error creating llmhub client: %v", err)
	}

	llm := &HubLLM{Client: client, Provider: selectedProvider, Model: *model, Debug: *debug}

	var searcher laconic.SearchProvider
	switch strings.ToLower(*searchProvider) {
	case "brave":
		if *braveKey == "" {
			log.Fatal("Error: -brave-key is required when using brave search")
		}
		searcher = search.NewBrave(*braveKey)
	default:
		searcher = search.NewDuckDuckGo()
	}

	agent := laconic.New(
		laconic.WithPlannerModel(llm),
		laconic.WithSynthesizerModel(llm),
		laconic.WithSearchProvider(searcher),
		laconic.WithMaxIterations(*maxIterations),
		laconic.WithStrategyName(*strategy),
		laconic.WithGraphReaderConfig(laconic.GraphReaderConfig{MaxSteps: *graphSteps}),
		laconic.WithFetchProvider(fetch.NewHTTP()),
		laconic.WithDebug(*debug),
	)

	fmt.Printf("Using provider %s with model %s\n", selectedProvider, *model)
	if strings.TrimSpace(*endpoint) != "" {
		fmt.Printf("Endpoint override: %s\n", strings.TrimSpace(*endpoint))
	}
	fmt.Printf("Strategy: %s\n", *strategy)
	fmt.Printf("Question: %s\n\n", question)

	// Load prior knowledge if the knowledge file exists.
	var answerOpts []laconic.AnswerOption
	if *knowledgeFile != "" {
		if kd, err := os.ReadFile(*knowledgeFile); err == nil {
			prior := strings.TrimSpace(string(kd))
			if prior != "" {
				fmt.Printf("Loaded prior knowledge from %s (%d bytes)\n", *knowledgeFile, len(prior))
				answerOpts = append(answerOpts, laconic.WithKnowledge(prior))
			}
		}
	}

	result, err := agent.Answer(context.Background(), question, answerOpts...)
	if err != nil {
		log.Printf("Warning: %v", err)
	}

	// Save collected knowledge to file.
	if *knowledgeFile != "" && result.Knowledge != "" {
		if err := os.WriteFile(*knowledgeFile, []byte(result.Knowledge), 0644); err != nil {
			log.Printf("Warning: failed to write knowledge file: %v", err)
		} else {
			fmt.Printf("Knowledge saved to %s (%d bytes)\n", *knowledgeFile, len(result.Knowledge))
		}
	}

	fmt.Printf("Answer:\n%s\n", result.Answer)
	if result.Cost > 0 {
		fmt.Printf("Total cost: $%.4f\n", result.Cost)
	}
}
