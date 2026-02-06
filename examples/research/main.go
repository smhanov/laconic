package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/smhanov/laconic"
	"github.com/smhanov/laconic/fetch"
	"github.com/smhanov/laconic/search"
)

// ---------------------------------------------------------------------------
// Ollama native API backend
// ---------------------------------------------------------------------------

// OllamaLLM implements laconic.LLMProvider using the Ollama /api/generate endpoint.
type OllamaLLM struct {
	Endpoint string
	Model    string
	Debug    bool
}

type ollamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"`
	Stream bool   `json:"stream"`
}

type ollamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (o *OllamaLLM) Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	if o.Debug {
		log.Printf("\n=== LLM Request (%s) ===\n[SYSTEM]\n%s\n\n[USER]\n%s\n=======================", o.Model, systemPrompt, userPrompt)
	}

	endpoint := normalizeEndpoint(o.Endpoint)
	url := fmt.Sprintf("%s/api/generate", endpoint)

	reqBody := ollamaRequest{
		Model:  o.Model,
		Prompt: userPrompt,
		System: systemPrompt,
		Stream: false,
	}

	body, err := doRequestWithRetries(ctx, url, "", reqBody, o.Debug, "ollama")
	if err != nil {
		return "", err
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	response := strings.TrimSpace(ollamaResp.Response)
	if o.Debug {
		log.Printf("\n=== LLM Response ===\n%s\n====================\n", response)
	}

	return response, nil
}

// ---------------------------------------------------------------------------
// OpenAI-compatible chat completions backend
// ---------------------------------------------------------------------------

// OpenAILLM implements laconic.LLMProvider using the OpenAI chat completions API.
// Works with any server that exposes the /v1/chat/completions endpoint
// (OpenAI, Ollama /v1, vLLM, LiteLLM, etc.).
type OpenAILLM struct {
	Endpoint string // base URL, e.g. https://api.openai.com or https://ollama.example.com/v1
	Model    string
	APIKey   string // optional — leave empty for keyless servers
	Debug    bool
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openaiRequest struct {
	Model    string          `json:"model"`
	Messages []openaiMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type openaiChoice struct {
	Message openaiMessage `json:"message"`
}

type openaiResponse struct {
	Choices []openaiChoice `json:"choices"`
}

func (o *OpenAILLM) Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	if o.Debug {
		log.Printf("\n=== LLM Request (%s) ===\n[SYSTEM]\n%s\n\n[USER]\n%s\n=======================", o.Model, systemPrompt, userPrompt)
	}

	endpoint := normalizeEndpoint(o.Endpoint)
	// Append /v1/chat/completions if the endpoint doesn't already end with a path
	url := strings.TrimRight(endpoint, "/")
	if !strings.HasSuffix(url, "/chat/completions") {
		if !strings.HasSuffix(url, "/v1") {
			url += "/v1"
		}
		url += "/chat/completions"
	}

	msgs := []openaiMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	reqBody := openaiRequest{
		Model:    o.Model,
		Messages: msgs,
		Stream:   false,
	}

	body, err := doRequestWithRetries(ctx, url, o.APIKey, reqBody, o.Debug, "openai")
	if err != nil {
		return "", err
	}

	var oaiResp openaiResponse
	if err := json.Unmarshal(body, &oaiResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}
	if len(oaiResp.Choices) == 0 {
		return "", fmt.Errorf("openai response contained no choices")
	}

	response := strings.TrimSpace(oaiResp.Choices[0].Message.Content)
	if o.Debug {
		log.Printf("\n=== LLM Response ===\n%s\n====================\n", response)
	}

	return response, nil
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

func normalizeEndpoint(endpoint string) string {
	if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
		return "http://" + endpoint
	}
	return endpoint
}

func doRequestWithRetries(ctx context.Context, url, apiKey string, reqBody interface{}, debug bool, label string) ([]byte, error) {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Use a client with a generous timeout so large-model requests don't
	// hang indefinitely but still have enough time to generate.
	client := &http.Client{Timeout: 10 * time.Minute}

	var body []byte
	maxRetries := 5
	baseDelay := 1 * time.Second

	for i := 0; i <= maxRetries; i++ {
		log.Printf("[%s] POST %s (attempt %d)…", label, url, i+1)
		start := time.Now()

		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to send request after %v: %w", time.Since(start).Truncate(time.Second), err)
		}

		if resp.StatusCode == http.StatusOK {
			body, err = io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, fmt.Errorf("failed to read response: %w", err)
			}
			log.Printf("[%s] response received in %v", label, time.Since(start).Truncate(time.Second))
			return body, nil
		}

		errBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusGatewayTimeout {
			if i == maxRetries {
				return nil, fmt.Errorf("%s API error after retries: %s - %s", label, resp.Status, string(errBody))
			}
			delay := baseDelay * time.Duration(1<<i)
			if debug {
				log.Printf("Got %s, retrying in %v...", resp.Status, delay)
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
				continue
			}
		}

		return nil, fmt.Errorf("%s API error: %s - %s", label, resp.Status, string(errBody))
	}

	return body, nil
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	backend := flag.String("backend", "ollama", "LLM backend: ollama or openai")
	endpoint := flag.String("endpoint", "", "API endpoint URL (default: localhost:11434 for ollama, https://api.openai.com for openai)")
	apiKey := flag.String("api-key", "", "API key for authenticated endpoints (e.g. OpenAI)")
	model := flag.String("model", "", "Model name to use (required)")
	promptFile := flag.String("prompt", "", "Path to prompt file (required)")
	maxIterations := flag.Int("max-iterations", 5, "Maximum search iterations")
	strategy := flag.String("strategy", "scratchpad", "Strategy to use: scratchpad or graph-reader")
	graphSteps := flag.Int("graph-max-steps", 8, "Maximum steps for graph-reader strategy")
	searchProvider := flag.String("search", "duckduckgo", "Search provider: duckduckgo or brave")
	braveKey := flag.String("brave-key", "", "Brave Search API key (required when -search=brave)")
	debug := flag.Bool("debug", false, "Print full LLM prompts and responses")

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

	// Build the LLM provider based on the chosen backend.
	var llm laconic.LLMProvider
	switch strings.ToLower(*backend) {
	case "openai":
		ep := *endpoint
		if ep == "" {
			ep = "https://api.openai.com"
		}
		llm = &OpenAILLM{
			Endpoint: ep,
			Model:    *model,
			APIKey:   *apiKey,
			Debug:    *debug,
		}
	default: // "ollama"
		ep := *endpoint
		if ep == "" {
			ep = "localhost:11434"
		}
		llm = &OllamaLLM{
			Endpoint: ep,
			Model:    *model,
			Debug:    *debug,
		}
	}

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

	fmt.Printf("Using %s backend at %s with model %s\n", *backend, *endpoint, *model)
	fmt.Printf("Strategy: %s\n", *strategy)
	fmt.Printf("Question: %s\n\n", question)

	ans, err := agent.Answer(context.Background(), question)
	if err != nil {
		log.Printf("Warning: %v", err)
	}
	fmt.Printf("Answer:\n%s\n", ans)
}
