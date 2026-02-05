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

	"github.com/smhanov/laconic"
	"github.com/smhanov/laconic/search"
)

// OllamaLLM implements laconic.LLMProvider using the Ollama API.
type OllamaLLM struct {
	Endpoint string
	Model    string
	Debug    bool
}

type ollamaRequest struct {
	Model  string          `json:"model"`
	Prompt string          `json:"prompt"`
	System string          `json:"system,omitempty"`
	Stream bool            `json:"stream"`
}

type ollamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (o *OllamaLLM) Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	if o.Debug {
		log.Printf("\n=== LLM Request (%s) ===\n[SYSTEM]\n%s\n\n[USER]\n%s\n=======================", o.Model, systemPrompt, userPrompt)
	}

	endpoint := o.Endpoint
	if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
		endpoint = "http://" + endpoint
	}
	url := fmt.Sprintf("%s/api/generate", endpoint)

	reqBody := ollamaRequest{
		Model:  o.Model,
		Prompt: userPrompt,
		System: systemPrompt,
		Stream: false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama API error: %s - %s", resp.Status, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
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

func main() {
	endpoint := flag.String("endpoint", "localhost:11434", "Ollama API endpoint (host:port)")
	model := flag.String("model", "", "Model name to use (required)")
	promptFile := flag.String("prompt", "", "Path to prompt file (required)")
	maxIterations := flag.Int("max-iterations", 5, "Maximum search iterations")
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

	llm := &OllamaLLM{
		Endpoint: *endpoint,
		Model:    *model,
		Debug:    *debug,
	}

	agent := laconic.New(
		laconic.WithPlannerModel(llm),
		laconic.WithSynthesizerModel(llm),
		laconic.WithSearchProvider(search.NewDuckDuckGo()),
		laconic.WithMaxIterations(*maxIterations),
	)

	fmt.Printf("Using Ollama at %s with model %s\n", *endpoint, *model)
	fmt.Printf("Question: %s\n\n", question)

	ans, err := agent.Answer(context.Background(), question)
	if err != nil {
		log.Printf("Warning: %v", err)
	}
	fmt.Printf("Answer:\n%s\n", ans)
}
