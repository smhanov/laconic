package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/smhanov/laconic"
)

// Tavily calls the Tavily search API.
type Tavily struct {
	APIKey string
	client *http.Client
	// Depth controls Tavily's depth parameter (basic or advanced).
	Depth string
}

// NewTavily constructs a Tavily search provider.
func NewTavily(apiKey string, depth string) *Tavily {
	if depth == "" {
		depth = "basic"
	}
	return &Tavily{APIKey: apiKey, Depth: depth, client: &http.Client{Timeout: 10 * time.Second}}
}

// NewTavilyWithClient constructs a Tavily search provider using the supplied HTTP client.
// This is useful for overriding the default timeout.
func NewTavilyWithClient(apiKey string, depth string, client *http.Client) *Tavily {
	if depth == "" {
		depth = "basic"
	}
	return &Tavily{APIKey: apiKey, Depth: depth, client: client}
}

// Search posts a query to Tavily.
func (t *Tavily) Search(ctx context.Context, query string) ([]laconic.SearchResult, error) {
	if strings.TrimSpace(t.APIKey) == "" {
		return nil, errors.New("tavily: API key is missing")
	}

	body := map[string]any{
		"query":   query,
		"api_key": t.APIKey,
		"depth":   t.Depth,
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	var resp *http.Response
	delay := 1 * time.Second
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.tavily.com/search", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err = t.client.Do(req)
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusTooManyRequests {
			break
		}
		resp.Body.Close()

		// Back off and retry on 429, doubling the delay each time up to 30 s.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
		if delay < 30*time.Second {
			delay *= 2
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tavily http %d", resp.StatusCode)
	}

	var response struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	results := make([]laconic.SearchResult, 0, len(response.Results))
	for _, r := range response.Results {
		results = append(results, laconic.SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Content})
		if len(results) >= 5 {
			break
		}
	}
	return results, nil
}
