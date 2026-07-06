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

const (
	serperEndpoint    = "https://google.serper.dev/search"
	serperResultLimit = 5
)

// Serper calls the Serper Google Search API. An API key is required via the
// X-API-KEY header.
type Serper struct {
	APIKey   string
	client   *http.Client
	endpoint string
}

// NewSerper constructs a Serper search provider.
func NewSerper(apiKey string) *Serper {
	return &Serper{
		APIKey:   apiKey,
		client:   &http.Client{Timeout: 10 * time.Second},
		endpoint: serperEndpoint,
	}
}

// NewSerperWithClient constructs a Serper search provider using the supplied HTTP client.
// This is useful for overriding the default timeout.
func NewSerperWithClient(apiKey string, client *http.Client) *Serper {
	return &Serper{APIKey: apiKey, client: client, endpoint: serperEndpoint}
}

// Search posts a query to Serper and converts organic results to SearchResult values.
func (s *Serper) Search(ctx context.Context, query string) ([]laconic.SearchResult, error) {
	if strings.TrimSpace(s.APIKey) == "" {
		return nil, errors.New("serper: API key is missing")
	}

	body := map[string]any{
		"q":   query,
		"num": serperResultLimit,
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	endpoint := s.endpoint
	if endpoint == "" {
		endpoint = serperEndpoint
	}

	var resp *http.Response
	delay := 1 * time.Second
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-API-KEY", s.APIKey)

		resp, err = s.client.Do(req)
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
		return nil, fmt.Errorf("serper http %d", resp.StatusCode)
	}

	var response struct {
		Organic []struct {
			Title   string `json:"title"`
			Link    string `json:"link"`
			Snippet string `json:"snippet"`
		} `json:"organic"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	results := make([]laconic.SearchResult, 0, len(response.Organic))
	for _, r := range response.Organic {
		results = append(results, laconic.SearchResult{Title: r.Title, URL: r.Link, Snippet: r.Snippet})
		if len(results) >= serperResultLimit {
			break
		}
	}
	return results, nil
}
