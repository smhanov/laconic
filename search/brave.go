package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/smhanov/laconic"
)

// Brave uses the Brave Search API. An API key is required via X-Subscription-Token.
type Brave struct {
	APIKey string
	client *http.Client
}

// NewBrave constructs a Brave search provider.
func NewBrave(apiKey string) *Brave {
	return &Brave{APIKey: apiKey, client: &http.Client{Timeout: 10 * time.Second}}
}

// Search executes a Brave query.
func (b *Brave) Search(ctx context.Context, query string) ([]laconic.SearchResult, error) {
	if strings.TrimSpace(b.APIKey) == "" {
		return nil, errors.New("brave: API key is missing")
	}
	encoded := url.QueryEscape(query)
	endpoint := fmt.Sprintf("https://api.search.brave.com/res/v1/web/search?q=%s", encoded)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", b.APIKey)

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("brave http %d", resp.StatusCode)
	}

	var payload struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}

	results := make([]laconic.SearchResult, 0, len(payload.Web.Results))
	for _, r := range payload.Web.Results {
		results = append(results, laconic.SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Description})
		if len(results) >= 5 {
			break
		}
	}

	return results, nil
}
