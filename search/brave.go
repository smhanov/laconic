package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/smhanov/laconic"
)

const braveMaxRetries = 3

// Brave uses the Brave Search API. An API key is required via X-Subscription-Token.
type Brave struct {
	APIKey string
	client *http.Client
}

// NewBrave constructs a Brave search provider.
func NewBrave(apiKey string) *Brave {
	return &Brave{APIKey: apiKey, client: &http.Client{Timeout: 10 * time.Second}}
}

// NewBraveWithClient constructs a Brave search provider using the supplied HTTP client.
// This is useful for overriding the default timeout.
func NewBraveWithClient(apiKey string, client *http.Client) *Brave {
	return &Brave{APIKey: apiKey, client: client}
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

	var resp *http.Response
	for attempt := 0; attempt <= braveMaxRetries; attempt++ {
		if attempt > 0 {
			// Clone the request for retries (body is nil for GET so this is safe).
			req, err = http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
			if err != nil {
				return nil, err
			}
			req.Header.Set("Accept", "application/json")
			req.Header.Set("X-Subscription-Token", b.APIKey)
		}

		resp, err = b.client.Do(req)
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusTooManyRequests {
			break
		}
		resp.Body.Close()

		wait := braveRetryDelay(resp.Header)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
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

// braveRetryDelay reads the X-RateLimit-Reset header to determine how long
// to wait before retrying. The header contains a comma-separated list of
// reset times in seconds (e.g. "1, 1419704"); we use the smallest value.
// Falls back to 1 second if the header is missing or unparseable.
func braveRetryDelay(h http.Header) time.Duration {
	raw := h.Get("X-RateLimit-Reset")
	if raw == "" {
		return 1 * time.Second
	}
	minReset := -1
	for _, part := range strings.Split(raw, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n < 0 {
			continue
		}
		if minReset < 0 || n < minReset {
			minReset = n
		}
	}
	if minReset <= 0 {
		return 1 * time.Second
	}
	return time.Duration(minReset) * time.Second
}
