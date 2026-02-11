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
	"sync"
	"time"

	"github.com/smhanov/laconic"
)

// braveKeyGate holds a per-API-key mutex and the earliest time that a request
// is allowed. All Brave instances sharing an API key share a single gate so
// that only one request per second is issued for that key, matching the
// Brave rate-limit of 1 req/s.
type braveKeyGate struct {
	mu        sync.Mutex
	readyAt   time.Time // earliest moment the next request may fire
}

var (
	braveGatesMu sync.Mutex
	braveGates   = map[string]*braveKeyGate{}
)

// braveGateFor returns (or creates) the shared gate for the given API key.
func braveGateFor(apiKey string) *braveKeyGate {
	braveGatesMu.Lock()
	defer braveGatesMu.Unlock()
	g, ok := braveGates[apiKey]
	if !ok {
		g = &braveKeyGate{}
		braveGates[apiKey] = g
	}
	return g
}

// waitAndLock blocks until the caller may issue a request, then returns with
// the gate locked. The caller MUST call gate.unlock(delay) after receiving
// the response to set the next allowed time and release the lock.
// Returns ctx.Err() if the context expires while waiting.
func (g *braveKeyGate) waitAndLock(ctx context.Context) error {
	g.mu.Lock()
	now := time.Now()
	if wait := g.readyAt.Sub(now); wait > 0 {
		g.mu.Unlock() // release while sleeping
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
		g.mu.Lock()
	}
	return nil
}

// unlock sets the minimum delay before the next request and releases the
// gate so the next waiter may proceed.
func (g *braveKeyGate) unlock(delay time.Duration) {
	g.readyAt = time.Now().Add(delay)
	g.mu.Unlock()
}

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

// Search executes a Brave query. Concurrent calls sharing the same API key
// are serialised through a shared per-key gate to respect rate limits.
func (b *Brave) Search(ctx context.Context, query string) ([]laconic.SearchResult, error) {
	if strings.TrimSpace(b.APIKey) == "" {
		return nil, errors.New("brave: API key is missing")
	}
	encoded := url.QueryEscape(query)
	endpoint := fmt.Sprintf("https://api.search.brave.com/res/v1/web/search?q=%s", encoded)

	gate := braveGateFor(b.APIKey)

	var resp *http.Response
	var err error
	for {
		// Wait for our turn under the shared gate.
		if err := gate.waitAndLock(ctx); err != nil {
			return nil, err
		}

		req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if reqErr != nil {
			gate.unlock(0)
			return nil, reqErr
		}
		req.Header.Set("Accept", "application/json")
		req.Header.Set("X-Subscription-Token", b.APIKey)

		resp, err = b.client.Do(req)
		if err != nil {
			gate.unlock(1 * time.Second) // back off before letting others try
			return nil, err
		}

		if resp.StatusCode != http.StatusTooManyRequests {
			// Use the per-second rate-limit header to pace the next caller.
			gate.unlock(braveNextDelay(resp.Header))
			break
		}

		// 429 — read the retry delay, tell the gate, then loop.
		wait := braveRetryDelay(resp.Header)
		resp.Body.Close()
		gate.unlock(wait)
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

// braveNextDelay reads X-RateLimit-Remaining to decide how long to hold the
// gate before allowing the next request. If the per-second bucket is
// exhausted (remaining == 0), we wait 1 second. Otherwise we allow
// immediately.
func braveNextDelay(h http.Header) time.Duration {
	raw := h.Get("X-RateLimit-Remaining")
	if raw == "" {
		return 1 * time.Second // be conservative when header is absent
	}
	// The header is comma-separated: "0, 14832" (per-second, per-month).
	parts := strings.SplitN(raw, ",", 2)
	perSecond, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 1 * time.Second
	}
	if perSecond <= 0 {
		return 1 * time.Second
	}
	return 0
}
