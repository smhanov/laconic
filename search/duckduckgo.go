package search

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/smhanov/laconic"
)

// ddgRateLimit enforces a global rate limit of 1 query per second across all
// DuckDuckGo instances and goroutines.
var ddgRateLimit struct {
	mu   sync.Mutex
	last time.Time
}

// DuckDuckGo implements a searcher using DuckDuckGo's HTML lite interface.
type DuckDuckGo struct {
	client *http.Client
}

// NewDuckDuckGo creates a DuckDuckGo searcher with a modest timeout.
func NewDuckDuckGo() *DuckDuckGo {
	return &DuckDuckGo{client: &http.Client{Timeout: 15 * time.Second}}
}

// NewDuckDuckGoWithClient creates a DuckDuckGo searcher using the supplied HTTP client.
// This is useful for overriding the default timeout.
func NewDuckDuckGoWithClient(client *http.Client) *DuckDuckGo {
	return &DuckDuckGo{client: client}
}

// Search scrapes the DuckDuckGo lite HTML page for results.
func (d *DuckDuckGo) Search(ctx context.Context, query string) ([]laconic.SearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, errors.New("query is empty")
	}

	// Enforce global 1 QPS rate limit.
	ddgRateLimit.mu.Lock()
	if wait := time.Until(ddgRateLimit.last.Add(time.Second)); wait > 0 {
		ddgRateLimit.mu.Unlock()
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		ddgRateLimit.mu.Lock()
	}
	ddgRateLimit.last = time.Now()
	ddgRateLimit.mu.Unlock()

	// Use the lite HTML version which is more stable for scraping
	endpoint := "https://lite.duckduckgo.com/lite/"
	
	formData := url.Values{}
	formData.Set("q", query)

	var resp *http.Response
	delay := 1 * time.Second
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(formData.Encode()))
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

		resp, err = d.client.Do(req)
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
		return nil, fmt.Errorf("duckduckgo http %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return parseHTMLResults(string(body)), nil
}

// parseHTMLResults extracts search results from the DuckDuckGo lite HTML.
// The lite page has a simple structure with result links and snippets.
func parseHTMLResults(html string) []laconic.SearchResult {
	var results []laconic.SearchResult

	// Pattern to find result links: <a rel="nofollow" href="URL" class='result-link'>TITLE</a>
	linkPattern := regexp.MustCompile(`<a[^>]*class=['"]result-link['"][^>]*href=['"]([^'"]+)['"][^>]*>([^<]+)</a>`)
	// Alternative pattern if class comes before href
	linkPattern2 := regexp.MustCompile(`<a[^>]*href=['"]([^'"]+)['"][^>]*class=['"]result-link['"][^>]*>([^<]+)</a>`)
	
	// Pattern to find snippets in <td> with class "result-snippet"
	snippetPattern := regexp.MustCompile(`<td[^>]*class=['"]result-snippet['"][^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</td>`)
	
	// First try the standard link patterns
	matches := linkPattern.FindAllStringSubmatch(html, -1)
	if len(matches) == 0 {
		matches = linkPattern2.FindAllStringSubmatch(html, -1)
	}

	snippetMatches := snippetPattern.FindAllStringSubmatch(html, -1)

	for i, match := range matches {
		if len(match) < 3 {
			continue
		}
		
		urlStr := strings.TrimSpace(match[1])
		title := strings.TrimSpace(match[2])
		
		// Clean up HTML entities
		title = cleanHTML(title)
		
		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = cleanHTML(snippetMatches[i][1])
		}
		
		// Skip ad results or empty results
		if urlStr == "" || title == "" {
			continue
		}
		
		results = append(results, laconic.SearchResult{
			Title:   title,
			URL:     urlStr,
			Snippet: snippet,
		})
		
		if len(results) >= 5 {
			break
		}
	}

	// If the regex approach didn't work well, try a simpler fallback
	// Look for any links that look like search results (external URLs)
	if len(results) == 0 {
		results = fallbackParse(html)
	}

	return results
}

// fallbackParse tries a simpler approach to extract links
func fallbackParse(html string) []laconic.SearchResult {
	var results []laconic.SearchResult
	
	// Look for links that appear to be search results
	linkPattern := regexp.MustCompile(`<a[^>]+href=['"]([^'"]+)['"][^>]*>([^<]+)</a>`)
	matches := linkPattern.FindAllStringSubmatch(html, -1)
	
	seen := make(map[string]bool)
	for _, match := range matches {
		if len(match) < 3 {
			continue
		}
		
		urlStr := strings.TrimSpace(match[1])
		title := cleanHTML(strings.TrimSpace(match[2]))
		
		// Skip DuckDuckGo internal links
		if strings.Contains(urlStr, "duckduckgo.com") || 
		   strings.HasPrefix(urlStr, "/") ||
		   strings.HasPrefix(urlStr, "#") ||
		   strings.HasPrefix(urlStr, "javascript:") {
			continue
		}
		
		// Skip if title is too short or looks like navigation
		if len(title) < 5 {
			continue
		}
		
		// Dedupe by URL
		if seen[urlStr] {
			continue
		}
		seen[urlStr] = true
		
		results = append(results, laconic.SearchResult{
			Title:   title,
			URL:     urlStr,
			Snippet: "",
		})
		
		if len(results) >= 5 {
			break
		}
	}
	
	return results
}

// cleanHTML removes HTML entities and tags
func cleanHTML(s string) string {
	// Remove HTML tags
	tagPattern := regexp.MustCompile(`<[^>]+>`)
	s = tagPattern.ReplaceAllString(s, "")
	
	// Decode common entities
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	
	return strings.TrimSpace(s)
}
