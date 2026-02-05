package fetch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const maxFetchBytes = 32 * 1024 // 32KB limit to avoid overwhelming LLM context

// HTTPFetcher retrieves raw text from a URL.
type HTTPFetcher struct {
	client *http.Client
}

// NewHTTP creates a HTTP fetcher with a modest timeout.
func NewHTTP() *HTTPFetcher {
	return &HTTPFetcher{client: &http.Client{Timeout: 15 * time.Second}}
}

// Fetch downloads the URL content, strips HTML to plain text, and truncates.
func (f *HTTPFetcher) Fetch(ctx context.Context, url string) (string, error) {
	trimmed := strings.TrimSpace(url)
	if trimmed == "" {
		return "", errors.New("fetch url is empty")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, trimmed, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

	resp, err := f.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("fetch http %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	text := stripHTML(string(body))
	if len(text) > maxFetchBytes {
		text = text[:maxFetchBytes] + "\n[TRUNCATED]"
	}
	return text, nil
}

var (
	reScript     = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	reStyle      = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	reNav        = regexp.MustCompile(`(?is)<nav[^>]*>.*?</nav>`)
	reHeader     = regexp.MustCompile(`(?is)<header[^>]*>.*?</header>`)
	reFooter     = regexp.MustCompile(`(?is)<footer[^>]*>.*?</footer>`)
	reTags       = regexp.MustCompile(`<[^>]+>`)
	reWhitespace = regexp.MustCompile(`[ \t]+`)
	reBlankLines = regexp.MustCompile(`\n{3,}`)
)

// stripHTML removes scripts, styles, nav/header/footer, then all tags.
func stripHTML(html string) string {
	s := reScript.ReplaceAllString(html, "")
	s = reStyle.ReplaceAllString(s, "")
	s = reNav.ReplaceAllString(s, "")
	s = reHeader.ReplaceAllString(s, "")
	s = reFooter.ReplaceAllString(s, "")
	s = reTags.ReplaceAllString(s, " ")

	// Decode common entities
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")

	// Collapse whitespace
	s = reWhitespace.ReplaceAllString(s, " ")
	// Normalize newlines
	lines := strings.Split(s, "\n")
	var out []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	s = strings.Join(out, "\n")
	s = reBlankLines.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}
