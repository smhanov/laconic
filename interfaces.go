package laconic

import "context"

// SearchResult is a single item returned by a SearchProvider.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// SearchProvider executes a query and returns results.
type SearchProvider interface {
	Search(ctx context.Context, query string) ([]SearchResult, error)
}

// FetchProvider retrieves raw content for a URL.
// Graph-based strategies can use it to read full pages when snippets are insufficient.
type FetchProvider interface {
	Fetch(ctx context.Context, url string) (string, error)
}

// LLMProvider is implemented by user-supplied language model clients.
type LLMProvider interface {
	Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error)
}
