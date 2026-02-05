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

// LLMProvider is implemented by user-supplied language model clients.
type LLMProvider interface {
	Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error)
}
