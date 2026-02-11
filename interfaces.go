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

// LLMResponse is returned by LLMProvider.Generate and carries both the
// generated text and the cost (in dollars) of the call.
type LLMResponse struct {
	Text string
	Cost float64
}

// LLMProvider is implemented by user-supplied language model clients.
type LLMProvider interface {
	Generate(ctx context.Context, systemPrompt, userPrompt string) (LLMResponse, error)
}

// Result is returned by Agent.Answer and carries the final answer text
// together with the total cost accumulated during the research loop.
type Result struct {
	Answer    string
	Cost      float64
	Knowledge string // collected knowledge from the research session
}

// AnswerOption configures a single call to Agent.Answer.
type AnswerOption func(*answerConfig)

type answerConfig struct {
	priorKnowledge string
}

// WithKnowledge supplies prior knowledge collected from a previous research
// session. This is typically the Knowledge field from a prior Result.
// Strategies use it to pre-populate their internal state so the agent can
// answer follow-up questions without re-searching for already-known facts.
func WithKnowledge(knowledge string) AnswerOption {
	return func(c *answerConfig) { c.priorKnowledge = knowledge }
}
