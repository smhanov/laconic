package laconic

const defaultMaxIterations = 5

// Option configures an Agent.
type Option func(*Agent)

// WithSearchProvider sets the search implementation.
func WithSearchProvider(searcher SearchProvider) Option {
	return func(a *Agent) { a.searcher = searcher }
}

// WithPlannerModel sets the model used for routing/planning.
func WithPlannerModel(m LLMProvider) Option {
	return func(a *Agent) { a.planner = m }
}

// WithSynthesizerModel sets the model used for compressing updates.
func WithSynthesizerModel(m LLMProvider) Option {
	return func(a *Agent) { a.synthesizer = m }
}

// WithFinalizerModel overrides the model used to produce the final answer.
func WithFinalizerModel(m LLMProvider) Option {
	return func(a *Agent) { a.finalizer = m }
}

// WithMaxIterations sets the maximum loop iterations.
func WithMaxIterations(n int) Option {
	return func(a *Agent) {
		if n > 0 {
			a.maxIterations = n
		}
	}
}
