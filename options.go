package laconic

const defaultMaxIterations = 5
const defaultGraphReaderSteps = 8

// Option configures an Agent.
type Option func(*Agent)

// WithSearchProvider sets the search implementation.
func WithSearchProvider(searcher SearchProvider) Option {
	return func(a *Agent) { a.searcher = searcher }
}

// WithFetchProvider sets the optional fetch implementation.
func WithFetchProvider(fetcher FetchProvider) Option {
	return func(a *Agent) { a.fetcher = fetcher }
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

// WithDebug enables debug logging of all LLM prompts and responses.
func WithDebug(enabled bool) Option {
	return func(a *Agent) { a.debug = enabled }
}

// WithStrategy sets a custom strategy instance.
func WithStrategy(strategy Strategy) Option {
	return func(a *Agent) { a.strategy = strategy }
}

// WithStrategyName selects a built-in or registered strategy by name.
func WithStrategyName(name string) Option {
	return func(a *Agent) { a.strategyName = name }
}

// WithStrategyFactory registers a strategy factory by name.
func WithStrategyFactory(name string, factory StrategyFactory) Option {
	return func(a *Agent) {
		if a.strategyFactories == nil {
			a.strategyFactories = make(map[string]StrategyFactory)
		}
		a.strategyFactories[name] = factory
	}
}

// GraphReaderConfig configures the GraphReader strategy.
type GraphReaderConfig struct {
	Planner   LLMProvider
	Extractor LLMProvider
	Neighbor  LLMProvider
	Finalizer LLMProvider
	MaxSteps  int
}

// WithGraphReaderConfig customizes the built-in GraphReader strategy.
func WithGraphReaderConfig(cfg GraphReaderConfig) Option {
	return func(a *Agent) { a.graphReaderConfig = cfg }
}
