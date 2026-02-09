package laconic

import "context"

// Strategy defines a configurable research loop.
type Strategy interface {
	Name() string
	Answer(ctx context.Context, question string) (Result, error)
}

// StrategyFactory creates a strategy using the Agent's configured dependencies.
type StrategyFactory func(a *Agent) (Strategy, error)
