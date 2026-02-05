package laconic

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Agent coordinates the planner, searcher, synthesizer, and finalizer.
type Agent struct {
	searcher          SearchProvider
	fetcher           FetchProvider
	planner           LLMProvider
	synthesizer       LLMProvider
	finalizer         LLMProvider
	maxIterations     int
	debug             bool
	strategy          Strategy
	strategyName      string
	strategyFactories map[string]StrategyFactory
	graphReaderConfig GraphReaderConfig
}

// New constructs an Agent with optional configuration.
func New(opts ...Option) *Agent {
	a := &Agent{
		maxIterations: defaultMaxIterations,
		strategyName:  "scratchpad",
		strategyFactories: map[string]StrategyFactory{
			"scratchpad":   newScratchpadStrategy,
			"graph-reader": newGraphReaderStrategy,
		},
	}
	for _, opt := range opts {
		opt(a)
	}
	if a.finalizer == nil {
		a.finalizer = a.synthesizer
	}
	return a
}

// Answer runs the loop until an answer is produced or the limit is reached.
func (a *Agent) Answer(ctx context.Context, question string) (string, error) {
	strategy, err := a.resolveStrategy()
	if err != nil {
		return "", err
	}
	return strategy.Answer(ctx, question)
}

func (a *Agent) resolveStrategy() (Strategy, error) {
	if a.strategy != nil {
		return a.strategy, nil
	}
	name := strings.TrimSpace(a.strategyName)
	if name == "" {
		name = "scratchpad"
	}
	factory := a.strategyFactories[name]
	if factory == nil {
		return nil, fmt.Errorf("unknown strategy: %s", name)
	}
	strategy, err := factory(a)
	if err != nil {
		return nil, err
	}
	a.strategy = strategy
	return strategy, nil
}

func (a *Agent) plan(ctx context.Context, pad Scratchpad) (PlannerDecision, error) {
	sys := plannerSystemPrompt
	user := buildPlannerUserPrompt(pad)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Planner System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Planner User Prompt:\n%s\n", user)
	}
	raw, err := a.planner.Generate(ctx, sys, user)
	if err != nil {
		return PlannerDecision{}, err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Planner Response:\n%s\n", raw)
	}
	// Strip <think> blocks from models like qwen3
	raw = StripThinkBlocks(raw)
	return parsePlannerDecision(raw)
}

func (a *Agent) synthesize(ctx context.Context, pad *Scratchpad, query string, results []SearchResult) error {
	sys := synthesizerSystemPrompt
	user := buildSynthesizerUserPrompt(*pad, query, results)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Synthesizer System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Synthesizer User Prompt:\n%s\n", user)
	}
	raw, err := a.synthesizer.Generate(ctx, sys, user)
	if err != nil {
		return err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Synthesizer Response:\n%s\n", raw)
	}
	// Strip <think> blocks from models like qwen3
	pad.Knowledge = StripThinkBlocks(raw)
	pad.CurrentStep = fmt.Sprintf("Last query: %s", query)
	return nil
}

func (a *Agent) finalize(ctx context.Context, pad Scratchpad) (string, error) {
	if a.finalizer == nil {
		return "", errors.New("finalizer model is not configured")
	}
	sys := finalizerSystemPrompt
	user := buildFinalizerUserPrompt(pad)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Finalizer User Prompt:\n%s\n", user)
	}
	out, err := a.finalizer.Generate(ctx, sys, user)
	if err != nil {
		return "", err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer Response:\n%s\n", out)
	}
	// Strip <think> blocks from models like qwen3
	return StripThinkBlocks(out), nil
}
