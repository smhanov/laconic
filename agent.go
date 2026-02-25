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
	searchCost        float64
	priorKnowledge    string // set per-call via AnswerOption
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
// Optional AnswerOption values can supply prior knowledge for follow-up
// questions (see WithKnowledge).
func (a *Agent) Answer(ctx context.Context, question string, opts ...AnswerOption) (Result, error) {
	var cfg answerConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	a.priorKnowledge = cfg.priorKnowledge
	defer func() { a.priorKnowledge = "" }()

	strategy, err := a.resolveStrategy()
	if err != nil {
		return Result{}, err
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

func (a *Agent) plan(ctx context.Context, pad Scratchpad) (PlannerDecision, float64, error) {
	sys := plannerSystemPrompt
	user := buildPlannerUserPrompt(pad)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Planner System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Planner User Prompt:\n%s\n", user)
	}
	resp, err := a.planner.Generate(ctx, sys, user)
	if err != nil {
		return PlannerDecision{}, 0, err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Planner Response:\n%s\n", resp.Text)
	}
	// Strip <think> blocks from models like qwen3; fall back to reasoning if text is empty.
	raw := getContent(resp, a.debug, "Planner")
	decision, err := parsePlannerDecision(raw)
	return decision, resp.Cost, err
}

func (a *Agent) synthesize(ctx context.Context, pad *Scratchpad, query string, results []SearchResult) (float64, error) {
	sys := synthesizerSystemPrompt
	user := buildSynthesizerUserPrompt(*pad, query, results)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Synthesizer System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Synthesizer User Prompt:\n%s\n", user)
	}
	resp, err := a.synthesizer.Generate(ctx, sys, user)
	if err != nil {
		return 0, err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Synthesizer Response:\n%s\n", resp.Text)
	}
	// Strip <think> blocks from models like qwen3; fall back to reasoning if text is empty.
	pad.Knowledge = getContent(resp, a.debug, "Synthesizer")
	pad.CurrentStep = fmt.Sprintf("Last query: %s", query)
	return resp.Cost, nil
}

func (a *Agent) finalize(ctx context.Context, pad Scratchpad) (string, float64, error) {
	if a.finalizer == nil {
		return "", 0, errors.New("finalizer model is not configured")
	}
	sys := finalizerSystemPrompt
	user := buildFinalizerUserPrompt(pad)
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer System Prompt:\n%s\n", sys)
		fmt.Printf("[LACONIC DEBUG] Finalizer User Prompt:\n%s\n", user)
	}
	resp, err := a.finalizer.Generate(ctx, sys, user)
	if err != nil {
		return "", 0, err
	}
	if a.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer Response:\n%s\n", resp.Text)
	}
	// Strip <think> blocks from models like qwen3; fall back to reasoning if text is empty.
	return getContent(resp, a.debug, "Finalizer"), resp.Cost, nil
}
