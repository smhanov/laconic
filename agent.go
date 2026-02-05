package laconic

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Agent coordinates the planner, searcher, synthesizer, and finalizer.
type Agent struct {
	searcher      SearchProvider
	planner       LLMProvider
	synthesizer   LLMProvider
	finalizer     LLMProvider
	maxIterations int
}

// New constructs an Agent with optional configuration.
func New(opts ...Option) *Agent {
	a := &Agent{maxIterations: defaultMaxIterations}
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
	question = strings.TrimSpace(question)
	if question == "" {
		return "", errors.New("question is empty")
	}
	if a.planner == nil {
		return "", errors.New("planner model is not configured")
	}
	if a.synthesizer == nil {
		return "", errors.New("synthesizer model is not configured")
	}

	pad := NewScratchpad(question)

	for i := 0; i < a.maxIterations; i++ {
		pad.IterationCount = i + 1

		decision, err := a.plan(ctx, pad)
		if err != nil {
			return "", fmt.Errorf("planner: %w", err)
		}

		switch decision.Action {
		case PlannerActionAnswer:
			// Enforce grounding: must have searched at least once before answering
			if strings.TrimSpace(pad.Knowledge) == "" {
				// Force a search if no knowledge has been gathered yet
				if a.searcher == nil {
					return "", errors.New("cannot answer without search: no search provider configured")
				}
				// Use the question as the search query
				results, err := a.searcher.Search(ctx, question)
				if err != nil {
					return "", fmt.Errorf("search: %w", err)
				}
				pad.AppendHistory(fmt.Sprintf("search[%d]: %s (forced)", pad.IterationCount, question))
				err = a.synthesize(ctx, &pad, question, results)
				if err != nil {
					return "", fmt.Errorf("synthesizer: %w", err)
				}
				continue // Re-evaluate after forced search
			}
			return a.finalize(ctx, pad)
		case PlannerActionSearch:
			if a.searcher == nil {
				return "", errors.New("search requested but no search provider configured")
			}
			results, err := a.searcher.Search(ctx, decision.Query)
			if err != nil {
				return "", fmt.Errorf("search: %w", err)
			}
			pad.AppendHistory(fmt.Sprintf("search[%d]: %s", pad.IterationCount, decision.Query))
			err = a.synthesize(ctx, &pad, decision.Query, results)
			if err != nil {
				return "", fmt.Errorf("synthesizer: %w", err)
			}
		default:
			return "", fmt.Errorf("unknown planner action: %s", decision.Action)
		}
	}

	// Best-effort finalization even if the planner never said "Answer".
	final, err := a.finalize(ctx, pad)
	if err != nil {
		return "", fmt.Errorf("max iterations reached without answer: %w", err)
	}
	return final, errors.New("max iterations reached; returning best-effort answer")
}

func (a *Agent) plan(ctx context.Context, pad Scratchpad) (PlannerDecision, error) {
	sys := plannerSystemPrompt
	user := buildPlannerUserPrompt(pad)
	raw, err := a.planner.Generate(ctx, sys, user)
	if err != nil {
		return PlannerDecision{}, err
	}
	// Strip <think> blocks from models like qwen3
	raw = StripThinkBlocks(raw)
	return parsePlannerDecision(raw)
}

func (a *Agent) synthesize(ctx context.Context, pad *Scratchpad, query string, results []SearchResult) error {
	sys := synthesizerSystemPrompt
	user := buildSynthesizerUserPrompt(*pad, query, results)
	raw, err := a.synthesizer.Generate(ctx, sys, user)
	if err != nil {
		return err
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
	out, err := a.finalizer.Generate(ctx, sys, user)
	if err != nil {
		return "", err
	}
	// Strip <think> blocks from models like qwen3
	return StripThinkBlocks(out), nil
}
