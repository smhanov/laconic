package laconic

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

type scratchpadStrategy struct {
	agent *Agent
}

func newScratchpadStrategy(a *Agent) (Strategy, error) {
	return &scratchpadStrategy{agent: a}, nil
}

func (s *scratchpadStrategy) Name() string {
	return "scratchpad"
}

func (s *scratchpadStrategy) Answer(ctx context.Context, question string) (string, error) {
	return s.agent.answerScratchpad(ctx, question)
}

func (a *Agent) answerScratchpad(ctx context.Context, question string) (string, error) {
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
