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

func (s *scratchpadStrategy) Answer(ctx context.Context, question string) (Result, error) {
	return s.agent.answerScratchpad(ctx, question)
}

func (a *Agent) answerScratchpad(ctx context.Context, question string) (Result, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return Result{}, errors.New("question is empty")
	}
	if a.planner == nil {
		return Result{}, errors.New("planner model is not configured")
	}
	if a.synthesizer == nil {
		return Result{}, errors.New("synthesizer model is not configured")
	}

	pad := NewScratchpad(question)
	var totalCost float64

	for i := 0; i < a.maxIterations; i++ {
		pad.IterationCount = i + 1

		decision, cost, err := a.plan(ctx, pad)
		totalCost += cost
		if err != nil {
			return Result{}, fmt.Errorf("planner: %w", err)
		}

		switch decision.Action {
		case PlannerActionAnswer:
			// Enforce grounding: must have searched at least once before answering
			if strings.TrimSpace(pad.Knowledge) == "" {
				// Force a search if no knowledge has been gathered yet
				if a.searcher == nil {
					return Result{}, errors.New("cannot answer without search: no search provider configured")
				}
				// Use the question as the search query
				results, err := a.searcher.Search(ctx, question)
				if err != nil {
					return Result{}, fmt.Errorf("search: %w", err)
				}
				totalCost += a.searchCost
				pad.AppendHistory(fmt.Sprintf("search[%d]: %s (forced)", pad.IterationCount, question))
				synthCost, err := a.synthesize(ctx, &pad, question, results)
				totalCost += synthCost
				if err != nil {
					return Result{}, fmt.Errorf("synthesizer: %w", err)
				}
				continue // Re-evaluate after forced search
			}
			answer, finCost, err := a.finalize(ctx, pad)
			totalCost += finCost
			if err != nil {
				return Result{}, err
			}
			return Result{Answer: answer, Cost: totalCost}, nil
		case PlannerActionSearch:
			if a.searcher == nil {
				return Result{}, errors.New("search requested but no search provider configured")
			}
			results, err := a.searcher.Search(ctx, decision.Query)
			if err != nil {
				return Result{}, fmt.Errorf("search: %w", err)
			}
			totalCost += a.searchCost
			pad.AppendHistory(fmt.Sprintf("search[%d]: %s", pad.IterationCount, decision.Query))
			synthCost, err := a.synthesize(ctx, &pad, decision.Query, results)
			totalCost += synthCost
			if err != nil {
				return Result{}, fmt.Errorf("synthesizer: %w", err)
			}
		default:
			return Result{}, fmt.Errorf("unknown planner action: %s", decision.Action)
		}
	}

	// Best-effort finalization even if the planner never said "Answer".
	final, finCost, err := a.finalize(ctx, pad)
	totalCost += finCost
	if err != nil {
		return Result{}, fmt.Errorf("max iterations reached without answer: %w", err)
	}
	return Result{Answer: final, Cost: totalCost}, errors.New("max iterations reached; returning best-effort answer")
}
