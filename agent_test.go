package laconic

import (
	"context"
	"errors"
	"testing"
)

type scriptedLLM struct {
	planner []string
	synth   []string
	final   []string

	plannerIdx int
	synthIdx   int
	finalIdx   int
}

func (s *scriptedLLM) next(list []string, idx *int) (string, error) {
	if *idx >= len(list) {
		return "", errors.New("no scripted response available")
	}
	resp := list[*idx]
	*idx = *idx + 1
	return resp, nil
}

func (s *scriptedLLM) Generate(_ context.Context, systemPrompt, _ string) (string, error) {
	switch systemPrompt {
	case plannerSystemPrompt:
		return s.next(s.planner, &s.plannerIdx)
	case synthesizerSystemPrompt:
		return s.next(s.synth, &s.synthIdx)
	case finalizerSystemPrompt:
		return s.next(s.final, &s.finalIdx)
	default:
		return "", errors.New("unknown system prompt")
	}
}

type fakeSearch struct{ results []SearchResult }

func (f fakeSearch) Search(_ context.Context, _ string) ([]SearchResult, error) {
	return f.results, nil
}

func TestAgentSearchThenAnswer(t *testing.T) {
	llm := &scriptedLLM{
		planner: []string{"Action: Search\nQuery: optical depth", "Action: Answer"},
		synth:   []string{"Blue sky due to Rayleigh scattering"},
		final:   []string{"Rayleigh scattering explains blue skies."},
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "Sky color", URL: "https://example.com", Snippet: "Rayleigh scattering"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(3),
	)

	got, err := agent.Answer(context.Background(), "Why is the sky blue?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == "" {
		t.Fatal("expected non-empty answer")
	}
}

func TestAgentMaxIterationsBestEffort(t *testing.T) {
	llm := &scriptedLLM{
		planner: []string{"Action: Search\nQuery: retry", "Action: Search\nQuery: retry", "Action: Search\nQuery: retry"},
		synth:   []string{"k1", "k2", "k3"},
		final:   []string{"best effort"},
	}
	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(2),
	)

	got, err := agent.Answer(context.Background(), "Q")
	if err == nil {
		t.Fatalf("expected best-effort error, got nil")
	}
	if got == "" {
		t.Fatalf("expected best-effort answer text")
	}
}
