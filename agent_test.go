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

	costPerCall float64
}

func (s *scriptedLLM) next(list []string, idx *int) (string, error) {
	if *idx >= len(list) {
		return "", errors.New("no scripted response available")
	}
	resp := list[*idx]
	*idx = *idx + 1
	return resp, nil
}

func (s *scriptedLLM) Generate(_ context.Context, systemPrompt, _ string) (LLMResponse, error) {
	var text string
	var err error
	switch systemPrompt {
	case plannerSystemPrompt:
		text, err = s.next(s.planner, &s.plannerIdx)
	case synthesizerSystemPrompt:
		text, err = s.next(s.synth, &s.synthIdx)
	case finalizerSystemPrompt:
		text, err = s.next(s.final, &s.finalIdx)
	default:
		return LLMResponse{}, errors.New("unknown system prompt")
	}
	if err != nil {
		return LLMResponse{}, err
	}
	return LLMResponse{Text: text, Cost: s.costPerCall}, nil
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

	res, err := agent.Answer(context.Background(), "Why is the sky blue?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Answer == "" {
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

	res, err := agent.Answer(context.Background(), "Q")
	if err == nil {
		t.Fatalf("expected best-effort error, got nil")
	}
	if res.Answer == "" {
		t.Fatalf("expected best-effort answer text")
	}
}

func TestAgentCostTracking(t *testing.T) {
	llm := &scriptedLLM{
		planner:     []string{"Action: Search\nQuery: test query", "Action: Answer"},
		synth:       []string{"some knowledge"},
		final:       []string{"final answer"},
		costPerCall: 0.01,
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithSearchCost(0.005),
		WithMaxIterations(3),
	)

	res, err := agent.Answer(context.Background(), "Test question")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Answer == "" {
		t.Fatal("expected non-empty answer")
	}
	// Expected cost: planner(0.01) + search(0.005) + synth(0.01) + planner(0.01) + finalizer(0.01) = 0.045
	expectedCost := 0.045
	if res.Cost < expectedCost-0.001 || res.Cost > expectedCost+0.001 {
		t.Fatalf("expected cost ~%.3f, got %.3f", expectedCost, res.Cost)
	}
}

func TestAgentZeroCostByDefault(t *testing.T) {
	llm := &scriptedLLM{
		planner: []string{"Action: Search\nQuery: test", "Action: Answer"},
		synth:   []string{"knowledge"},
		final:   []string{"answer"},
		// costPerCall defaults to 0
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(3),
	)

	res, err := agent.Answer(context.Background(), "Q")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Cost != 0 {
		t.Fatalf("expected zero cost by default, got %f", res.Cost)
	}
}

func TestResultKnowledge(t *testing.T) {
	llm := &scriptedLLM{
		planner: []string{"Action: Search\nQuery: test query", "Action: Answer"},
		synth:   []string{"synthesized knowledge about the topic"},
		final:   []string{"final answer"},
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(3),
	)

	res, err := agent.Answer(context.Background(), "Test question")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Knowledge == "" {
		t.Fatal("expected non-empty Knowledge in result")
	}
	if res.Knowledge != "synthesized knowledge about the topic" {
		t.Fatalf("unexpected Knowledge: %q", res.Knowledge)
	}
}

func TestPriorKnowledge(t *testing.T) {
	// The planner sees non-empty knowledge and decides to answer immediately.
	llm := &scriptedLLM{
		planner: []string{"Action: Answer"},
		final:   []string{"follow-up answer using prior knowledge"},
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(3),
	)

	res, err := agent.Answer(context.Background(), "Follow-up question",
		WithKnowledge("previously collected knowledge"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Answer == "" {
		t.Fatal("expected non-empty answer")
	}
	// Knowledge should contain the prior knowledge (possibly updated)
	if res.Knowledge == "" {
		t.Fatal("expected Knowledge to be preserved in result")
	}
}

func TestPriorKnowledgeCleared(t *testing.T) {
	// Verify that prior knowledge from one call does not leak into the next.
	llm := &scriptedLLM{
		planner: []string{
			"Action: Answer",          // first call (with prior knowledge)
			"Action: Search\nQuery: q", // second call (without)
			"Action: Answer",
		},
		synth: []string{"new knowledge"},
		final: []string{"answer1", "answer2"},
	}

	searcher := fakeSearch{results: []SearchResult{{Title: "t", URL: "u", Snippet: "s"}}}

	agent := New(
		WithPlannerModel(llm),
		WithSynthesizerModel(llm),
		WithSearchProvider(searcher),
		WithMaxIterations(3),
	)

	// First call with prior knowledge — planner sees non-empty knowledge, answers.
	res1, err := agent.Answer(context.Background(), "Q1",
		WithKnowledge("prior stuff"),
	)
	if err != nil {
		t.Fatalf("call 1: unexpected error: %v", err)
	}
	if res1.Knowledge != "prior stuff" {
		t.Fatalf("call 1: expected prior knowledge preserved, got %q", res1.Knowledge)
	}

	// Second call without prior knowledge — agent must search.
	res2, err := agent.Answer(context.Background(), "Q2")
	if err != nil {
		t.Fatalf("call 2: unexpected error: %v", err)
	}
	if res2.Knowledge != "new knowledge" {
		t.Fatalf("call 2: expected fresh knowledge, got %q", res2.Knowledge)
	}
}
