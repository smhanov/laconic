// Package laconic provides a lightweight research agent framework that uses
// a scratchpad-based architecture to keep prompts compact and efficient.
//
// Unlike traditional ReAct agents that append raw traces until context overflows,
// laconic maintains a compressed Scratchpad state and summarizes each iteration,
// making it practical on 4k/8k context models and low-cost LLM backends.
//
// # Architecture
//
// The agent operates in a Planner → Search → Synthesizer → Finalizer loop:
//
//  1. The Planner examines the scratchpad and decides whether to answer or search.
//  2. If searching, the SearchProvider executes the query.
//  3. The Synthesizer compresses results into the scratchpad's knowledge state.
//  4. When ready (or max iterations reached), the Finalizer produces the answer.
//
// # Basic Usage
//
//	agent := laconic.New(
//	    laconic.WithPlannerModel(myLLM),
//	    laconic.WithSynthesizerModel(myLLM),
//	    laconic.WithSearchProvider(search.NewDuckDuckGo()),
//	    laconic.WithMaxIterations(5),
//	)
//
//	answer, err := agent.Answer(ctx, "What is the capital of France?")
//
// # Interfaces
//
// Implement LLMProvider to connect any language model:
//
//	type LLMProvider interface {
//	    Generate(ctx context.Context, systemPrompt, userPrompt string) (string, error)
//	}
//
// Implement SearchProvider to use any search backend:
//
//	type SearchProvider interface {
//	    Search(ctx context.Context, query string) ([]SearchResult, error)
//	}
//
// See the examples/ollama directory for a complete CLI example using Ollama.
package laconic
