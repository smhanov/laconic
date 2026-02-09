// Package laconic provides a lightweight research agent framework that uses
// a scratchpad-based architecture to keep prompts compact and efficient.
//
// Unlike traditional ReAct agents that append raw traces until context overflows,
// laconic maintains a compressed Scratchpad state and summarizes each iteration,
// making it practical on 4k/8k context models and low-cost LLM backends.
//
// # Architecture
//
// The agent operates in a Planner → Search → Synthesizer → Finalizer loop
// for the default scratchpad strategy:
//
//  1. The Planner examines the scratchpad and decides whether to answer or search.
//  2. If searching, the SearchProvider executes the query.
//  3. The Synthesizer compresses results into the scratchpad's knowledge state.
//  4. When ready (or max iterations reached), the Finalizer produces the answer.
//
// # Cost Tracking
//
// Every LLM call and search call can report a cost. LLMProvider.Generate returns
// an LLMResponse containing both text and cost. Search costs are configured via
// WithSearchCost. Agent.Answer returns a Result struct with the final answer and
// the total accumulated cost.
//
// # Basic Usage
//
//	agent := laconic.New(
//	    laconic.WithPlannerModel(myLLM),
//	    laconic.WithSynthesizerModel(myLLM),
//	    laconic.WithSearchProvider(search.NewDuckDuckGo()),
//	    laconic.WithSearchCost(0.005),
//	    laconic.WithMaxIterations(5),
//	)
//
//	result, err := agent.Answer(ctx, "What is the capital of France?")
//	fmt.Println(result.Answer)
//	fmt.Printf("Cost: $%.4f\n", result.Cost)
//
// # Interfaces
//
// Implement LLMProvider to connect any language model:
//
//	type LLMProvider interface {
//	    Generate(ctx context.Context, systemPrompt, userPrompt string) (LLMResponse, error)
//	}
//
// LLMResponse carries the generated text and the cost of the call:
//
//	type LLMResponse struct {
//	    Text string
//	    Cost float64
//	}
//
// Implement SearchProvider to use any search backend:
//
//	type SearchProvider interface {
//	    Search(ctx context.Context, query string) ([]SearchResult, error)
//	}
//
// Strategy options include the default scratchpad loop and the graph-based
// "graph-reader" strategy.
// See the examples/basic directory for a complete example.
package laconic
