package main

import (
	"context"
	"fmt"
	"log"

	"github.com/smhanov/laconic"
	"github.com/smhanov/laconic/search"
)

// demoLLM is a trivial scripted model. Replace with your own provider.
type demoLLM struct{}

func (demoLLM) Generate(_ context.Context, systemPrompt, userPrompt string) (laconic.LLMResponse, error) {
	switch systemPrompt {
	case "You are a focused research planner. Decide whether to answer or search.":
		return laconic.LLMResponse{Text: "Action: Search\nQuery: why is the sky blue", Cost: 0.001}, nil
	case "You compress search findings into a concise knowledge state. Keep only facts that help answer the question.":
		return laconic.LLMResponse{Text: "Rayleigh scattering makes the sky appear blue to human eyes.", Cost: 0.001}, nil
	default:
		return laconic.LLMResponse{Text: "The sky is blue because shorter wavelengths scatter more in the atmosphere.", Cost: 0.002}, nil
	}
}

func main() {
	agent := laconic.New(
		laconic.WithPlannerModel(demoLLM{}),
		laconic.WithSynthesizerModel(demoLLM{}),
		laconic.WithSearchProvider(search.NewDuckDuckGo()),
		laconic.WithSearchCost(0.005), // $0.005 per search call
		laconic.WithMaxIterations(3),
	)

	result, err := agent.Answer(context.Background(), "Why is the sky blue?")
	if err != nil {
		log.Printf("best-effort error: %v", err)
	}
	fmt.Println(result.Answer)
	fmt.Printf("Total cost: $%.4f\n", result.Cost)
}
