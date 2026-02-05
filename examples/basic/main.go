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

func (demoLLM) Generate(_ context.Context, systemPrompt, userPrompt string) (string, error) {
	switch systemPrompt {
	case "You are a focused research planner. Decide whether to answer or search.":
		return "Action: Search\nQuery: why is the sky blue", nil
	case "You compress search findings into a concise knowledge state. Keep only facts that help answer the question.":
		return "Rayleigh scattering makes the sky appear blue to human eyes.", nil
	default:
		return "The sky is blue because shorter wavelengths scatter more in the atmosphere.", nil
	}
}

func main() {
	agent := laconic.New(
		laconic.WithPlannerModel(demoLLM{}),
		laconic.WithSynthesizerModel(demoLLM{}),
		laconic.WithSearchProvider(search.NewDuckDuckGo()),
		laconic.WithMaxIterations(3),
	)

	ans, err := agent.Answer(context.Background(), "Why is the sky blue?")
	if err != nil {
		log.Printf("best-effort error: %v", err)
	}
	fmt.Println(ans)
}
