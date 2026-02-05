# laconic

A tiny Go library for building scratchpad-style research agents that stay fast and cheap on small-context models.

## Why?

Most "ReAct" agents keep appending raw traces until the prompt overflows. Laconic instead keeps a compact Scratchpad and summarizes each step, making it practical on 4k/8k models and low-cost backends.

## Features

- Stateful scratchpad loop (Planner → Search → Synthesizer → Answer).
- Model-agnostic: bring your own `LLMProvider` adapter.
- Swappable search providers (DuckDuckGo, Brave, Tavily) + custom interface.
- Dual-model setup: higher-quality planner, cheaper synthesizer/finalizer.
- Minimal dependencies (stdlib only).

## Quick start

```bash
go get github.com/smhanov/laconic
```

Implement an `LLMProvider` adapter around your favorite client (OpenAI, Ollama, Anthropic, etc.).

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/smhanov/laconic"
    "github.com/smhanov/laconic/search"
)

// toyLLM echoes deterministic strings; swap with your real model.
type toyLLM struct{}

func (toyLLM) Generate(_ context.Context, system, user string) (string, error) {
    if system == "You are a focused research planner. Decide whether to answer or search." {
        if len(user) > 0 {
            return "Action: Search\nQuery: why is the sky blue", nil
        }
    }
    if system == "You compress search findings into a concise knowledge state. Keep only facts that help answer the question." {
        return "The sky looks blue because shorter wavelengths scatter more.", nil
    }
    return "Rayleigh scattering makes the sky blue.", nil
}

func main() {
    agent := laconic.New(
        laconic.WithPlannerModel(toyLLM{}),
        laconic.WithSynthesizerModel(toyLLM{}),
        laconic.WithSearchProvider(search.NewDuckDuckGo()),
        laconic.WithMaxIterations(3),
    )

    ans, err := agent.Answer(context.Background(), "Why is the sky blue?")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(ans)
}
```

Run the example in this repo:

```bash
cd examples/basic
go run .
```

### Ollama CLI Demo

A fully functional CLI tool that uses Ollama and DuckDuckGo search is available in `examples/ollama`.

```bash
# Create a prompt file
echo "Why is the sky blue?" > question.txt

# Run with your local Ollama instance (default: localhost:11434)
go run ./examples/ollama/ -model mistral -prompt question.txt

# Or point to a remote endpoint
go run ./examples/ollama/ -model llama3 -endpoint ollama.example.com -prompt question.txt
```

## API surface

- `SearchProvider` interface: plug any search backend.
- `LLMProvider` interface: your adapter for any model.
- `Agent` with `Answer(ctx, question)` entrypoint.
- Functional options: `WithSearchProvider`, `WithPlannerModel`, `WithSynthesizerModel`, `WithFinalizerModel`, `WithMaxIterations`.

## Architecture highlights

- **Scratchpad** keeps `OriginalQuestion`, `Knowledge`, `History`, and `IterationCount` small and bounded.
- **Planner** decides `Answer` vs `Search` with a compact prompt.
- **Synthesizer** compresses search results back into the scratchpad.
- **Finalizer** writes the user-facing answer (can reuse synthesizer or be separate).

See detailed design in [docs/architecture.md](docs/architecture.md) and prompt shapes in [docs/prompts.md](docs/prompts.md).

## Search providers

- DuckDuckGo: free, no API key required (uses HTML scraping).
- Brave: requires `X-Subscription-Token` header.
- Tavily: requires API key; supports `basic`/`advanced` depth.

You can always bring your own provider by implementing `Search(ctx, query) ([]SearchResult, error)`.

## Testing

```bash
go test ./...
```

Tests use fully offline stubs; no API calls are made.

## Roadmap

- Add caching layer to avoid duplicate searches.
- Extend planner to accept tool choices beyond search.
- Provide optional streaming-friendly interfaces.

## License

MIT. See [LICENSE](LICENSE).
