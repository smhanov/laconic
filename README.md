# laconic

A tiny Go library for building research agents that stay fast and cheap on small-context models.

## Why?

Most "ReAct" agents keep appending raw traces until the prompt overflows. Laconic instead compresses state at every step — either via a rolling scratchpad or a notebook of atomic facts — making it practical on 4k/8k models and low-cost backends.

## Features

- Two built-in research strategies: **Scratchpad** (iterative search loop) and **Graph Reader** (graph-based web exploration).
- Model-agnostic: bring your own `LLMProvider` adapter (OpenAI, Ollama, Anthropic, etc.).
- Swappable search providers (DuckDuckGo, Brave, Tavily) + custom `SearchProvider` interface.
- Optional `FetchProvider` for reading full web pages (used by Graph Reader).
- Dual-model support: use a stronger planner and a cheaper synthesizer/finalizer to save cost.
- Minimal dependencies (stdlib only, no vendor SDKs).
- Pluggable strategy system — register your own with `WithStrategyFactory`.

## Quick start

```bash
go get github.com/smhanov/laconic
```

Implement an `LLMProvider` adapter around your favorite client, then wire it up:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/smhanov/laconic"
    "github.com/smhanov/laconic/search"
)

func main() {
    agent := laconic.New(
        laconic.WithPlannerModel(myLLM),
        laconic.WithSynthesizerModel(myLLM),
        laconic.WithSearchProvider(search.NewDuckDuckGo()),
        laconic.WithMaxIterations(5),
    )

    ans, err := agent.Answer(context.Background(), "Why is the sky blue?")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(ans)
}
```

A minimal hardcoded example lives in `examples/basic/`. Run it with:

```bash
cd examples/basic
go run .
```

### CLI Demo

A fully functional CLI tool is available in `examples/research/`. It supports both Ollama (native API) and OpenAI-compatible backends, multiple search providers, and both strategies.

```bash
# Create a prompt file
echo "Why is the sky blue?" > question.txt

# Run with your local Ollama instance using the native API (default)
go run ./examples/research/ -model mistral -prompt question.txt

# Point to a remote Ollama endpoint
go run ./examples/research/ -model llama3 -endpoint ollama.example.com -prompt question.txt

# Use an Ollama server via its OpenAI-compatible endpoint
go run ./examples/research/ \
    -backend openai \
    -endpoint https://ollama.example.com \
    -model llama3 \
    -prompt question.txt

# Use the real OpenAI API
go run ./examples/research/ \
    -backend openai \
    -api-key $OPENAI_API_KEY \
    -model gpt-4o \
    -prompt question.txt

# Use the graph-reader strategy with Brave search
go run ./examples/research/ \
    -model llama3 \
    -prompt question.txt \
    -strategy graph-reader \
    -graph-max-steps 6 \
    -search brave \
    -brave-key YOUR_API_KEY

# Enable debug logging to see all LLM prompts and responses
go run ./examples/research/ -model mistral -prompt question.txt -debug
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `-backend` | `ollama` | LLM backend: `ollama` (native API) or `openai` (chat completions) |
| `-model` | *(required)* | Model name |
| `-endpoint` | varies | API endpoint URL (default: `localhost:11434` for ollama, `https://api.openai.com` for openai) |
| `-api-key` | | API key for authenticated endpoints (e.g. OpenAI) |
| `-prompt` | *(required)* | Path to a text file containing the question |
| `-strategy` | `scratchpad` | Strategy: `scratchpad` or `graph-reader` |
| `-max-iterations` | `5` | Maximum search iterations (scratchpad) |
| `-graph-max-steps` | `8` | Maximum exploration steps (graph-reader) |
| `-search` | `duckduckgo` | Search provider: `duckduckgo` or `brave` |
| `-brave-key` | | Brave Search API key (required with `-search brave`) |
| `-debug` | `false` | Print all LLM prompts and responses |

## Strategies

Laconic ships with two built-in strategies. Both compress state to stay within small context windows, but they differ fundamentally in how they plan, search, and accumulate knowledge.

### Scratchpad (default)

The scratchpad strategy runs a tight **Planner → Search → Synthesizer → Finalizer** loop.

**How it works:**

1. A `Scratchpad` is initialized with the user's question. It holds four fields: `OriginalQuestion`, `Knowledge` (a free-text summary), `History` (a log of past searches), and `IterationCount`.
2. Each iteration, the **Planner** LLM examines the scratchpad snapshot and emits one of two actions:
   - `Action: Answer` — enough information has been gathered.
   - `Action: Search` + `Query: <query>` — more information is needed.
3. The search provider executes the query and returns a list of results (title, URL, snippet).
4. The **Synthesizer** LLM receives the existing knowledge plus the new results and rewrites the `Knowledge` field as a concise, deduplicated summary. Raw search results are discarded — only the compressed summary survives.
5. This repeats until the Planner chooses `Answer` or `maxIterations` is reached.
6. The **Finalizer** LLM (which can be the same model or a separate, stronger one) turns the final knowledge state into a user-facing answer.

**Key properties:**

- **Flat context size.** Because the Synthesizer overwrites the knowledge field each iteration, prompt size stays bounded regardless of how many searches are performed. This makes it ideal for 4k/8k context models.
- **Grounding enforcement.** The planner is instructed to never answer from internal knowledge alone — at least one search must succeed before an answer is produced. If the planner tries to answer with an empty knowledge section, the agent forces a search automatically.
- **Simple mental model.** The loop is linear: plan, search, compress, repeat. There is no branching or backtracking.
- **Configurable iteration cap.** Set via `WithMaxIterations(n)`. Default is 5. If the cap is hit without a planner "Answer" decision, a best-effort finalization is returned alongside an error.

**When to choose scratchpad:**

- Simple, single-focus questions ("What is the population of Tokyo?").
- Environments with very small context windows (4k–8k tokens).
- When you want minimal LLM calls and fast answers (typically 2–4 calls total).
- When you don't have a `FetchProvider` and only need search snippets.
- As a lightweight, cheap default for most use cases.

```go
agent := laconic.New(
    laconic.WithPlannerModel(myLLM),
    laconic.WithSynthesizerModel(myLLM),
    laconic.WithSearchProvider(search.NewDuckDuckGo()),
    laconic.WithMaxIterations(5),
    laconic.WithStrategyName("scratchpad"), // this is the default
)
```

### Graph Reader

The graph-reader strategy implements a **graph-based exploration loop** inspired by the [GraphReader paper](https://arxiv.org/abs/2406.14550). Instead of a single rolling summary, it builds a **notebook of atomic facts** by traversing a dynamically constructed graph of search queries.

**How it works:**

1. The **Planner** LLM creates a *Rational Plan* — a structured breakdown of the question into a multi-step strategy and a list of *key elements* (entities, concepts, names) that need to be resolved.
2. From the plan, initial search queries ("nodes") are generated — typically 3–5 targeted queries covering the key elements.
3. The agent processes nodes from a queue in breadth-first order. For each node:
   - The search provider executes the query.
   - The **Extractor** LLM reads the search snippets and pulls out *atomic facts* — single, self-contained truths that directly help answer the question. Each fact is tagged with its source URL.
   - If snippets are promising but incomplete, the Extractor can flag URLs for deep reading. If a `FetchProvider` is configured, the agent fetches full page content and extracts additional facts from it.
   - Facts are deduplicated before being added to the notebook (exact matches and substring containment are both caught).
4. After processing each node, an **Answer Check** LLM evaluates whether the notebook contains enough facts to answer the question. If yes, exploration stops early.
5. A **Neighbor Selection** LLM examines the current notebook and suggests new search queries based on what has been learned and what gaps remain. These are added to the queue (skipping already-visited queries).
6. When exploration ends (either the answer check passes or `MaxSteps` is exhausted), the **Finalizer** LLM synthesizes a grounded answer from the notebook facts.

**Key properties:**

- **Structured fact accumulation.** Knowledge is stored as a list of discrete atomic facts with source URLs, not a free-text blob. This prevents important details from being summarized away and makes the final answer more traceable.
- **Dynamic exploration.** The graph expands based on what the agent learns — neighbor queries are generated from newly discovered facts, allowing the agent to follow chains of reasoning that weren't predictable upfront.
- **Deep reading.** When a `FetchProvider` is configured, the agent can follow promising URLs and extract facts from full page content, not just search snippets. Ad and tracker URLs are automatically filtered.
- **Early termination.** The answer check runs after each node, so the agent stops as soon as it has enough information — it won't waste calls exploring further if the notebook already covers the question.
- **Higher LLM cost.** Each node requires multiple LLM calls (extraction, answer check, neighbor selection), so this strategy uses significantly more LLM calls than scratchpad. A typical run with `MaxSteps: 6` might make 15–25 LLM calls.
- **Better for complex questions.** Multi-entity, multi-hop, and causal reasoning questions benefit from the structured plan and fact-by-fact accumulation.

**When to choose graph-reader:**

- Multi-hop questions that require chaining facts ("Who founded the company that acquired the maker of the drug used to treat X?").
- Questions involving multiple entities that each need separate research.
- When answer quality matters more than speed or cost.
- When you have a `FetchProvider` and want the agent to read full web pages for deeper evidence.
- When working with models that have 16k+ context windows (the notebook of facts can grow larger than a scratchpad summary).

```go
agent := laconic.New(
    laconic.WithPlannerModel(myLLM),
    laconic.WithSynthesizerModel(myLLM),
    laconic.WithSearchProvider(search.NewBrave(apiKey)),
    laconic.WithFetchProvider(fetch.NewHTTP()),
    laconic.WithStrategyName("graph-reader"),
    laconic.WithGraphReaderConfig(laconic.GraphReaderConfig{MaxSteps: 8}),
)
```

The `GraphReaderConfig` struct also lets you assign different LLM providers to each role if desired:

```go
laconic.WithGraphReaderConfig(laconic.GraphReaderConfig{
    Planner:   strongModel,   // generates the rational plan and initial queries
    Extractor: cheapModel,    // extracts atomic facts from search results / pages
    Neighbor:  cheapModel,    // suggests next queries to explore
    Finalizer: strongModel,   // writes the final answer
    MaxSteps:  10,
})
```

### Strategy comparison

| | Scratchpad | Graph Reader |
|---|---|---|
| **State format** | Free-text `Knowledge` summary | Notebook of atomic facts with source URLs |
| **Exploration** | Linear (one query at a time) | Graph-based (breadth-first with dynamic neighbors) |
| **Context growth** | Flat (summary is overwritten each step) | Grows with fact count (but stays structured) |
| **LLM calls per run** | ~2–4 (plan + synthesize + finalize) | ~15–25 (plan + extract × N + check × N + neighbors × N + finalize) |
| **Deep page reading** | No | Yes (via `FetchProvider`) |
| **Early termination** | Planner decides when to answer | Answer check evaluates notebook sufficiency |
| **Best for** | Simple factual questions, tight budgets | Multi-hop reasoning, complex research |
| **Min context window** | 4k tokens | 16k+ tokens recommended |
| **Requires FetchProvider** | No | No, but strongly recommended |

### Custom strategies

You can register your own strategy:

```go
agent := laconic.New(
    laconic.WithStrategyFactory("my-strategy", func(a *laconic.Agent) (laconic.Strategy, error) {
        return &myStrategy{}, nil
    }),
    laconic.WithStrategyName("my-strategy"),
)
```

A `Strategy` must implement `Name() string` and `Answer(ctx, question) (string, error)`.

## API surface

### Interfaces

- `LLMProvider` — your adapter for any language model. Single method: `Generate(ctx, systemPrompt, userPrompt) (string, error)`.
- `SearchProvider` — plug any search backend. Single method: `Search(ctx, query) ([]SearchResult, error)`.
- `FetchProvider` — optional URL fetcher for reading full web pages. Single method: `Fetch(ctx, url) (string, error)`.
- `Strategy` — pluggable research loop. Methods: `Name() string`, `Answer(ctx, question) (string, error)`.

### Agent

Create with `laconic.New(opts...)`, then call `agent.Answer(ctx, question)`.

### Functional options

| Option | Description |
|--------|-------------|
| `WithPlannerModel(m)` | LLM used for routing/planning decisions |
| `WithSynthesizerModel(m)` | LLM used for compressing search results |
| `WithFinalizerModel(m)` | LLM used to produce the final answer (defaults to synthesizer) |
| `WithSearchProvider(s)` | Search backend implementation |
| `WithFetchProvider(f)` | URL fetcher for full-page reading (optional) |
| `WithMaxIterations(n)` | Max loop iterations for scratchpad strategy (default: 5) |
| `WithStrategyName(name)` | Select a strategy by name: `"scratchpad"` or `"graph-reader"` |
| `WithStrategy(s)` | Inject a custom `Strategy` instance directly |
| `WithStrategyFactory(name, fn)` | Register a custom strategy factory |
| `WithGraphReaderConfig(cfg)` | Configure the graph-reader strategy (MaxSteps, per-role LLMs) |
| `WithDebug(bool)` | Log all LLM prompts and responses to stdout |

## Search providers

| Provider | API key required | Notes |
|----------|-----------------|-------|
| DuckDuckGo | No | Free; scrapes the lite HTML interface |
| Brave | Yes (`X-Subscription-Token`) | Fast, structured JSON API |
| Tavily | Yes | Supports `basic` and `advanced` depth modes |

```go
search.NewDuckDuckGo()
search.NewBrave("your-api-key")
search.NewTavily("your-api-key", "advanced")
```

Bring your own provider by implementing `SearchProvider`.

## Architecture highlights

- **Scratchpad** keeps `OriginalQuestion`, `Knowledge`, `History`, and `IterationCount` small and bounded.
- **Graph Reader** maintains a `Notebook` of `AtomicFact` entries and a queue of `Node` queries with visited-set tracking.
- **Planner** decides the next action with a compact prompt.
- **Synthesizer / Extractor** compresses raw search results into the state representation.
- **Finalizer** writes the user-facing answer from the accumulated knowledge.
- **`<think>` block stripping** — models like Qwen3 that emit `<think>...</think>` reasoning blocks are handled transparently.

See detailed design in [docs/architecture.md](docs/architecture.md) and prompt shapes in [docs/prompts.md](docs/prompts.md).

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
