# Architecture

Laconic implements a small, loop-based agent focused on keeping prompts compact.

## Components

- `Agent`: orchestrates the Planner → Search → Synthesizer → Finalizer loop.
- `Scratchpad`: mutable state with `OriginalQuestion`, `Knowledge`, `History`, `CurrentStep`, `IterationCount`.
- `SearchProvider`: user-swappable search backend interface.
- `FetchProvider`: optional URL fetcher for strategies that read full pages.
- `LLMProvider`: user-supplied model wrapper; no vendor SDKs are pulled in. Returns `LLMResponse` with text and cost.
- `Strategy`: pluggable research loop implementations (default: scratchpad, optional: graph-reader).
- `Result`: returned by `Agent.Answer`, carries the answer text and accumulated cost.

## Cost tracking

Every LLM call returns an `LLMResponse{Text, Cost}`. The strategy accumulates all LLM costs and search costs (configured via `WithSearchCost`) into the `Result.Cost` field. When the cost is not relevant, providers can return `Cost: 0` and the total will simply be zero.

## Loop flow

1. Initialize Scratchpad from the user question.
2. Planner reads the snapshot and decides between `Action: Answer` or `Action: Search` plus a query.
3. If `Search`, the SearchProvider executes the query; raw results are passed to the Synthesizer.
4. Synthesizer rewrites the Scratchpad `Knowledge` with concise facts, discarding raw hits.
5. Repeat until the planner chooses `Answer` or `maxIterations` is reached.
6. Finalizer turns the Scratchpad into user-facing prose. If the loop hit the iteration cap, a best-effort answer is returned with an error.

## Design choices

- **Scratchpad compression** keeps context size flat even with many iterations.
- **Dual-model support** allows a strong planner and cheaper synthesizer/finalizer to save cost.
- **Cost accounting** sums LLM and search costs so callers can monitor spend.
- **No embedded LLM clients** keeps dependencies light and lets you swap providers freely.

## Extending

- Swap search: implement `SearchProvider` (e.g., corporate wiki, vector DB, custom scraper).
- Add observability: wrap `LLMProvider.Generate` or `SearchProvider.Search` with your tracing middleware.
- Add tools: extend planner prompt and decision parser to recognize more actions.
- Add strategies: register a new `StrategyFactory` and select it via `WithStrategyName`.

## GraphReader strategy

The `graph-reader` strategy implements a graph-based research loop inspired by GraphReader. It maintains a notebook of atomic facts and explores neighbor queries until the notebook is sufficient to answer the question.
