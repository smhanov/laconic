# Prompt shapes

These are the concise prompt skeletons used by the agent. Adapt them in your own forks as needed.

## Planner

System: `You are a focused research planner. Decide whether to answer or search.`

User:
```
Review the scratchpad and choose an action.
If enough information is present to answer the original question, output: Action: Answer
Otherwise output: Action: Search
Query: <best next query>

Scratchpad:
<snapshot>
```

Expected outputs:
- `Action: Answer`
- `Action: Search` followed by a `Query:` line.

## Synthesizer

System: `You compress search findings into a concise knowledge state. Keep only facts that help answer the question.`

User includes question, existing knowledge, latest query, and structured results. Model responds **only** with updated knowledge text.

## Finalizer

System: `You write the final answer using the knowledge state. If information is insufficient, say so clearly.`

User includes the original question and the final knowledge block. If knowledge is empty or weak, the model should reply: `I could not find enough information yet.`

## GraphReader

The `graph-reader` strategy uses JSON-only prompts for planning, atomic fact extraction, neighbor selection, and answer checks. See the templates in `graph/prompts.go` for exact shapes and output schemas.
