package graph

import "text/template"

// PlanPromptTemplate generates the initial strategy.
const PlanPromptTemplate = `
You are an expert researcher.
User Question: {{.Question}}

Create a "Rational Plan" to answer this question.
1. Break the question down into logical steps.
2. Identify specific Key Elements (names, places, concepts) we need to find.
3. Keep it concise.

Output JSON format:
{
    "strategy": ["step 1", "step 2"],
    "key_elements": ["Entity A", "Entity B"]
}
`

// InitialNodesTemplate generates the first batch of search queries.
const InitialNodesTemplate = `
Plan:
{{range .Strategy}}- {{.}}
{{end}}
Key Elements:
{{range .KeyElements}}- {{.}}
{{end}}

Generate 3-5 specific Search Queries (Nodes) to start our research.
These should be the most likely to yield direct "Atomic Facts" about the key elements.

Output JSON format:
["query 1", "query 2", "query 3"]
`

// ExtractFactsTemplate is the core compression logic for snippets.
const ExtractFactsTemplate = `
Goal: {{.Plan.OriginalQuestion}}
Current Step: Researching "{{.CurrentNode}}"

Search Snippets:
{{range .Snippets}}
- [{{.URL}}] {{.Content}}
{{end}}

Task:
1. Extract "Atomic Facts" from these snippets. An atomic fact is a single, self-contained truth that DIRECTLY helps answer the Goal.
2. STRICT RELEVANCE: Only extract facts that mention specific entities, numbers, dates, or details asked for in the Goal. Skip background info, general definitions, and tangential topics.
3. If a snippet is promising but cut off/incomplete, add its URL to "read_more_urls".
4. If snippets only contain titles with no detail, add their URLs to "read_more_urls".
5. Prefer fewer, high-quality facts over many low-relevance ones.

Output ONLY raw JSON (no markdown, no code blocks):
{
    "new_facts": [
        {"content": "Fact 1", "source_url": "url..."},
        {"content": "Fact 2", "source_url": "url..."}
    ],
    "read_more_urls": ["url1", "url2"]
}
`

// ExtractFactsFromTextTemplate handles full page content.
const ExtractFactsFromTextTemplate = `
Goal: {{.Plan.OriginalQuestion}}
We fetched full content from: {{.SourceURL}}

Content:
{{.Content}}

Task:
1. Extract "Atomic Facts" from this content. An atomic fact is a single, self-contained truth that DIRECTLY helps answer the Goal.
2. STRICT RELEVANCE: Only extract facts that mention specific entities, numbers, dates, or details asked for in the Goal. Skip general background, definitions, and tangential topics.
3. IGNORE: navigation, ads, cookie notices, newsletter forms, footers, sidebars, website metadata, drug warnings, and side effect lists.
4. If the content is irrelevant or contains only boilerplate, return an empty list.
5. Prefer fewer, high-quality facts over many low-relevance ones.

Output ONLY raw JSON (no markdown, no code blocks):
{
    "new_facts": [
        {"content": "Fact 1", "source_url": "url..."}
    ]
}
`

// NeighborSelectTemplate decides where to go next.
const NeighborSelectTemplate = `
Current Goal: {{.Plan.OriginalQuestion}}
Current Notebook (What we know):
{{range .Notebook.Clues}}- {{.Content}}
{{end}}

We just finished researching "{{.CurrentNode}}".

Task:
Generate new Search Queries ("Neighbors") to explore next.
- These should be based on the new facts we just found.
- If we are missing specific details from the Plan, target those.
- Do NOT suggest queries we have already visited.

Output JSON format:
["next query 1", "next query 2"]
`

// AnswerCheckTemplate checks if we can answer from the notebook.
const AnswerCheckTemplate = `
Goal: {{.Plan.OriginalQuestion}}
Notebook:
{{if .Notebook.Clues}}{{range .Notebook.Clues}}- {{.Content}}
{{end}}{{else}}(empty - no facts collected yet)
{{end}}
Task:
Look ONLY at the facts listed above in the Notebook section.
Do NOT use your own knowledge.
If the notebook is empty or says "(empty", you MUST return can_answer: false.
If the notebook facts cover all parts of the goal, return can_answer: true.
Otherwise return can_answer: false.

Output JSON format:
{
    "can_answer": true
}
`

var (
	TmplPlan        = template.Must(template.New("plan").Parse(PlanPromptTemplate))
	TmplInit        = template.Must(template.New("init").Parse(InitialNodesTemplate))
	TmplExtract     = template.Must(template.New("extract").Parse(ExtractFactsTemplate))
	TmplExtractText = template.Must(template.New("extract_text").Parse(ExtractFactsFromTextTemplate))
	TmplNeighbors   = template.Must(template.New("neighbors").Parse(NeighborSelectTemplate))
	TmplAnswerCheck = template.Must(template.New("answer_check").Parse(AnswerCheckTemplate))
)
