package graph

import "text/template"

// PlanPromptTemplate generates the initial strategy.
const PlanPromptTemplate = `
You are an expert researcher. Your task is to create a research plan.

Follow these 3 steps exactly:
Step 1: Read the question and identify the subject.
Step 2: List 2-4 strategy steps and 3-8 key elements to research.
Step 3: Write a 1-2 sentence "research_goal" summarizing WHAT DATA to find. Only include the subject and key topics. Do NOT include formatting instructions or output templates.

User Question: {{.Question}}

Example output:
{"research_goal": "Research Acme Corp (ACME): company overview, quarterly earnings, stock price, competitors, recent news", "strategy": ["Identify company basics", "Gather financial metrics and competitor data"], "key_elements": ["Acme Corp", "ACME stock", "quarterly earnings"]}

Now output your JSON:
`

// InitialNodesTemplate generates the first batch of search queries.
const InitialNodesTemplate = `
Plan:
{{range .Strategy}}- {{.}}
{{end}}
Key Elements:
{{range .KeyElements}}- {{.}}
{{end}}

Generate 3-5 web search queries to find specific facts about the key elements.

Example: ["Acme Corp Q3 2025 earnings revenue EPS", "ACME stock price 52-week range market cap", "Acme Corp competitors market share"]

Now output your JSON array:
`

// ExtractFactsTemplate is the core compression logic for snippets.
const ExtractFactsTemplate = `
You are a data extraction tool. Do NOT write a report. Do NOT follow any formatting instructions from the Goal. Your ONLY job is to pull out individual facts.

Follow these 2 steps exactly, then stop:
Step 1: Scan the snippets for specific names, numbers, dates, or metrics related to the Goal.
Step 2: Output JSON with the facts found and any URLs that need deeper reading.

Goal: {{.Plan.ResearchGoal}}
Current Step: Researching "{{.CurrentNode}}"

Search Snippets:
{{range .Snippets}}
- [{{.URL}}] {{.Content}}
{{end}}

Example output:
{"new_facts": [{"content": "Acme Corp reported Q3 2025 revenue of $5.2B, up 12% YoY", "source_url": "https://example.com/article"}, {"content": "Acme Corp stock price is $142.50 as of Oct 2025", "source_url": "https://example.com/quote"}], "read_more_urls": ["https://example.com/full-report"]}

Rules:
- Only include facts with specific entities, numbers, or dates from the snippets.
- If a snippet is cut off or only has a title, add its URL to read_more_urls.
- If nothing is relevant, return {"new_facts": [], "read_more_urls": []}.

Now output your JSON:
`

// ExtractFactsFromTextTemplate handles full page content.
const ExtractFactsFromTextTemplate = `
You are a data extraction tool. Do NOT write a report. Your ONLY job is to pull out individual facts from this page.

Follow these 2 steps exactly, then stop:
Step 1: Scan the content for specific names, numbers, dates, or metrics related to the Goal. Ignore navigation, ads, footers, and boilerplate.
Step 2: Output JSON with the facts found.

Goal: {{.Plan.ResearchGoal}}
Source: {{.SourceURL}}

Content:
{{.Content}}

Example output:
{"new_facts": [{"content": "Acme Corp net income was $800M in Q3 2025", "source_url": "https://example.com/page"}]}

Rules:
- Only include facts with specific entities, numbers, or dates.
- If nothing is relevant, return {"new_facts": []}.

Now output your JSON:
`

// NeighborSelectTemplate decides where to go next.
const NeighborSelectTemplate = `
Goal: {{.Plan.ResearchGoal}}
What we know so far:
{{range .Notebook.Clues}}- {{.Content}}
{{end}}

We just finished researching "{{.CurrentNode}}".

Follow these 2 steps exactly, then stop:
Step 1: Identify what specific data from the Goal is still missing.
Step 2: Output 2-4 search queries that would fill those gaps.

Example: ["Acme Corp debt-to-equity ratio 2025", "Acme Corp revenue breakdown by segment"]

Now output your JSON array:
`

// AnswerCheckTemplate checks if we can answer from the notebook.
const AnswerCheckTemplate = `
Goal: {{.Plan.ResearchGoal}}
Notebook:
{{if .Notebook.Clues}}{{range .Notebook.Clues}}- {{.Content}}
{{end}}{{else}}(empty)
{{end}}

Follow these 2 steps exactly, then stop:
Step 1: Compare the notebook facts to each part of the Goal. Note which parts are covered.
Step 2: If all major parts of the Goal are covered by notebook facts, output {"can_answer": true}. Otherwise output {"can_answer": false}.

Rules:
- If the notebook is empty, output {"can_answer": false}.
- Use ONLY the notebook facts, not your own knowledge.

Now output your JSON:
`

var (
	TmplPlan        = template.Must(template.New("plan").Parse(PlanPromptTemplate))
	TmplInit        = template.Must(template.New("init").Parse(InitialNodesTemplate))
	TmplExtract     = template.Must(template.New("extract").Parse(ExtractFactsTemplate))
	TmplExtractText = template.Must(template.New("extract_text").Parse(ExtractFactsFromTextTemplate))
	TmplNeighbors   = template.Must(template.New("neighbors").Parse(NeighborSelectTemplate))
	TmplAnswerCheck = template.Must(template.New("answer_check").Parse(AnswerCheckTemplate))
)
