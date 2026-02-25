package laconic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"text/template"
	"time"

	"github.com/smhanov/laconic/graph"
)

const (
	graphPlannerSystemPrompt     = "You are a research planner. Think briefly, then output valid JSON. Keep your reasoning under 200 words."
	graphExtractorSystemPrompt   = "You are a data extraction tool. Read the snippets, extract facts, output JSON. Think briefly, keep reasoning under 100 words. Do not write a report."
	graphNeighborSystemPrompt    = "You are a research navigator. Think briefly, then output a JSON array of search queries. Keep reasoning under 100 words."
	graphAnswerCheckSystemPrompt = "You are a research validator. Think briefly, then output JSON. Keep reasoning under 50 words."
	graphFinalizerSystemPrompt   = "Write the answer using only the provided knowledge. Think briefly, keep reasoning under 200 words. Then write a thorough answer."
	graphCondenserSystemPrompt   = "Condense these facts into one brief paragraph. Keep all numbers, dates, and names. Remove duplicates. Think briefly, keep reasoning under 50 words. Output only the paragraph."

	// graphFinalizerRetrySystemPrompt is the simplified system prompt used
	// when the primary finalizer attempt returns empty content (model spent
	// all output tokens on thinking). It avoids mentioning thinking at all,
	// which may help some models allocate more tokens to the answer.
	graphFinalizerRetrySystemPrompt = "Answer the question using the provided knowledge. Be concise."

	// maxExtractContentLen limits the page content sent to the extractor.
	// Prevents overwhelming the model's context window with huge pages.
	maxExtractContentLen = 8000

	// maxDirectFacts is the maximum number of deduplicated facts sent
	// directly to the finalizer. Above this threshold, facts are compressed
	// into compact knowledge paragraphs via batched LLM calls to fit
	// within model output-token limits.
	maxDirectFacts = 40

	// factCondenseBatch is the number of facts per condensation LLM call.
	factCondenseBatch = 25

	// maxRetryKnowledgeLen caps the knowledge block length on finalizer
	// retry attempts. Shorter input leaves more output-token budget.
	maxRetryKnowledgeLen = 1500

	// maxFinalizerRetries is how many retry attempts to make if the
	// finalizer returns empty content.
	maxFinalizerRetries = 2
)

type graphReaderStrategy struct {
	agent *Agent
	cfg   GraphReaderConfig
}

// stripThinking removes <think> blocks from the response, logging the reasoning
// content when debug mode is enabled. The label identifies which step produced it.
func (s *graphReaderStrategy) stripThinking(label, text string) string {
	if s.agent.debug {
		// Extract and log reasoning blocks before stripping.
		matches := thinkBlockRegex.FindAllStringSubmatch(text, -1)
		for _, m := range matches {
			reasoning := strings.TrimSpace(m[1])
			if reasoning != "" {
				// Truncate very long reasoning to avoid flooding logs.
				if len(reasoning) > 2000 {
					reasoning = reasoning[:2000] + "... [truncated]"
				}
				fmt.Printf("[LACONIC DEBUG] %s Reasoning (%d chars):\n%s\n", label, len(m[1]), reasoning)
			}
		}
	}
	return StripThinkBlocks(text)
}

// getResponseContent extracts usable text from an LLM response, handling the
// case where thinking models put content in the Reasoning field instead of Text.
// It first tries resp.Text (with <think> blocks stripped). If that's empty and
// resp.Reasoning is non-empty, falls back to reasoning content.
func (s *graphReaderStrategy) getResponseContent(label string, resp LLMResponse) string {
	text := s.stripThinking(label, resp.Text)
	if strings.TrimSpace(text) != "" {
		return text
	}
	// Model used reasoning tokens and produced no text content.
	// The reasoning may contain the useful output (e.g. JSON).
	if strings.TrimSpace(resp.Reasoning) != "" {
		if s.agent.debug {
			r := resp.Reasoning
			if len(r) > 2000 {
				r = r[:2000] + "... [truncated]"
			}
			fmt.Printf("[LACONIC DEBUG] %s: Text empty, falling back to reasoning (%d chars):\n%s\n",
				label, len(resp.Reasoning), r)
		}
		// Strip any <think> blocks that might appear within reasoning too.
		return StripThinkBlocks(resp.Reasoning)
	}
	return ""
}

var thinkBlockRegex = regexp.MustCompile(`(?s)<think>(.*?)</think>`) //nolint:gochecknoglobals

func newGraphReaderStrategy(a *Agent) (Strategy, error) {
	cfg := a.graphReaderConfig
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = defaultGraphReaderSteps
	}
	if cfg.Planner == nil {
		cfg.Planner = a.planner
	}
	if cfg.Extractor == nil {
		cfg.Extractor = a.synthesizer
	}
	if cfg.Neighbor == nil {
		cfg.Neighbor = cfg.Extractor
	}
	if cfg.Finalizer == nil {
		cfg.Finalizer = a.finalizer
	}

	return &graphReaderStrategy{agent: a, cfg: cfg}, nil
}

func (s *graphReaderStrategy) Name() string {
	return "graph-reader"
}

func (s *graphReaderStrategy) Answer(ctx context.Context, question string) (Result, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return Result{}, errors.New("question is empty")
	}
	if s.cfg.Planner == nil {
		return Result{}, errors.New("planner model is not configured")
	}
	if s.cfg.Extractor == nil {
		return Result{}, errors.New("extractor model is not configured")
	}
	if s.cfg.Neighbor == nil {
		return Result{}, errors.New("neighbor model is not configured")
	}
	if s.cfg.Finalizer == nil {
		return Result{}, errors.New("finalizer model is not configured")
	}
	if s.agent.searcher == nil {
		return Result{}, errors.New("search provider is not configured")
	}

	var totalCost float64

	state := graph.NewAgentState(question)

	// Pre-populate notebook from prior knowledge if supplied.
	if pk := s.agent.priorKnowledge; pk != "" {
		var priorFacts []graph.AtomicFact
		if err := json.Unmarshal([]byte(pk), &priorFacts); err == nil {
			state.Notebook.Clues = append(state.Notebook.Clues, priorFacts...)
		} else {
			// Plain text: wrap as a single atomic fact.
			state.Notebook.Clues = append(state.Notebook.Clues, graph.AtomicFact{
				ID:      "prior-1",
				Content: pk,
			})
		}
	}

	plan, cost, err := s.generatePlan(ctx, question)
	totalCost += cost
	if err != nil {
		return Result{}, fmt.Errorf("graph planner: %w", err)
	}
	state.Plan = plan

	initialNodes, cost, err := s.generateInitialNodes(ctx, state.Plan)
	totalCost += cost
	if err != nil {
		return Result{}, fmt.Errorf("graph init nodes: %w", err)
	}
	for _, node := range initialNodes {
		state.Queue = append(state.Queue, node)
	}

	for step := 0; step < s.cfg.MaxSteps && len(state.Queue) > 0; step++ {
		current := state.Queue[0]
		state.Queue = state.Queue[1:]

		if state.Visited[current.Name] {
			continue
		}
		state.Visited[current.Name] = true

		results, err := s.agent.searcher.Search(ctx, current.Name)
		if err != nil {
			return Result{}, fmt.Errorf("search: %w", err)
		}
		totalCost += s.agent.searchCost

		extraction, cost, err := s.extractFacts(ctx, state.Plan, current.Name, results)
		totalCost += cost
		if err != nil {
			if s.agent.debug {
				fmt.Printf("[LACONIC DEBUG] Fact extraction failed: %v\n", err)
			}
		}
		if err == nil {
			s.addFacts(state, extraction.NewFacts)
			for _, url := range extraction.ReadMoreURLs {
				if s.agent.fetcher == nil {
					continue
				}
				if isAdOrTrackerURL(url) {
					if s.agent.debug {
						fmt.Printf("[LACONIC DEBUG] Skipping ad/tracker URL: %s\n", url)
					}
					continue
				}
				content, err := s.agent.fetcher.Fetch(ctx, url)
				if err != nil {
					continue
				}
				// Skip trivially short pages (titles only, JS-rendered, etc.)
				if len(strings.TrimSpace(content)) < 200 {
					if s.agent.debug {
						fmt.Printf("[LACONIC DEBUG] Skipping too-short page content (%d chars): %s\n", len(content), url)
					}
					continue
				}
				deepFacts, cost, err := s.extractFactsFromText(ctx, state.Plan, url, content)
				totalCost += cost
				if err != nil {
					continue
				}
				s.addFacts(state, deepFacts)
			}
		}

		if len(state.Notebook.Clues) == 0 {
			if s.agent.debug {
				fmt.Println("[LACONIC DEBUG] Notebook still empty, skipping answer check")
			}
		} else if len(state.Notebook.Clues) < 5 {
			if s.agent.debug {
				fmt.Printf("[LACONIC DEBUG] Only %d facts collected, skipping answer check (need ≥5)\n", len(state.Notebook.Clues))
			}
		} else {
			canAnswer, cost, err := s.canAnswer(ctx, state)
			totalCost += cost
			if err == nil && canAnswer {
				break
			}
		}

		neighbors, cost, err := s.findNeighbors(ctx, state, current.Name)
		totalCost += cost
		if err != nil {
			continue
		}
		for _, node := range neighbors {
			if state.Visited[node.Name] || s.isQueued(state, node.Name) {
				continue
			}
			state.Queue = append(state.Queue, node)
		}
	}

	answer, cost, err := s.finalize(ctx, state)
	totalCost += cost
	if err != nil {
		return Result{}, err
	}

	// Encode collected knowledge as JSON.
	knowledge := ""
	if len(state.Notebook.Clues) > 0 {
		if kb, err := json.Marshal(state.Notebook.Clues); err == nil {
			knowledge = string(kb)
		}
	}
	return Result{Answer: answer, Cost: totalCost, Knowledge: knowledge}, nil
}

type planResponse struct {
	ResearchGoal string   `json:"research_goal"`
	Strategy     []string `json:"strategy"`
	KeyElements  []string `json:"key_elements"`
}

type extractResponse struct {
	NewFacts     []graph.AtomicFact `json:"new_facts"`
	ReadMoreURLs []string           `json:"read_more_urls"`
}

type answerCheckResponse struct {
	CanAnswer bool `json:"can_answer"`
}

func (s *graphReaderStrategy) generatePlan(ctx context.Context, question string) (graph.RationalPlan, float64, error) {
	user, err := renderTemplate(graph.TmplPlan, map[string]any{"Question": question})
	if err != nil {
		return graph.RationalPlan{}, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Plan System Prompt:\n%s\n", graphPlannerSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Plan User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Planner.Generate(ctx, graphPlannerSystemPrompt, user)
	if err != nil {
		return graph.RationalPlan{}, 0, err
	}
	raw := s.getResponseContent("Graph Plan", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Plan Response:\n%s\n", raw)
	}

	var parsed planResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &parsed); err != nil {
		return graph.RationalPlan{}, resp.Cost, fmt.Errorf("plan JSON parse: %w (raw: %.200s)", err, raw)
	}

	researchGoal := strings.TrimSpace(parsed.ResearchGoal)
	if researchGoal == "" {
		// Fallback: strip formatting instructions from the question.
		// Look for keywords that start output formatting sections.
		goal := question
		for _, marker := range []string{"FORMAT YOUR RESPONSE", "FORMAT:", "OUTPUT FORMAT", "RESPONSE FORMAT", "\n#"} {
			if idx := strings.Index(goal, marker); idx > 0 {
				goal = strings.TrimSpace(goal[:idx])
				break
			}
		}
		// Truncate to a reasonable length
		if len(goal) > 500 {
			goal = goal[:500]
		}
		researchGoal = goal
	}

	return graph.RationalPlan{
		OriginalQuestion: question,
		ResearchGoal:     researchGoal,
		Strategy:         trimStrings(parsed.Strategy),
		KeyElements:      trimStrings(parsed.KeyElements),
	}, resp.Cost, nil
}

func (s *graphReaderStrategy) generateInitialNodes(ctx context.Context, plan graph.RationalPlan) ([]graph.Node, float64, error) {
	user, err := renderTemplate(graph.TmplInit, plan)
	if err != nil {
		return nil, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Init System Prompt:\n%s\n", graphPlannerSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Init User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Planner.Generate(ctx, graphPlannerSystemPrompt, user)
	if err != nil {
		return nil, 0, err
	}
	raw := s.getResponseContent("Graph Init", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Init Response:\n%s\n", raw)
	}

	var queries []string
	if err := json.Unmarshal([]byte(extractJSON(raw)), &queries); err != nil {
		return nil, resp.Cost, fmt.Errorf("init nodes JSON parse: %w (raw: %.200s)", err, raw)
	}
	queries = trimStrings(queries)

	nodes := make([]graph.Node, 0, len(queries))
	for _, q := range queries {
		if q == "" {
			continue
		}
		nodes = append(nodes, graph.Node{Name: q, Rationale: "initial", Depth: 0})
	}
	return nodes, resp.Cost, nil
}

// extractJSON attempts to extract a JSON object or array from an LLM response
// that may wrap the JSON in markdown code blocks or include leading text.
func extractJSON(raw string) string {
	// Try to find JSON in markdown code blocks first
	codeBlockRe := regexp.MustCompile("(?s)```(?:json)?\\s*\n(.*?)\n```")
	if m := codeBlockRe.FindStringSubmatch(raw); len(m) == 2 {
		return strings.TrimSpace(m[1])
	}
	// Find first { or [ and last } or ]
	start := -1
	var opener, closer byte
	for i := 0; i < len(raw); i++ {
		if raw[i] == '{' || raw[i] == '[' {
			start = i
			opener = raw[i]
			if opener == '{' {
				closer = '}'
			} else {
				closer = ']'
			}
			break
		}
	}
	if start < 0 {
		return raw
	}
	end := -1
	for i := len(raw) - 1; i >= start; i-- {
		if raw[i] == closer {
			end = i + 1
			break
		}
	}
	if end < 0 {
		return raw
	}
	return raw[start:end]
}

func (s *graphReaderStrategy) extractFacts(ctx context.Context, plan graph.RationalPlan, currentNode string, results []SearchResult) (extractResponse, float64, error) {
	snippets := make([]map[string]string, 0, len(results))
	for _, r := range results {
		content := strings.TrimSpace(r.Snippet)
		if content == "" {
			content = strings.TrimSpace(r.Title)
		}
		snippets = append(snippets, map[string]string{
			"URL":     strings.TrimSpace(r.URL),
			"Content": content,
		})
	}
	user, err := renderTemplate(graph.TmplExtract, map[string]any{
		"Plan":        plan,
		"CurrentNode": currentNode,
		"Snippets":    snippets,
	})
	if err != nil {
		return extractResponse{}, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Extract System Prompt:\n%s\n", graphExtractorSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Extract User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Extractor.Generate(ctx, graphExtractorSystemPrompt, user)
	if err != nil {
		return extractResponse{}, 0, err
	}
	raw := s.getResponseContent("Graph Extract", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Extract Response:\n%s\n", raw)
	}

	var parsed extractResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &parsed); err != nil {
		return extractResponse{}, resp.Cost, fmt.Errorf("extract JSON parse: %w (raw: %.200s)", err, raw)
	}

	return parsed, resp.Cost, nil
}

func (s *graphReaderStrategy) extractFactsFromText(ctx context.Context, plan graph.RationalPlan, sourceURL, content string) ([]graph.AtomicFact, float64, error) {
	// Truncate very long page content to avoid overwhelming the model.
	if len(content) > maxExtractContentLen {
		if s.agent.debug {
			fmt.Printf("[LACONIC DEBUG] Truncating page content from %d to %d chars: %s\n", len(content), maxExtractContentLen, sourceURL)
		}
		content = content[:maxExtractContentLen]
	}
	user, err := renderTemplate(graph.TmplExtractText, map[string]any{
		"Plan":      plan,
		"SourceURL": sourceURL,
		"Content":   content,
	})
	if err != nil {
		return nil, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText System Prompt:\n%s\n", graphExtractorSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Extractor.Generate(ctx, graphExtractorSystemPrompt, user)
	if err != nil {
		return nil, 0, err
	}
	raw := s.getResponseContent("Graph ExtractText", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText Response:\n%s\n", raw)
	}

	var parsed struct {
		NewFacts []graph.AtomicFact `json:"new_facts"`
	}
	if err := json.Unmarshal([]byte(extractJSON(raw)), &parsed); err != nil {
		return nil, resp.Cost, fmt.Errorf("extract text JSON parse: %w (raw: %.200s)", err, raw)
	}

	return parsed.NewFacts, resp.Cost, nil
}

func (s *graphReaderStrategy) findNeighbors(ctx context.Context, state *graph.AgentState, currentNode string) ([]graph.Node, float64, error) {
	user, err := renderTemplate(graph.TmplNeighbors, map[string]any{
		"Plan":        state.Plan,
		"Notebook":    state.Notebook,
		"CurrentNode": currentNode,
	})
	if err != nil {
		return nil, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors System Prompt:\n%s\n", graphNeighborSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Neighbor.Generate(ctx, graphNeighborSystemPrompt, user)
	if err != nil {
		return nil, 0, err
	}
	raw := s.getResponseContent("Graph Neighbors", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors Response:\n%s\n", raw)
	}

	var queries []string
	if err := json.Unmarshal([]byte(extractJSON(raw)), &queries); err != nil {
		return nil, resp.Cost, fmt.Errorf("neighbors JSON parse: %w (raw: %.200s)", err, raw)
	}
	queries = trimStrings(queries)

	nodes := make([]graph.Node, 0, len(queries))
	for _, q := range queries {
		if q == "" {
			continue
		}
		nodes = append(nodes, graph.Node{Name: q, Rationale: "neighbor"})
	}
	return nodes, resp.Cost, nil
}

func (s *graphReaderStrategy) canAnswer(ctx context.Context, state *graph.AgentState) (bool, float64, error) {
	user, err := renderTemplate(graph.TmplAnswerCheck, map[string]any{
		"Plan":     state.Plan,
		"Notebook": state.Notebook,
	})
	if err != nil {
		return false, 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck System Prompt:\n%s\n", graphAnswerCheckSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck User Prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Planner.Generate(ctx, graphAnswerCheckSystemPrompt, user)
	if err != nil {
		return false, 0, err
	}
	raw := s.getResponseContent("Graph AnswerCheck", resp)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck Response:\n%s\n", raw)
	}

	var parsed answerCheckResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &parsed); err != nil {
		return false, resp.Cost, fmt.Errorf("answer check JSON parse: %w (raw: %.200s)", err, raw)
	}
	return parsed.CanAnswer, resp.Cost, nil
}

// finalize generates the final answer using a two-phase approach designed
// to work within tight output-token limits (e.g. 8192 tokens):
//
//  1. Condensation: strip URLs, deduplicate, then batch-compress large fact
//     sets into compact knowledge paragraphs via small LLM calls.
//  2. Question trimming: strip research/step instructions from the original
//     question, keeping only the ResearchGoal and any formatting template
//     (detected by "FORMAT" marker). This dramatically reduces thinking
//     token consumption since the model doesn't re-process research steps.
//  3. Generation: produce the answer from the condensed knowledge and
//     compact question, fitting within the output-token budget.
func (s *graphReaderStrategy) finalize(ctx context.Context, state *graph.AgentState) (string, float64, error) {
	totalCost := 0.0

	// Phase 1: Build a compact knowledge block from notebook facts.
	knowledgeBlock, cost, err := s.buildKnowledge(ctx, state.Notebook.Clues)
	totalCost += cost
	if err != nil {
		return "", totalCost, err
	}

	// Phase 2: Build a compact question for the finalizer.
	compactQuestion := s.buildFinalizerQuestion(state)

	// Phase 3: Attempt finalization with full compact question.
	result, reasoning, cost, err := s.attemptFinalize(ctx, graphFinalizerSystemPrompt, compactQuestion, knowledgeBlock)
	totalCost += cost
	if err != nil {
		return "", totalCost, err
	}
	if strings.TrimSpace(result) != "" {
		return result, totalCost, nil
	}

	// Phase 4: Retry with progressively simpler prompts.
	// When the model spends all output tokens on reasoning and produces no
	// content text, we can use the reasoning itself as pre-digested context
	// for a follow-up call with a much simpler prompt. This typically
	// succeeds because the heavy analysis is already done.
	goal := state.Plan.ResearchGoal
	if goal == "" {
		goal = state.Plan.OriginalQuestion
		if len(goal) > 500 {
			goal = goal[:500]
		}
	}

	for attempt := 1; attempt <= maxFinalizerRetries; attempt++ {
		if s.agent.debug {
			fmt.Printf("[LACONIC DEBUG] Finalizer returned empty, retry %d/%d (reasoning=%d chars)\n",
				attempt, maxFinalizerRetries, len(reasoning))
		}

		// Build context for retry. If we have reasoning from the previous
		// attempt, use it as pre-digested analysis to reduce thinking load.
		retryKnowledge := knowledgeBlock
		if reasoning != "" {
			// The model already analyzed the facts; use its reasoning as
			// the knowledge input. Truncate to fit token budget.
			truncReasoning := reasoning
			if len(truncReasoning) > maxRetryKnowledgeLen {
				truncReasoning = truncReasoning[:maxRetryKnowledgeLen]
				if idx := strings.LastIndex(truncReasoning, ". "); idx > 0 {
					truncReasoning = truncReasoning[:idx+1]
				}
			}
			retryKnowledge = truncReasoning
		} else {
			// No reasoning available; truncate raw knowledge further.
			truncLimit := maxRetryKnowledgeLen / attempt
			if len(retryKnowledge) > truncLimit {
				retryKnowledge = retryKnowledge[:truncLimit]
				if idx := strings.LastIndex(retryKnowledge, ". "); idx > 0 {
					retryKnowledge = retryKnowledge[:idx+1]
				}
			}
		}

		result, reasoning, cost, err = s.attemptFinalize(ctx, graphFinalizerRetrySystemPrompt, goal, retryKnowledge)
		totalCost += cost
		if err != nil {
			return "", totalCost, err
		}
		if strings.TrimSpace(result) != "" {
			return result, totalCost, nil
		}
	}

	// Phase 5: All retries exhausted. Return the condensed knowledge itself
	// as a fallback so the caller gets *something*.
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer retries exhausted, returning condensed knowledge as fallback\n")
	}
	if strings.TrimSpace(knowledgeBlock) != "" {
		return knowledgeBlock, totalCost, nil
	}
	return "", totalCost, fmt.Errorf("finalizer produced no output after %d retries", maxFinalizerRetries+1)
}

// attemptFinalize makes a single finalizer LLM call and returns the
// answer with think blocks stripped, plus any reasoning content. It
// returns an empty answer string (not an error) when the model produced
// only thinking/reasoning content, allowing the caller to retry.
func (s *graphReaderStrategy) attemptFinalize(ctx context.Context, systemPrompt, question, knowledge string) (answer string, reasoning string, cost float64, err error) {
	var b bytes.Buffer
	b.WriteString("Question:\n")
	b.WriteString(question)
	b.WriteString("\n\nKnowledge:\n")
	if knowledge == "" {
		b.WriteString("(none collected)\n")
	} else {
		b.WriteString(knowledge)
	}
	b.WriteString("\nAnswer using only the knowledge above.")

	user := b.String()
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer attempt (%d chars) system: %s\n", len(user), systemPrompt)
		fmt.Printf("[LACONIC DEBUG] Finalizer user prompt:\n%s\n", user)
	}
	resp, err := s.cfg.Finalizer.Generate(ctx, systemPrompt, user)
	if err != nil {
		return "", "", 0, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer raw text (%d chars):\n%s\n", len(resp.Text), resp.Text)
		if resp.Reasoning != "" {
			r := resp.Reasoning
			if len(r) > 2000 {
				r = r[:2000] + "... [truncated]"
			}
			fmt.Printf("[LACONIC DEBUG] Finalizer reasoning (%d chars):\n%s\n", len(resp.Reasoning), r)
		}
	}
	answer = s.stripThinking("Finalizer", resp.Text)
	return answer, resp.Reasoning, resp.Cost, nil
}

// buildFinalizerQuestion constructs a compact question for the finalizer by
// combining the ResearchGoal with any formatting template found in the
// original question. This avoids sending research instructions that would
// waste output tokens on unnecessary thinking.
func (s *graphReaderStrategy) buildFinalizerQuestion(state *graph.AgentState) string {
	original := state.Plan.OriginalQuestion
	goal := state.Plan.ResearchGoal

	// Look for a formatting template marker in the original question.
	// Common markers: "FORMAT YOUR RESPONSE", "FORMAT:", "OUTPUT FORMAT"
	formatMarkers := []string{"FORMAT YOUR RESPONSE", "FORMAT:", "OUTPUT FORMAT"}
	formatSection := ""
	for _, marker := range formatMarkers {
		idx := strings.Index(strings.ToUpper(original), marker)
		if idx >= 0 {
			formatSection = strings.TrimSpace(original[idx:])
			break
		}
	}

	if goal == "" {
		// No ResearchGoal available; use original but truncate if too long.
		if len(original) > 2000 {
			return original[:2000]
		}
		return original
	}

	var b strings.Builder
	b.WriteString(goal)
	if formatSection != "" {
		b.WriteString("\n\n")
		b.WriteString(formatSection)
	}
	return b.String()
}

// buildKnowledge converts raw notebook clues into a compact knowledge block
// suitable for the finalizer. For small fact sets, facts are listed directly
// (without URLs). For larger sets, facts are compressed in batches through
// LLM condensation calls to stay within context/output token budgets.
func (s *graphReaderStrategy) buildKnowledge(ctx context.Context, clues []graph.AtomicFact) (string, float64, error) {
	if len(clues) == 0 {
		return "", 0, nil
	}

	// Strip URLs and deduplicate.
	facts := deduplicateFactTexts(clues)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Finalizer: %d clues deduplicated to %d unique facts\n", len(clues), len(facts))
	}

	// If facts are few enough, list them directly.
	if len(facts) <= maxDirectFacts {
		var b bytes.Buffer
		for _, f := range facts {
			b.WriteString("- ")
			b.WriteString(f)
			b.WriteString("\n")
		}
		return b.String(), 0, nil
	}

	// Condense in batches.
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Condensing %d facts in batches of %d\n", len(facts), factCondenseBatch)
	}
	totalCost := 0.0
	var condensed []string
	for i := 0; i < len(facts); i += factCondenseBatch {
		end := i + factCondenseBatch
		if end > len(facts) {
			end = len(facts)
		}
		batch := facts[i:end]

		var b bytes.Buffer
		for _, f := range batch {
			b.WriteString("- ")
			b.WriteString(f)
			b.WriteString("\n")
		}

		if s.agent.debug {
			fmt.Printf("[LACONIC DEBUG] Condensing batch %d-%d of %d\n", i+1, end, len(facts))
		}
		resp, err := s.cfg.Finalizer.Generate(ctx, graphCondenserSystemPrompt, b.String())
		if err != nil {
			return "", totalCost, fmt.Errorf("fact condensation batch %d-%d: %w", i+1, end, err)
		}
		totalCost += resp.Cost
		text := strings.TrimSpace(s.getResponseContent("Condense", resp))
		if text != "" {
			condensed = append(condensed, text)
		}
	}

	result := strings.Join(condensed, "\n\n")
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Condensed %d facts into %d chars across %d paragraphs\n", len(facts), len(result), len(condensed))
	}
	return result, totalCost, nil
}

// deduplicateFactTexts strips source URLs and deduplicates fact content,
// returning clean text strings. Uses case-insensitive comparison and
// substring containment to catch near-duplicates.
func deduplicateFactTexts(clues []graph.AtomicFact) []string {
	var result []string
	for _, c := range clues {
		text := strings.TrimSpace(c.Content)
		if text == "" {
			continue
		}
		lower := strings.ToLower(text)
		dup := false
		for _, existing := range result {
			existingLower := strings.ToLower(existing)
			if lower == existingLower ||
				strings.Contains(existingLower, lower) ||
				strings.Contains(lower, existingLower) {
				dup = true
				break
			}
		}
		if !dup {
			result = append(result, text)
		}
	}
	return result
}

func (s *graphReaderStrategy) addFacts(state *graph.AgentState, facts []graph.AtomicFact) {
	for _, fact := range facts {
		content := strings.TrimSpace(fact.Content)
		if content == "" {
			continue
		}
		// Deduplicate: exact match or one contains the other (case-insensitive)
		lowerContent := strings.ToLower(content)
		dup := false
		for _, existing := range state.Notebook.Clues {
			lowerExisting := strings.ToLower(strings.TrimSpace(existing.Content))
			if lowerContent == lowerExisting ||
				strings.Contains(lowerExisting, lowerContent) ||
				strings.Contains(lowerContent, lowerExisting) {
				dup = true
				break
			}
		}
		if dup {
			if s.agent.debug {
				fmt.Printf("[LACONIC DEBUG] Skipping duplicate fact: %.80s\n", content)
			}
			continue
		}
		if fact.Timestamp == 0 {
			fact.Timestamp = time.Now().Unix()
		}
		if strings.TrimSpace(fact.ID) == "" {
			fact.ID = fmt.Sprintf("fact-%d", len(state.Notebook.Clues)+1)
		}
		fact.Content = content
		state.Notebook.Clues = append(state.Notebook.Clues, fact)
	}
}

func (s *graphReaderStrategy) isQueued(state *graph.AgentState, name string) bool {
	for _, node := range state.Queue {
		if node.Name == name {
			return true
		}
	}
	return false
}

func renderTemplate(tmpl *template.Template, data any) (string, error) {
	var b bytes.Buffer
	if err := tmpl.Execute(&b, data); err != nil {
		return "", err
	}
	return b.String(), nil
}

func trimStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, v := range values {
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	return out
}

// isAdOrTrackerURL returns true if the URL looks like an ad redirect or tracking URL.
func isAdOrTrackerURL(url string) bool {
	lower := strings.ToLower(url)
	adPatterns := []string{
		"duckduckgo.com/y.js",
		"ad_domain=",
		"ad_provider=",
		"ad_type=",
		"doubleclick.net",
		"googlesyndication.com",
		"googleadservices.com",
		"click.linksynergy.com",
		"redirect.viglink.com",
		"/aclk?",
		"amazon-adsystem.com",
		"ads.yahoo.com",
		"clickserve",
		"tracking.php",
	}
	for _, pat := range adPatterns {
		if strings.Contains(lower, pat) {
			return true
		}
	}
	return false
}
