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
	graphPlannerSystemPrompt     = "You are a research planner. Output valid JSON only."
	graphExtractorSystemPrompt   = "You are a research compressor. Output valid JSON only."
	graphNeighborSystemPrompt    = "You are a research navigator. Output valid JSON only."
	graphAnswerCheckSystemPrompt = "You are a research validator. Output valid JSON only."
	graphFinalizerSystemPrompt   = "You synthesize a concise answer grounded only in the notebook facts."
)

type graphReaderStrategy struct {
	agent *Agent
	cfg   GraphReaderConfig
}

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

func (s *graphReaderStrategy) Answer(ctx context.Context, question string) (string, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return "", errors.New("question is empty")
	}
	if s.cfg.Planner == nil {
		return "", errors.New("planner model is not configured")
	}
	if s.cfg.Extractor == nil {
		return "", errors.New("extractor model is not configured")
	}
	if s.cfg.Neighbor == nil {
		return "", errors.New("neighbor model is not configured")
	}
	if s.cfg.Finalizer == nil {
		return "", errors.New("finalizer model is not configured")
	}
	if s.agent.searcher == nil {
		return "", errors.New("search provider is not configured")
	}

	state := graph.NewAgentState(question)
	plan, err := s.generatePlan(ctx, question)
	if err != nil {
		return "", fmt.Errorf("graph planner: %w", err)
	}
	state.Plan = plan

	initialNodes, err := s.generateInitialNodes(ctx, state.Plan)
	if err != nil {
		return "", fmt.Errorf("graph init nodes: %w", err)
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
			continue
		}

		extraction, err := s.extractFacts(ctx, state.Plan, current.Name, results)
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
				deepFacts, err := s.extractFactsFromText(ctx, state.Plan, url, content)
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
		} else {
			canAnswer, err := s.canAnswer(ctx, state)
			if err == nil && canAnswer {
				break
			}
		}

		neighbors, err := s.findNeighbors(ctx, state, current.Name)
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

	return s.finalize(ctx, state)
}

type planResponse struct {
	Strategy    []string `json:"strategy"`
	KeyElements []string `json:"key_elements"`
}

type extractResponse struct {
	NewFacts     []graph.AtomicFact `json:"new_facts"`
	ReadMoreURLs []string           `json:"read_more_urls"`
}

type answerCheckResponse struct {
	CanAnswer bool `json:"can_answer"`
}

func (s *graphReaderStrategy) generatePlan(ctx context.Context, question string) (graph.RationalPlan, error) {
	user, err := renderTemplate(graph.TmplPlan, map[string]any{"Question": question})
	if err != nil {
		return graph.RationalPlan{}, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Plan System Prompt:\n%s\n", graphPlannerSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Plan User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Planner.Generate(ctx, graphPlannerSystemPrompt, user)
	if err != nil {
		return graph.RationalPlan{}, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Plan Response:\n%s\n", raw)
	}

	var resp planResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &resp); err != nil {
		return graph.RationalPlan{}, fmt.Errorf("plan JSON parse: %w (raw: %.200s)", err, raw)
	}

	return graph.RationalPlan{
		OriginalQuestion: question,
		Strategy:         trimStrings(resp.Strategy),
		KeyElements:      trimStrings(resp.KeyElements),
	}, nil
}

func (s *graphReaderStrategy) generateInitialNodes(ctx context.Context, plan graph.RationalPlan) ([]graph.Node, error) {
	user, err := renderTemplate(graph.TmplInit, plan)
	if err != nil {
		return nil, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Init System Prompt:\n%s\n", graphPlannerSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Init User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Planner.Generate(ctx, graphPlannerSystemPrompt, user)
	if err != nil {
		return nil, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Init Response:\n%s\n", raw)
	}

	var queries []string
	if err := json.Unmarshal([]byte(extractJSON(raw)), &queries); err != nil {
		return nil, fmt.Errorf("init nodes JSON parse: %w (raw: %.200s)", err, raw)
	}
	queries = trimStrings(queries)

	nodes := make([]graph.Node, 0, len(queries))
	for _, q := range queries {
		if q == "" {
			continue
		}
		nodes = append(nodes, graph.Node{Name: q, Rationale: "initial", Depth: 0})
	}
	return nodes, nil
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

func (s *graphReaderStrategy) extractFacts(ctx context.Context, plan graph.RationalPlan, currentNode string, results []SearchResult) (extractResponse, error) {
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
		return extractResponse{}, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Extract System Prompt:\n%s\n", graphExtractorSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Extract User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Extractor.Generate(ctx, graphExtractorSystemPrompt, user)
	if err != nil {
		return extractResponse{}, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Extract Response:\n%s\n", raw)
	}

	var resp extractResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &resp); err != nil {
		return extractResponse{}, fmt.Errorf("extract JSON parse: %w (raw: %.200s)", err, raw)
	}

	return resp, nil
}

func (s *graphReaderStrategy) extractFactsFromText(ctx context.Context, plan graph.RationalPlan, sourceURL, content string) ([]graph.AtomicFact, error) {
	user, err := renderTemplate(graph.TmplExtractText, map[string]any{
		"Plan":      plan,
		"SourceURL": sourceURL,
		"Content":   content,
	})
	if err != nil {
		return nil, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText System Prompt:\n%s\n", graphExtractorSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Extractor.Generate(ctx, graphExtractorSystemPrompt, user)
	if err != nil {
		return nil, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph ExtractText Response:\n%s\n", raw)
	}

	var resp struct {
		NewFacts []graph.AtomicFact `json:"new_facts"`
	}
	if err := json.Unmarshal([]byte(extractJSON(raw)), &resp); err != nil {
		return nil, fmt.Errorf("extract text JSON parse: %w (raw: %.200s)", err, raw)
	}

	return resp.NewFacts, nil
}

func (s *graphReaderStrategy) findNeighbors(ctx context.Context, state *graph.AgentState, currentNode string) ([]graph.Node, error) {
	user, err := renderTemplate(graph.TmplNeighbors, map[string]any{
		"Plan":        state.Plan,
		"Notebook":    state.Notebook,
		"CurrentNode": currentNode,
	})
	if err != nil {
		return nil, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors System Prompt:\n%s\n", graphNeighborSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Neighbor.Generate(ctx, graphNeighborSystemPrompt, user)
	if err != nil {
		return nil, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Neighbors Response:\n%s\n", raw)
	}

	var queries []string
	if err := json.Unmarshal([]byte(extractJSON(raw)), &queries); err != nil {
		return nil, fmt.Errorf("neighbors JSON parse: %w (raw: %.200s)", err, raw)
	}
	queries = trimStrings(queries)

	nodes := make([]graph.Node, 0, len(queries))
	for _, q := range queries {
		if q == "" {
			continue
		}
		nodes = append(nodes, graph.Node{Name: q, Rationale: "neighbor"})
	}
	return nodes, nil
}

func (s *graphReaderStrategy) canAnswer(ctx context.Context, state *graph.AgentState) (bool, error) {
	user, err := renderTemplate(graph.TmplAnswerCheck, map[string]any{
		"Plan":     state.Plan,
		"Notebook": state.Notebook,
	})
	if err != nil {
		return false, err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck System Prompt:\n%s\n", graphAnswerCheckSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck User Prompt:\n%s\n", user)
	}
	raw, err := s.cfg.Planner.Generate(ctx, graphAnswerCheckSystemPrompt, user)
	if err != nil {
		return false, err
	}
	raw = StripThinkBlocks(raw)
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph AnswerCheck Response:\n%s\n", raw)
	}

	var resp answerCheckResponse
	if err := json.Unmarshal([]byte(extractJSON(raw)), &resp); err != nil {
		return false, fmt.Errorf("answer check JSON parse: %w (raw: %.200s)", err, raw)
	}
	return resp.CanAnswer, nil
}

func (s *graphReaderStrategy) finalize(ctx context.Context, state *graph.AgentState) (string, error) {
	var b bytes.Buffer
	b.WriteString("User Question:\n")
	b.WriteString(state.Plan.OriginalQuestion)
	b.WriteString("\n\nNotebook Facts:\n")
	if len(state.Notebook.Clues) == 0 {
		b.WriteString("(empty)\n")
	} else {
		for _, clue := range state.Notebook.Clues {
			b.WriteString("- ")
			b.WriteString(strings.TrimSpace(clue.Content))
			if strings.TrimSpace(clue.SourceURL) != "" {
				b.WriteString(" (")
				b.WriteString(strings.TrimSpace(clue.SourceURL))
				b.WriteString(")")
			}
			b.WriteString("\n")
		}
	}
	b.WriteString("\nWrite a direct answer grounded in these facts. If facts are insufficient, say so clearly.")

	user := b.String()
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Finalizer System Prompt:\n%s\n", graphFinalizerSystemPrompt)
		fmt.Printf("[LACONIC DEBUG] Graph Finalizer User Prompt:\n%s\n", user)
	}
	out, err := s.cfg.Finalizer.Generate(ctx, graphFinalizerSystemPrompt, user)
	if err != nil {
		return "", err
	}
	if s.agent.debug {
		fmt.Printf("[LACONIC DEBUG] Graph Finalizer Response:\n%s\n", out)
	}
	return StripThinkBlocks(out), nil
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
