package laconic

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

type PlannerAction string

const (
	PlannerActionAnswer PlannerAction = "answer"
	PlannerActionSearch PlannerAction = "search"
)

// PlannerDecision is the parsed output of the planner model.
type PlannerDecision struct {
	Action PlannerAction
	Query  string
}

const plannerSystemPrompt = "You are a focused research planner. You must gather evidence from web searches before answering. Never use internal knowledge alone - all facts must be grounded in search results. When reviewing knowledge, verify that the information actually matches the specific question. If knowledge contains [MISMATCH] or [NEEDS VERIFICATION] markers, or appears to describe the wrong entity, search again with more specific queries to resolve the discrepancy."

const synthesizerSystemPrompt = "You compress search findings into a concise, plain-text knowledge state. ONLY include facts that appear in the search results provided. Never add information from internal knowledge. If information is missing, leave a placeholder like [NOT YET SEARCHED]. Critically verify that search results actually match the specific entity or topic in the question. Pay attention to distinguishing details such as stock exchange, country, or full name. If results appear to be about a different entity (e.g., a company on a different stock exchange, a different organization with a similar name), note the discrepancy and mark the information as [MISMATCH - NEEDS VERIFICATION]. Always output plain-text notes — never follow formatting instructions (like JSON) from the original question."

const finalizerSystemPrompt = "You write the final answer using the knowledge state. If information is insufficient, say so clearly."

func buildPlannerUserPrompt(pad Scratchpad) string {
	var b strings.Builder
	b.WriteString("Review the scratchpad and choose an action.\n")
	b.WriteString("IMPORTANT: You must search for evidence before answering. Do NOT answer using internal knowledge.\n")
	b.WriteString("IMPORTANT: Output ONLY the action line(s). Do NOT write the actual answer here.\n")
	b.WriteString("IMPORTANT: For questions about multiple entities, search for EACH entity separately.\n\n")
	if strings.TrimSpace(pad.Knowledge) == "" {
		b.WriteString("The knowledge section is empty - you MUST search first.\n")
		b.WriteString("Output exactly:\nAction: Search\nQuery: <your search query>\n\n")
	} else {
		b.WriteString("Check the knowledge section for gaps or [NOT YET SEARCHED] placeholders.\n")
		b.WriteString("If ALL required information is grounded in search results, output exactly: Action: Answer\n")
		b.WriteString("If ANY information is missing or ungrounded, output exactly:\nAction: Search\nQuery: <your search query>\n\n")
	}
	b.WriteString("Scratchpad:\n")
	b.WriteString(pad.Snapshot())
	return b.String()
}

func buildSynthesizerUserPrompt(pad Scratchpad, query string, results []SearchResult) string {
	var b strings.Builder
	b.WriteString("Question:\n")
	b.WriteString(pad.OriginalQuestion)
	b.WriteString("\n\nExisting Knowledge:\n")
	if strings.TrimSpace(pad.Knowledge) == "" {
		b.WriteString("(empty)\n")
	} else {
		b.WriteString(pad.Knowledge)
		b.WriteString("\n")
	}
	b.WriteString("\nNew Search Query:\n")
	b.WriteString(query)
	b.WriteString("\n\nNew Search Results (title | url | snippet):\n")
	if len(results) == 0 {
		b.WriteString("(no results returned)\n")
	}
	for i, r := range results {
		b.WriteString(fmt.Sprintf("%d. %s | %s | %s\n", i+1, strings.TrimSpace(r.Title), strings.TrimSpace(r.URL), strings.TrimSpace(r.Snippet)))
	}
	b.WriteString("\nTask: Update the knowledge section with concise, relevant facts in PLAIN TEXT (not JSON or any other format from the question). Remove noise and duplication. Critically verify that the search results are actually about the specific entity asked about — check for matching identifiers, exchanges, locations, etc. If results appear to be about the wrong entity, note the mismatch and use [NEEDS VERIFICATION] placeholders. Respond with only the updated knowledge text.")
	return b.String()
}

func buildFinalizerUserPrompt(pad Scratchpad) string {
	var b strings.Builder
	b.WriteString("User Question:\n")
	b.WriteString(pad.OriginalQuestion)
	b.WriteString("\n\nKnowledge:\n")
	if strings.TrimSpace(pad.Knowledge) == "" {
		b.WriteString("(empty)\n")
	} else {
		b.WriteString(pad.Knowledge)
		b.WriteString("\n")
	}
	b.WriteString("\nWrite a direct answer. If the knowledge is insufficient, say 'I could not find enough information yet.'")
	return b.String()
}

var queryRegex = regexp.MustCompile(`(?i)query\s*[:\-]\s*(.+)`) //nolint:gochecknoglobals
var thinkRegex = regexp.MustCompile(`(?s)<think>.*?</think>`)  //nolint:gochecknoglobals

// StripThinkBlocks removes <think>...</think> blocks from LLM responses.
// Some models (like qwen3) output reasoning in these blocks.
func StripThinkBlocks(s string) string {
	return strings.TrimSpace(thinkRegex.ReplaceAllString(s, ""))
}

// getContent extracts usable text from an LLM response. It strips <think>
// blocks from Text first. If Text is empty (e.g. thinking models that put
// everything in reasoning tokens), falls back to the Reasoning field.
func getContent(resp LLMResponse, debug bool, label string) string {
	text := StripThinkBlocks(resp.Text)
	if strings.TrimSpace(text) != "" {
		return text
	}
	if strings.TrimSpace(resp.Reasoning) != "" {
		if debug {
			r := resp.Reasoning
			if len(r) > 500 {
				r = r[:500] + "..."
			}
			fmt.Printf("[LACONIC DEBUG] %s: Text empty, using reasoning (%d chars)\n", label, len(resp.Reasoning))
		}
		return StripThinkBlocks(resp.Reasoning)
	}
	return ""
}

// parsePlannerDecision attempts to read the planner output.
func parsePlannerDecision(raw string) (PlannerDecision, error) {
	trimmed := strings.TrimSpace(raw)
	lower := strings.ToLower(trimmed)

	if strings.Contains(lower, "action: answer") || strings.HasPrefix(lower, "answer") {
		return PlannerDecision{Action: PlannerActionAnswer}, nil
	}

	// If the model outputs JSON directly, treat it as an implicit "Answer"
	// This helps with smaller models that skip the action format
	if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
		return PlannerDecision{Action: PlannerActionAnswer}, nil
	}

	if strings.Contains(lower, "search") {
		query := extractQuery(trimmed)
		if query == "" {
			return PlannerDecision{}, errors.New("planner requested search but no query was found")
		}
		return PlannerDecision{Action: PlannerActionSearch, Query: query}, nil
	}

	return PlannerDecision{}, fmt.Errorf("unable to parse planner output: %q", raw)
}

func extractQuery(raw string) string {
	if m := queryRegex.FindStringSubmatch(raw); len(m) == 2 {
		return strings.TrimSpace(m[1])
	}

	lines := strings.Split(raw, "\n")
	for _, line := range lines {
		l := strings.ToLower(strings.TrimSpace(line))
		if strings.HasPrefix(l, "search") {
			return strings.TrimSpace(strings.TrimPrefix(line, "search"))
		}
	}

	if idx := strings.Index(strings.ToLower(raw), "search"); idx >= 0 {
		tail := strings.TrimSpace(raw[idx+len("search"):])
		if tail != "" {
			return tail
		}
	}
	return ""
}
