package laconic

import (
	"fmt"
	"strings"
)

// Scratchpad holds the evolving state of the agent.
type Scratchpad struct {
	OriginalQuestion string
	CurrentStep      string
	Knowledge        string
	History          []string
	IterationCount   int
}

// NewScratchpad initializes scratchpad with the original question.
func NewScratchpad(question string) Scratchpad {
	return Scratchpad{OriginalQuestion: strings.TrimSpace(question)}
}

// AppendHistory adds a concise action log entry.
func (s *Scratchpad) AppendHistory(entry string) {
	if entry == "" {
		return
	}
	s.History = append(s.History, entry)
}

// Snapshot renders the scratchpad state for prompting.
func (s Scratchpad) Snapshot() string {
	var b strings.Builder
	b.WriteString("Question: \n")
	b.WriteString(s.OriginalQuestion)
	b.WriteString("\n\nCurrent Step:\n")
	if s.CurrentStep == "" {
		b.WriteString("(none yet)")
	} else {
		b.WriteString(s.CurrentStep)
	}
	b.WriteString("\n\nKnowledge:\n")
	if strings.TrimSpace(s.Knowledge) == "" {
		b.WriteString("(empty)")
	} else {
		b.WriteString(s.Knowledge)
	}
	if len(s.History) > 0 {
		b.WriteString("\n\nHistory:\n")
		b.WriteString(strings.Join(s.History, "\n"))
	}
	b.WriteString("\n\nIteration: ")
	b.WriteString(fmt.Sprintf("%d", s.IterationCount))
	return b.String()
}
