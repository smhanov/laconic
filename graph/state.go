package graph

import "time"

// AtomicFact represents a single piece of verified information
// extracted from a search result or web page.
type AtomicFact struct {
	ID        string `json:"id"`
	Content   string `json:"content"`
	SourceURL string `json:"source_url,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

// Node represents a search topic or query in the exploration graph.
type Node struct {
	Name      string `json:"name"`
	Rationale string `json:"rationale,omitempty"`
	Depth     int    `json:"depth"`
}

// Notebook acts as the agent's short-term memory, highly compressed.
type Notebook struct {
	Clues []AtomicFact `json:"clues"`
}

// RationalPlan defines the strategy.
type RationalPlan struct {
	OriginalQuestion string   `json:"original_question"`
	Strategy         []string `json:"strategy"`
	KeyElements      []string `json:"key_elements"`
}

// AgentState holds the complete state of the research session.
type AgentState struct {
	Plan     RationalPlan
	Notebook Notebook
	Queue    []Node
	Visited  map[string]bool
}

// NewAgentState initializes the graph agent state.
func NewAgentState(question string) *AgentState {
	return &AgentState{
		Plan:     RationalPlan{OriginalQuestion: question},
		Notebook: Notebook{Clues: make([]AtomicFact, 0)},
		Queue:    make([]Node, 0),
		Visited:  make(map[string]bool),
	}
}

// NewAtomicFact creates a fact with a timestamp.
func NewAtomicFact(content, sourceURL string) AtomicFact {
	return AtomicFact{
		Content:   content,
		SourceURL: sourceURL,
		Timestamp: time.Now().Unix(),
	}
}
