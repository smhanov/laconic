package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestSerperSearch(t *testing.T) {
	var gotMethod, gotAPIKey, gotContentType string
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotAPIKey = r.Header.Get("X-API-KEY")
		gotContentType = r.Header.Get("Content-Type")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"organic": [
				{"title": "one", "link": "https://example.com/1", "snippet": "first"},
				{"title": "two", "link": "https://example.com/2", "snippet": "second"},
				{"title": "three", "link": "https://example.com/3", "snippet": "third"},
				{"title": "four", "link": "https://example.com/4", "snippet": "fourth"},
				{"title": "five", "link": "https://example.com/5", "snippet": "fifth"},
				{"title": "six", "link": "https://example.com/6", "snippet": "sixth"}
			]
		}`))
	}))
	defer server.Close()

	provider := NewSerperWithClient("test-key", server.Client())
	provider.endpoint = server.URL

	results, err := provider.Search(context.Background(), "golang search")
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}

	if gotMethod != http.MethodPost {
		t.Fatalf("method = %q, want %q", gotMethod, http.MethodPost)
	}
	if gotAPIKey != "test-key" {
		t.Fatalf("X-API-KEY = %q, want test-key", gotAPIKey)
	}
	if gotContentType != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", gotContentType)
	}
	if gotBody["q"] != "golang search" {
		t.Fatalf("q = %#v, want golang search", gotBody["q"])
	}
	if gotBody["num"] != float64(serperResultLimit) {
		t.Fatalf("num = %#v, want %d", gotBody["num"], serperResultLimit)
	}

	if len(results) != serperResultLimit {
		t.Fatalf("len(results) = %d, want %d", len(results), serperResultLimit)
	}
	if results[0].Title != "one" || results[0].URL != "https://example.com/1" || results[0].Snippet != "first" {
		t.Fatalf("first result = %#v", results[0])
	}
	if results[4].Title != "five" {
		t.Fatalf("last returned result = %#v, want title five", results[4])
	}
}

func TestSerperSearchRequiresAPIKey(t *testing.T) {
	_, err := NewSerper(" ").Search(context.Background(), "golang")
	if err == nil {
		t.Fatal("Search returned nil error, want missing API key error")
	}
}

func TestSerperSearchLive(t *testing.T) {
	apiKey := strings.TrimSpace(os.Getenv("SERPER_API_KEY"))
	if apiKey == "" {
		t.Skip("SERPER_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	results, err := NewSerper(apiKey).Search(ctx, "laconic Go research agent")
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}
	if results[0].Title == "" || results[0].URL == "" {
		t.Fatalf("first result missing title or URL: %#v", results[0])
	}
}
