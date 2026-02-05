// Package search provides search provider implementations for the laconic agent.
//
// Available providers:
//
//   - DuckDuckGo: Free, no API key required (uses HTML scraping of lite.duckduckgo.com)
//   - Brave: Requires API key via X-Subscription-Token header
//   - Tavily: Requires API key, supports basic/advanced depth modes
//
// # DuckDuckGo Example
//
//	provider := search.NewDuckDuckGo()
//	results, err := provider.Search(ctx, "golang web frameworks")
//
// # Brave Example
//
//	provider := search.NewBrave("your-api-key")
//	results, err := provider.Search(ctx, "best practices for API design")
//
// # Tavily Example
//
//	provider := search.NewTavily("your-api-key", "advanced")
//	results, err := provider.Search(ctx, "climate change research 2024")
//
// # Custom Providers
//
// Implement the laconic.SearchProvider interface to add your own search backend:
//
//	type SearchProvider interface {
//	    Search(ctx context.Context, query string) ([]laconic.SearchResult, error)
//	}
package search
