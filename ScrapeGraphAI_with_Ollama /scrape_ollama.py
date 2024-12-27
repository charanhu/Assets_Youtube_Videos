from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "temperature": 0,
        "format": "json",
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
}

smart_scraper = SmartScraperGraph(
    prompt="Where is Charan currently working?",
    source="https://charanhu.github.io",
    config=graph_config,
)

result = smart_scraper.run()

print(result)
