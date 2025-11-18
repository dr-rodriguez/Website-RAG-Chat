# Website RAG Chat

Adapted from https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent-v2

## Installation

Use uv to install the dependencies.
```bash
uv sync
```

With the environment activated, install the playwright browser.
```bash
playwright install
```

Copy `.env.example` to `.env` and fill in the values.

## Usage

1. Crawl the website and insert the documents into the ChromaDB database.
2. Run the RAG builder to chunk the documents and insert them into the ChromaDB database.
3. Run the RAG agent to answer questions about the documents.

