# RAG Pipeline FastAPI Backend

Production-ready FastAPI backend for LangGraph RAG pipeline with authentication, document ingestion, and streaming chat.

## Features

- 🔐 JWT-based authentication with refresh tokens
- 📄 Multi-format document ingestion (PDF, DOCX, URLs, TXT, **PNG / JPG / WEBP images**)
- 🖼️ **Multi-modal RAG**: text + tables + images. Tables and images are summarized at ingest, raw originals are passed to GPT-4o-mini at answer time
- 🔍 **Hybrid retrieval** (PGVector cosine + Postgres full-text search) fused with Reciprocal Rank Fusion
- 🎯 **Pluggable cross-encoder reranker** (BGE default, Cohere optional)
- 🗑️ Document management (list, delete by ID or record, cascades into raw image / table assets)
- 💬 Streaming chat with RAG workflow
- 📜 Chat history with thread management
- 🐘 PostgreSQL with pgvector for vector storage
- 🔄 LangGraph checkpointing for conversation state
- 🧠 Memory management (delete thread memory, user-scoped cleanup)
- 📊 Retrieval logging for debugging and improvement
- ⚙️ Configurable prompts via JSON file
- 🔧 Extensible tool calling (retriever + custom tools via registry)

## Architecture

```
rag-agent-fastapi-backend/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Configuration, security, database
│   ├── models/           # Pydantic schemas and SQLAlchemy models
│   ├── services/         # Business logic layer
│   ├── workflows/        # LangGraph workflow components
│   │   └── tools/        # Tool registry and custom tools
│   └── utils/            # Utility functions
├── alembic/              # Database migrations
└── tests/                # Test suite
```

## Multi-Modal RAG

This backend implements [**Option 3** of the LangChain semi-structured /
multi-modal RAG cookbook](https://blog.langchain.com/semi-structured-multi-modal-rag/):
text is embedded as-is, tables and images are summarized by GPT-4o-mini,
the **summaries** are embedded for retrieval, and the **raw** tables and
images are stored separately and passed back to GPT-4o-mini at answer
synthesis time — so the model sees exact numbers in tables and the actual
pixels of charts, not paraphrases.

### Supported file types

| Type        | Extracted              | Notes                                                      |
| ----------- | ---------------------- | ---------------------------------------------------------- |
| **PDF**     | text + tables + images | Tables via `page.find_tables()`; images via `get_images()` |
| **DOCX**    | text                   |                                                            |
| **TXT / MD**| text                   |                                                            |
| **PNG / JPG / JPEG / WEBP** | image      | Captioned by gpt-4o-mini, stored as one image asset        |

### Pipeline overview

**Ingestion**

```
file ─┬─► text ──────────────────────────► chunk ─► embed ─► PGVector
      │
      ├─► tables (markdown)  ─► summarize ─► embed (summary) ─► PGVector
      │                       └─ raw markdown ──────────────► table_elements
      │
      └─► images (bytes)     ─► caption  ──► embed (caption) ─► PGVector
                              └─ raw bytes ───────────────────► image_assets
```

**Retrieval**

```
query
  ├─► dense  (PGVector cosine, k=20)  ─┐
  ├─► sparse (Postgres FTS,    k=20)  ─┴► RRF (k=60) ─► top-20
  └─► reranker (BGE / Cohere) ──────────────────────► top-5
```

**Answer time** — the retrieved chunks carry `image_asset_id=...` and
`table_id=...` markers. `generate_answer` parses the markers, fetches the
raw tables (markdown) and images (base64) from Postgres, and assembles a
multimodal `HumanMessage` for gpt-4o-mini.

### Example chat queries

```bash
# Image upload + visual question
curl -X POST http://localhost:8000/api/v1/digest \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@architecture-diagram.png"

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Which services in the diagram talk to the queue?"}'

# PDF with embedded tables
curl -X POST http://localhost:8000/api/v1/digest \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@quarterly-report.pdf"

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the Q3 revenue across all regions?"}'
```

### New environment variables (all have sensible defaults)

```bash
# Multi-modal LLM (used for image captions, table summaries, multimodal answers)
MULTIMODAL_MODEL=gpt-4o-mini
MAX_IMAGES_PER_DOC=50
MAX_TABLES_PER_DOC=30
MAX_IMAGES_IN_ANSWER=4
MAX_TABLES_IN_ANSWER=4

# Hybrid retrieval (dense + Postgres FTS, fused with RRF)
HYBRID_DENSE_K=20
HYBRID_SPARSE_K=20
RRF_K=60
HYBRID_FINAL_K=20

# Reranker
RERANKER_PROVIDER=bge          # 'bge' | 'cohere' | 'noop'
BGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
COHERE_RERANK_MODEL=rerank-v4.0-fast
COHERE_API_KEY=                # required when RERANKER_PROVIDER=cohere
RERANK_TOP_K=5
```

> **Deep dive**: see [`MULTIMODAL_RAG_ARCHITECTURE.md`](MULTIMODAL_RAG_ARCHITECTURE.md)
> for the full data model, ingestion + retrieval diagrams, multi-vector
> swap details, RRF math, user isolation guarantees, prompt definitions,
> and operational notes.

## Setup

1. **Install dependencies:**
   ```bash
   cd rag-agent-fastapi-backend
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file in the project root with the following variables:
   ```bash
   # Required
   DATABASE_URL=postgresql://user:password@localhost:5432/dbname
   OPENAI_API_KEY=your-openai-api-key-here
   SECRET_KEY=your-secret-key-for-jwt-tokens
   
   # Optional (with defaults)
   USER_AGENT=RAG-Pipeline-Bot/1.0
   REFRESH_TOKEN_EXPIRE_DAYS=7
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

3. **Initialize database:**
   ```bash
   # Run all migrations (creates tables including token_blacklist for refresh tokens)
   alembic upgrade head
   ```
   
   **Note:** The migration system includes the `token_blacklist` table for refresh token and logout functionality. If you encounter issues, ensure the `alembic/script.py.mako` template file exists.

4. **Run the application:**
   ```bash
   python run.py
   # or
   uvicorn app.main:app --reload
   ```

## Docker Setup

The easiest way to run the application is using Docker Compose:

1. **Create `.env` file:**
   ```bash
   OPENAI_API_KEY=your-openai-api-key-here
   SECRET_KEY=your-secret-key-for-jwt-tokens
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

The Docker setup includes:
- PostgreSQL with pgvector extension
- Automatic database migrations on startup
- Health checks and auto-restart
- Persistent data volumes

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get access/refresh tokens
- `POST /api/v1/auth/refresh` - Refresh access token using refresh token
- `POST /api/v1/auth/logout` - Logout and invalidate access token

### Documents
- `POST /api/v1/digest` - Ingest documents (files and/or URLs)
- `POST /api/v1/digest/urls` - Ingest documents from URLs only (JSON body)
- `DELETE /api/v1/remove` - Remove documents by vector store IDs
- `DELETE /api/v1/remove/by-record/{record_id}` - Remove document by database record ID
- `GET /api/v1/data` - List user's documents

### Chat
- `POST /api/v1/chat` - Stream chat response (SSE)
- `GET /api/v1/history` - Get chat threads
- `GET /api/v1/history/{thread_id}` - Get thread messages

### Memory Management
- `DELETE /api/v1/memory/{thread_id}` - Delete memory for a specific thread (verifies ownership)
- `DELETE /api/v1/memory` - Delete all memory for the current user (user-scoped)

## Usage Examples

### Register and Login
```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login (returns access_token and refresh_token)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Refresh access token
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your-refresh-token-here"}'

# Logout (invalidates access token)
curl -X POST http://localhost:8000/api/v1/auth/logout \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Ingest Documents
```bash
# Upload files (PDF text + embedded tables + embedded images,
# DOCX/TXT text, or direct PNG/JPG/WEBP image uploads)
curl -X POST http://localhost:8000/api/v1/digest \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@document.pdf" \
  -F "files=@document.docx" \
  -F "files=@diagram.png"

# Ingest URLs (using main endpoint)
curl -X POST http://localhost:8000/api/v1/digest \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F 'urls=["https://example.com/article"]'

# Ingest URLs only (simpler JSON endpoint - recommended)
curl -X POST http://localhost:8000/api/v1/digest/urls \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/article", "https://recombd.com/"],
    "metadata": {"source": "web"}
  }'
```

### Remove Documents
```bash
# Remove by vector store document IDs (from /data endpoint)
curl -X DELETE http://localhost:8000/api/v1/remove \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc-id-1", "doc-id-2"]}'

# Remove by database record ID (simpler - from /data endpoint)
curl -X DELETE http://localhost:8000/api/v1/remove/by-record/2 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Chat
```bash
# RAG query (uses ingested documents)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is reward hacking?", "thread_id": null}'

# Tool-style query (uses built-in tools, e.g. date/time or calculator)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is today'\''s date and time?", "thread_id": null}'
```

### Memory Management
```bash
# Delete memory for a specific thread (verifies thread belongs to user)
curl -X DELETE http://localhost:8000/api/v1/memory/{thread_id} \
  -H "Authorization: Bearer YOUR_TOKEN"

# Delete all memory for the current user (only user's threads)
curl -X DELETE http://localhost:8000/api/v1/memory \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Development

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Testing
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Code Quality
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linter
ruff check .

# Format code
black .

# Type checking
mypy app/
```

## CI/CD

This project uses GitHub Actions for continuous integration and deployment.

### CI Pipeline
The CI pipeline runs on every push and pull request:
- **Lint**: Code linting with Ruff and formatting check with Black
- **Test**: Runs pytest test suite with PostgreSQL service
- **Build**: Verifies Docker image builds successfully
- **Security**: Runs Trivy vulnerability scanner

### CD Pipeline
The CD pipeline runs on version tags (e.g., `v1.0.0`):
- **Build and Push**: Builds and pushes Docker image to registries
- **Deploy**: Deploys to production (configure as needed)

See [.github/workflows/README.md](.github/workflows/README.md) for detailed workflow documentation.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for embeddings and chat |
| `SECRET_KEY` | Yes | - | JWT secret key (use strong random string) |
| `USER_AGENT` | No | - | User agent string for web scraping |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | 30 | Access token expiration time |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | 7 | Refresh token expiration time |

### Prompts Configuration

System prompts and RAG workflow prompts are configured in `app/workflows/prompts.json`. You can modify:
- Document grading prompts
- Question rewriting prompts
- Answer generation prompts
- Retriever tool configuration

See [app/workflows/PROMPTS_GUIDE.md](app/workflows/PROMPTS_GUIDE.md) for details.

### Tool Calling

The RAG workflow supports multiple tools: the document retriever plus any custom tools you register. Tools live in `app/workflows/tools/`.

**Built-in example tools** (in `app/workflows/tools/example_tools.py`):

- **get_current_datetime** – Returns the current date and time (optional format string). Use for questions like “What is today’s date?” or “What time is it?”
- **calculate** – Evaluates mathematical expressions (e.g. `2 + 2`, `sqrt(16)`). Use for questions like “Calculate 15 * 23.”

**How to add a new tool:**

1. Create a function with a clear docstring (the LLM uses it to decide when to call the tool) and typed parameters. Use `Annotated[str, "description"]` for parameter descriptions.
2. Decorate it with `@tool_registry.register` in any module under `app/workflows/tools/`.
3. Import that module in `app/workflows/tools/__init__.py` so it is loaded at startup.

Example:

```python
# In app/workflows/tools/my_tools.py
from typing import Annotated
from app.workflows.tools import tool_registry

@tool_registry.register
def my_tool(query: Annotated[str, "What to search for"]) -> str:
    """Short description for the LLM. Use when the user asks about X."""
    return "result"
```

Then add `from app.workflows.tools import my_tools  # noqa: E402, F401` to `app/workflows/tools/__init__.py`.

**Testing tool calling:** Send a chat message such as “What is today’s date and time?” or “Calculate 10 * 5” to confirm the agent uses the tools and returns correct results.

### Retrieval Logging

The system automatically logs retrieval queries and retrieved documents to `retrieval_logs.jsonl`. Use the `view_retrieval_logs.py` script to analyze:

```bash
# View latest entries
python view_retrieval_logs.py

# Search by query
python view_retrieval_logs.py query "your search term"

# Filter by user
python view_retrieval_logs.py user 16

# Get statistics
python view_retrieval_logs.py stats
```

## Production Deployment

See `Dockerfile` and `docker-compose.yml` for containerized deployment.

## Additional Resources

- [DOCKER.md](DOCKER.md) - Detailed Docker setup instructions
- [.github/workflows/README.md](.github/workflows/README.md) - CI/CD pipeline documentation
- [app/workflows/PROMPTS_GUIDE.md](app/workflows/PROMPTS_GUIDE.md) - Prompt configuration guide

## License

MIT

