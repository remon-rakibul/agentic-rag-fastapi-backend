# RAG Pipeline FastAPI Backend

Production-ready FastAPI backend for LangGraph RAG pipeline with authentication, document ingestion, and streaming chat.

## Features

- 🔐 JWT-based authentication with refresh tokens
- 📄 Multi-format document ingestion (PDF, DOCX, URLs, TXT)
- 🗑️ Document management (list, delete by ID or record)
- 💬 Streaming chat with RAG workflow
- 📜 Chat history with thread management
- 🐘 PostgreSQL with pgvector for vector storage
- 🔄 LangGraph checkpointing for conversation state
- 🧠 Memory management (delete thread memory, user-scoped cleanup)
- 📊 Retrieval logging for debugging and improvement
- ⚙️ Configurable prompts via JSON file

## Architecture

```
rag-agent-fastapi-backend/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Configuration, security, database
│   ├── models/           # Pydantic schemas and SQLAlchemy models
│   ├── services/         # Business logic layer
│   ├── workflows/       # LangGraph workflow components
│   └── utils/            # Utility functions
├── alembic/              # Database migrations
└── tests/                # Test suite
```

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
# Upload files
curl -X POST http://localhost:8000/api/v1/digest \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@document.pdf" \
  -F "files=@document.docx"

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
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is reward hacking?", "thread_id": null}'
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

