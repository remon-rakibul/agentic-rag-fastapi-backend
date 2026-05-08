# GitHub Actions Workflows

This directory contains CI/CD workflows for the RAG Pipeline FastAPI Backend.

## Workflows

### 1. CI (`ci.yml`)
Runs on every push and pull request to main/master/develop branches.

**Jobs:**
- **Lint**: Code linting with Ruff, formatting check with Black, and type checking with mypy
- **Test**: Runs pytest test suite with PostgreSQL service
- **Build**: Builds Docker image to verify it compiles correctly
- **Security**: Runs Trivy vulnerability scanner

### 2. CD (`cd.yml`)
Runs on version tags (v*.*.*) or manual workflow dispatch.

**Jobs:**
- **Build and Push**: Builds and pushes Docker image to Docker Hub and GitHub Container Registry
- **Deploy**: Deploys to production (configure deployment steps as needed)

### 3. Docker Build Test (`docker-build.yml`)
Runs on every push and pull request to verify Docker container can start successfully.

**Jobs:**
- **Docker Build Test**: Builds Docker image and tests container startup with PostgreSQL service

## Required Secrets

Configure these secrets in your GitHub repository settings:

### For CI/CD:
- `OPENAI_API_KEY`: OpenAI API key for testing (optional, can use dummy value for CI)

### For CD (Docker Registry):
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token

### For Deployment:
- `DEPLOYMENT_URL`: Production deployment URL (optional)
- Additional deployment secrets as needed (SSH keys, etc.)

## Setup Instructions

1. **Enable GitHub Actions**: Workflows are automatically enabled when pushed to the repository.

2. **Configure Secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add the required secrets listed above

3. **Test the Pipeline**:
   - Push a commit to trigger CI
   - Create a pull request to see all checks
   - Tag a release (e.g., `v1.0.0`) to trigger CD

## Workflow Status Badge

Add this to your README.md to show workflow status:

```markdown
![CI](https://github.com/your-username/your-repo/workflows/CI/badge.svg)
```

## Customization

### Adjust Python Version
Edit `.github/workflows/ci.yml` and change:
```yaml
python-version: '3.12'
```

### Add More Tests
Add test files to `tests/` directory. They will be automatically discovered by pytest.

### Modify Deployment
Edit `.github/workflows/cd.yml` and customize the `deploy` job steps for your deployment target.

