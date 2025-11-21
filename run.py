"""Run the FastAPI application."""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Control reload via RELOAD env var (default: true for development)
    # Set RELOAD=false to disable auto-reload (e.g., in production)
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    # Get the directory containing this script (refactored/)
    base_dir = Path(__file__).parent.resolve()
    app_dir = base_dir / "app"
    
    # Use absolute path and ensure it exists
    app_dir_str = str(app_dir.resolve())
    
    # Note: watchfiles always watches the current working directory in addition to reload_dirs
    # If you hit file watch limits, increase the system limit:
    #   sudo sysctl fs.inotify.max_user_watches=524288
    #   Or make it permanent: echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        reload_dirs=[app_dir_str] if reload else None,  # Only watch app/ directory
        reload_includes=["*.py"],  # Only watch Python files
        reload_excludes=[
            "**/__pycache__/**",  # Exclude all __pycache__ directories recursively
            "**/*.pyc", "**/*.pyo", "**/*.pyd",  # Compiled Python files (recursive)
            "**/*.egg-info/**",  # Build artifacts
            "**/.pytest_cache/**", "**/.mypy_cache/**",  # Tool caches
            "**/venv/**", "**/env/**", "**/.venv/**",  # Virtual environments
            "**/alembic/versions/**",  # Alembic migration files (don't need to watch)
            "**/tests/**",  # Test files
            "**/*.md", "**/*.txt", "**/*.log",  # Documentation and logs
        ]
    )

