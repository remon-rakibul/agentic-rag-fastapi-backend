"""Pytest configuration and fixtures."""
import pytest
import os
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment variables before importing app
os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL", 
    "postgresql://langchain:langchain@localhost:5432/langchain"
)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")
os.environ["SECRET_KEY"] = os.getenv("SECRET_KEY", "test-secret-key-for-testing")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "RAG-Pipeline-Test/1.0")
os.environ["DEBUG"] = "true"

from app.core.database import Base, get_db
from app.main import app


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session."""
    # Use in-memory SQLite for faster tests (if not using PostgreSQL)
    # For PostgreSQL tests, use the actual database
    database_url = os.getenv("TEST_DATABASE_URL")
    
    if database_url and "postgresql" in database_url:
        # Use PostgreSQL
        engine = create_engine(database_url, poolclass=StaticPool)
    else:
        # Use in-memory SQLite for unit tests
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Create authentication headers for testing."""
    # Register a test user
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    
    if login_response.status_code == 200:
        token = login_response.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    return {}

