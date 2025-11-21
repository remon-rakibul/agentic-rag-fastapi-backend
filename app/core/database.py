"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create SQLAlchemy engine
# Convert to psycopg (psycopg3) format for SQLAlchemy
db_url = settings.DATABASE_URL
# Replace asyncpg scheme with psycopg3
db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
# If it's plain postgresql://, use psycopg3
if db_url.startswith("postgresql://") and "+" not in db_url:
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

