#!/bin/bash
set -e

echo "Starting RAG Pipeline API..."

# Wait for PostgreSQL to be ready
echo "Waiting for database to be ready..."
until pg_isready -h postgres -U langchain -d langchain 2>/dev/null; do
    echo "Database is unavailable - sleeping"
    sleep 1
done

echo "Database is ready!"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting FastAPI application..."
exec python run.py

