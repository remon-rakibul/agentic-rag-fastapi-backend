"""Database URI normalization utilities."""
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


def normalize_db_uri_for_asyncpg(db_uri: str) -> str:
    """Normalize database URI to postgresql+asyncpg:// format required by PGEngine.
    
    Handles conversion from postgresql:// or postgres:// to postgresql+asyncpg://
    and removes query parameters that asyncpg doesn't support (like sslmode).
    """
    # Parse the URI
    parsed = urlparse(db_uri)
    
    # Determine the scheme
    if parsed.scheme in ("postgresql+asyncpg", "postgresql+psycopg", "postgresql+psycopg2", "postgresql+psycopg3"):
        # Already has a driver specified, but might need to change to asyncpg
        if parsed.scheme != "postgresql+asyncpg":
            scheme = "postgresql+asyncpg"
        else:
            scheme = parsed.scheme
    elif parsed.scheme in ("postgresql", "postgres"):
        scheme = "postgresql+asyncpg"
    else:
        raise ValueError(
            f"Unsupported database URI scheme. Expected postgresql:// or postgres://, "
            f"got: {parsed.scheme}"
        )
    
    # Parse and filter query parameters
    # asyncpg doesn't support sslmode parameter - it handles SSL differently
    query_params = parse_qs(parsed.query)
    
    # Remove sslmode and other asyncpg-incompatible parameters
    filtered_params = {
        k: v for k, v in query_params.items() 
        if k not in ("sslmode",)  # Add other incompatible params here if needed
    }
    
    # Reconstruct the URI
    new_query = urlencode(filtered_params, doseq=True) if filtered_params else ""
    
    normalized = urlunparse((
        scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    
    return normalized

