"""Security utilities for authentication and authorization."""
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from app.core.config import settings

# Use bcrypt directly instead of passlib to avoid backend detection issues
# This is more reliable and avoids the compatibility issues between passlib and bcrypt versions

# Token type constants
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    # Ensure password is bytes for bcrypt
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    
    try:
        return bcrypt.checkpw(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    # Ensure password is a string
    if not isinstance(password, str):
        password = str(password)
    
    # Convert to bytes for bcrypt (bcrypt requires bytes)
    password_bytes = password.encode('utf-8')
    
    # Bcrypt has a 72-byte limit - truncate if necessary (shouldn't happen with normal passwords)
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Generate salt and hash password
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string (bcrypt returns bytes)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "type": TOKEN_TYPE_ACCESS
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "type": TOKEN_TYPE_REFRESH
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as e:
        # Log the error for debugging (in production, you might want to use a logger)
        # For now, just return None to indicate invalid token
        return None
    except Exception as e:
        # Catch any other exceptions (e.g., invalid token format)
        return None


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token (access or refresh)."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None
    except Exception:
        return None


def is_token_blacklisted(token: str, db) -> bool:
    """Check if a token is in the blacklist."""
    from app.models.database import TokenBlacklist
    from sqlalchemy.orm import Session
    
    if isinstance(db, Session):
        blacklisted = db.query(TokenBlacklist).filter(
            TokenBlacklist.token == token,
            TokenBlacklist.expires_at > datetime.now(timezone.utc)
        ).first()
        return blacklisted is not None
    return False


def blacklist_token(token: str, db, expires_at: Optional[datetime] = None) -> None:
    """Add a token to the blacklist."""
    from app.models.database import TokenBlacklist
    from sqlalchemy.orm import Session
    
    if not isinstance(db, Session):
        return
    
    # If expires_at not provided, decode token to get expiration
    if expires_at is None:
        payload = decode_token(token)
        if payload and "exp" in payload:
            expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        else:
            # Default to 30 days if can't decode
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    
    # Check if token already blacklisted
    existing = db.query(TokenBlacklist).filter(TokenBlacklist.token == token).first()
    if existing:
        return  # Already blacklisted
    
    blacklist_entry = TokenBlacklist(
        token=token,
        expires_at=expires_at
    )
    db.add(blacklist_entry)
    db.commit()

