"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from app.core.database import get_db
from app.core.security import (
    verify_password, 
    get_password_hash, 
    create_access_token,
    create_refresh_token,
    decode_token,
    blacklist_token,
    is_token_blacklisted,
    TOKEN_TYPE_REFRESH
)
from app.api.dependencies import get_current_user
from app.core.config import settings
from app.models.database import User
from app.models.schemas import UserCreate, UserResponse, Token, TokenRefresh, LogoutResponse

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Validate input
        if not user_data.email or not user_data.email.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email cannot be empty"
            )
        
        if not user_data.password or not user_data.password.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password cannot be empty"
            )
        
        # Validate password strength
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Normalize email (lowercase, trim)
        normalized_email = user_data.email.strip().lower()
        
        # Check if user already exists
        try:
            existing_user = db.query(User).filter(User.email == normalized_email).first()
        except Exception as db_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database query error: {str(db_error)}"
            )
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        try:
            hashed_password = get_password_hash(user_data.password)
        except Exception as hash_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Password hashing error: {str(hash_error)}"
            )
        
        # Create new user
        try:
            user = User(
                email=normalized_email,
                hashed_password=hashed_password,
                created_at=datetime.now(timezone.utc)  # Set explicitly to ensure it's available
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        except Exception as db_error:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user in database: {str(db_error)}"
            )
        
        return user
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Rollback on any unexpected error
        db.rollback()
        import traceback
        error_detail = f"Unexpected registration error: {str(e)}"
        print(f"❌ REGISTRATION ERROR: {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.post("/login", response_model=Token)
def login(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Login and get access token.
    
    **Important for Swagger UI:**
    After logging in, copy ONLY the `access_token` value (not the entire JSON object).
    Then click the "Authorize" button at the top of the page and paste just the token value.
    
    Example: If the response is `{"access_token": "eyJhbGc...", "token_type": "bearer"}`,
    copy only `eyJhbGc...` (without quotes) into the Bearer token field.
    """
    try:
        # Validate input
        if not user_data.email or not user_data.email.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email cannot be empty"
            )
        
        if not user_data.password or not user_data.password.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password cannot be empty"
            )
        
        # Query user from database
        try:
            user = db.query(User).filter(User.email == user_data.email.strip().lower()).first()
        except Exception as db_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(db_error)}"
            )
        
        # Verify user exists
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        try:
            password_valid = verify_password(user_data.password, user.hashed_password)
        except Exception as verify_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Password verification error: {str(verify_error)}"
            )
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive. Please contact support."
            )
        
        # Create access and refresh tokens
        try:
            # Note: python-jose requires 'sub' to be a string, not an integer
            token_data = {"sub": str(user.id)}  # Convert to string for JWT compliance
            
            access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data=token_data,
                expires_delta=access_token_expires
            )
            
            refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            refresh_token = create_refresh_token(
                data=token_data,
                expires_delta=refresh_token_expires
            )
        except Exception as token_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create tokens: {str(token_error)}"
            )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        import traceback
        error_detail = f"Unexpected login error: {str(e)}"
        print(f"❌ LOGIN ERROR: {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.post("/refresh", response_model=Token)
def refresh_token(
    refresh_data: TokenRefresh,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Validates the refresh token and returns a new access token and refresh token.
    """
    try:
        # Decode refresh token
        payload = decode_token(refresh_data.refresh_token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if token is blacklisted
        if is_token_blacklisted(refresh_data.refresh_token, db):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token type
        token_type = payload.get("type")
        if token_type != TOKEN_TYPE_REFRESH:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Refresh token required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user ID from token
        user_id_raw = payload.get("sub")
        if user_id_raw is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        # Validate user exists and is active
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID in token",
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )
        
        # Create new tokens
        token_data = {"sub": str(user.id)}
        
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data=token_data,
            expires_delta=access_token_expires
        )
        
        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = create_refresh_token(
            data=token_data,
            expires_delta=refresh_token_expires
        )
        
        # Optionally blacklist the old refresh token (rotate refresh tokens)
        # This is a security best practice but optional
        try:
            old_token_expires = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
            blacklist_token(refresh_data.refresh_token, db, expires_at=old_token_expires)
        except Exception:
            # If blacklisting fails, continue anyway (non-critical)
            pass
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Unexpected refresh token error: {str(e)}"
        print(f"❌ REFRESH TOKEN ERROR: {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.post("/logout", response_model=LogoutResponse)
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Logout the current user by blacklisting their access token.
    
    The token will be invalidated and cannot be used for further requests.
    """
    try:
        token = credentials.credentials
        
        # Decode token to get expiration
        payload = decode_token(token)
        expires_at = None
        if payload and "exp" in payload:
            expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        else:
            # Default expiration if can't decode
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Blacklist the token
        blacklist_token(token, db, expires_at=expires_at)
        
        return {"message": "Successfully logged out"}
    
    except Exception as e:
        import traceback
        error_detail = f"Unexpected logout error: {str(e)}"
        print(f"❌ LOGOUT ERROR: {error_detail}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

