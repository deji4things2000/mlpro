# Authentication logic for FastAPI backend
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from .config import settings
from .database import get_db, User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def verify_password(plain_password, hashed_password):
	return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
	return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
	to_encode = data.copy()
	expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
	to_encode.update({"exp": expire})
	encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
	return encoded_jwt

async def authenticate_user(db: AsyncSession, username: str, password: str):
	result = await db.execute(select(User).where(User.username == username))
	user = result.scalar_one_or_none()
	if not user or not verify_password(password, user.hashed_password):
		return None
	return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
	credentials_exception = HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail="Could not validate credentials",
		headers={"WWW-Authenticate": "Bearer"},
	)
	try:
		payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
		username: str = payload.get("sub")
		if username is None:
			raise credentials_exception
	except JWTError:
		raise credentials_exception
	result = await db.execute(select(User).where(User.username == username))
	user = result.scalar_one_or_none()
	if user is None:
		raise credentials_exception
	return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
	if not current_user.is_active:
		raise HTTPException(status_code=400, detail="Inactive user")
	return current_user

# FastAPI router for authentication endpoints
from fastapi import APIRouter
router = APIRouter()

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
	user = await authenticate_user(db, form_data.username, form_data.password)
	if not user:
		raise HTTPException(status_code=400, detail="Incorrect username or password")
	access_token = create_access_token(data={"sub": user.username})
	return {"access_token": access_token, "token_type": "bearer"}
