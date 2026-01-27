# Pydantic models for FastAPI backend
from pydantic import BaseModel, Field, EmailStr
from typing import Any, Dict, List, Optional
from datetime import datetime

class UserBase(BaseModel):
	email: EmailStr
	username: str
	full_name: Optional[str] = None

class UserCreate(UserBase):
	password: str

class UserResponse(UserBase):
	id: int
	is_active: bool
	is_superuser: bool
	created_at: datetime
	updated_at: Optional[datetime] = None

	class Config:
		orm_mode = True

class DataSourceCreate(BaseModel):
	name: str
	type: str
	connection_config: Dict[str, Any]

class DataSourceResponse(DataSourceCreate):
	id: int
	user_id: int
	is_active: bool
	last_connected: Optional[datetime] = None
	created_at: datetime
	updated_at: Optional[datetime] = None

	class Config:
		orm_mode = True

class ConnectionTestResponse(BaseModel):
	success: bool
	latency: float
	message: str
	source_type: str

class QueryRequest(BaseModel):
	data_source_id: int
	query: str
	params: Optional[Dict[str, Any]] = None
	use_cache: bool = True
	cache_ttl: Optional[int] = 300

class QueryResponse(BaseModel):
	query_id: int
	status: str
	data: Optional[Any] = None
	error_message: Optional[str] = None
	execution_time: Optional[float] = None
	cache_hit: Optional[bool] = False
	created_at: Optional[datetime] = None
	message: Optional[str] = None

class CrossSourceQueryRequest(BaseModel):
	sources: List[int]
	queries: List[str]
	join_on: Optional[str] = None
