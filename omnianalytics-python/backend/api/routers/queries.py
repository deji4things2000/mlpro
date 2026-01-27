from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import json
import asyncio
import uuid

from ..database import get_db, QueryExecution, DataSource, User
from ..auth import get_current_user
from ..schemas import QueryRequest, QueryResponse, CrossSourceQueryRequest
from ..services.query_engine import QueryEngine
from ..services.cache import cache_result, get_cached_result

router = APIRouter()

@router.post("/execute", response_model=QueryResponse)
async def execute_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute a query on a data source."""
    
    # Check cache first
    cache_key = f"query:{current_user.id}:{hash(str(query_request.dict()))}"
    cached_result = await get_cached_result(cache_key)
    
    if cached_result and query_request.use_cache:
        return QueryResponse(
            **cached_result,
            cache_hit=True
        )
    
    # Get data source
    result = await db.execute(
        select(DataSource)
        .where(
            DataSource.id == query_request.data_source_id,
            DataSource.user_id == current_user.id
        )
    )
    data_source = result.scalar_one_or_none()
    
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
    
    # Create query execution record
    db_query = QueryExecution(
        query=query_request.query,
        data_source_id=data_source.id,
        user_id=current_user.id,
        status="running"
    )
    
    db.add(db_query)
    await db.commit()
    await db.refresh(db_query)
    
    # Execute query in background
    background_tasks.add_task(
        execute_query_async,
        db_query.id,
        data_source,
        query_request,
        cache_key,
        query_request.cache_ttl or 300
    )
    
    return QueryResponse(
        query_id=db_query.id,
        status="running",
        message="Query execution started"
    )

@router.post("/execute-cross-source", response_model=QueryResponse)
async def execute_cross_source_query(
    query_request: CrossSourceQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Execute a query across multiple data sources."""
    
    # Create query execution record
    db_query = QueryExecution(
        query=json.dumps(query_request.dict()),
        data_source_id=None,  # No single data source
        user_id=current_user.id,
        status="running"
    )
    
    db.add(db_query)
    await db.commit()
    await db.refresh(db_query)
    
    # Execute cross-source query in background
    background_tasks.add_task(
        execute_cross_source_query_async,
        db_query.id,
        query_request,
        current_user.id,
        db
    )
    
    return QueryResponse(
        query_id=db_query.id,
        status="running",
        message="Cross-source query execution started"
    )

@router.get("/{query_id}", response_model=QueryResponse)
async def get_query_result(
    query_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get query execution result."""
    result = await db.execute(
        select(QueryExecution)
        .where(
            QueryExecution.id == query_id,
            QueryExecution.user_id == current_user.id
        )
    )
    query_exec = result.scalar_one_or_none()
    
    if not query_exec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    return QueryResponse(
        query_id=query_exec.id,
        status=query_exec.status,
        data=query_exec.result,
        error_message=query_exec.error_message,
        execution_time=query_exec.execution_time,
        cache_hit=query_exec.cache_hit,
        created_at=query_exec.created_at
    )

@router.get("/", response_model=List[QueryResponse])
async def list_queries(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List query executions for the current user."""
    result = await db.execute(
        select(QueryExecution)
        .where(QueryExecution.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(QueryExecution.created_at.desc())
    )
    
    queries = result.scalars().all()
    
    return [
        QueryResponse(
            query_id=q.id,
            status=q.status,
            data=q.result,
            error_message=q.error_message,
            execution_time=q.execution_time,
            cache_hit=q.cache_hit,
            created_at=q.created_at
        )
        for q in queries
    ]

# Background task functions
async def execute_query_async(
    query_id: int,
    data_source: DataSource,
    query_request: QueryRequest,
    cache_key: str,
    cache_ttl: int
):
    """Execute query asynchronously."""
    from ..database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        try:
            # Get query engine
            query_engine = QueryEngine()
            
            # Execute query
            start_time = asyncio.get_event_loop().time()
            result = await query_engine.execute_single_source(
                data_source=data_source,
                query=query_request.query,
                params=query_request.params
            )
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000  # ms
            
            # Update query execution
            query_exec = await session.get(QueryExecution, query_id)
            query_exec.status = "completed"
            query_exec.result = result
            query_exec.execution_time = execution_time
            
            await session.commit()
            
            # Cache result
            await cache_result(
                key=cache_key,
                result={
                    "data": result,
                    "execution_time": execution_time,
                    "status": "completed"
                },
                ttl=cache_ttl
            )
            
        except Exception as e:
            # Update query execution with error
            query_exec = await session.get(QueryExecution, query_id)
            query_exec.status = "failed"
            query_exec.error_message = str(e)
            
            await session.commit()

async def execute_cross_source_query_async(
    query_id: int,
    query_request: CrossSourceQueryRequest,
    user_id: int,
    db: AsyncSession
):
    """Execute cross-source query asynchronously."""
    try:
        # Get query engine
        query_engine = QueryEngine()
        
        # Execute cross-source query
        start_time = asyncio.get_event_loop().time()
        result = await query_engine.execute_cross_source(
            query_request=query_request,
            user_id=user_id,
            db=db
        )
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000  # ms
        
        # Update query execution
        query_exec = await db.get(QueryExecution, query_id)
        query_exec.status = "completed"
        query_exec.result = result
        query_exec.execution_time = execution_time
        
        await db.commit()
        
    except Exception as e:
        # Update query execution with error
        query_exec = await db.get(QueryExecution, query_id)
        query_exec.status = "failed"
        query_exec.error_message = str(e)
        
        await db.commit()