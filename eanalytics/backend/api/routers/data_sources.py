from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import json
import asyncio

from ..database import get_db, DataSource, User
from ..auth import get_current_user
from ..schemas import DataSourceCreate, DataSourceResponse, ConnectionTestResponse
from ..services.connectors import get_connector

router = APIRouter()

@router.get("/", response_model=List[DataSourceResponse])
async def list_data_sources(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all data sources for the current user."""
    result = await db.execute(
        select(DataSource)
        .where(DataSource.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(DataSource.created_at.desc())
    )
    return result.scalars().all()

@router.get("/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific data source."""
    result = await db.execute(
        select(DataSource)
        .where(
            DataSource.id == source_id,
            DataSource.user_id == current_user.id
        )
    )
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
    
    return source

@router.post("/", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_data_source(
    data_source: DataSourceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new data source."""
    # Create data source
    db_source = DataSource(
        name=data_source.name,
        type=data_source.type,
        connection_config=data_source.connection_config,
        user_id=current_user.id
    )
    
    db.add(db_source)
    await db.commit()
    await db.refresh(db_source)
    
    return db_source

@router.post("/test-connection", response_model=ConnectionTestResponse)
async def test_connection(
    data_source: DataSourceCreate,
    current_user: User = Depends(get_current_user)
):
    """Test connection to a data source."""
    try:
        # Get connector for the data source type
        connector = get_connector(
            source_type=data_source.type,
            config=data_source.connection_config
        )
        
        # Test connection
        success, latency, message = await asyncio.to_thread(
            connector.test_connection
        )
        
        return ConnectionTestResponse(
            success=success,
            latency=latency,
            message=message,
            source_type=data_source.type
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connection test failed: {str(e)}"
        )

@router.delete("/{source_id}")
async def delete_data_source(
    source_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a data source."""
    result = await db.execute(
        select(DataSource)
        .where(
            DataSource.id == source_id,
            DataSource.user_id == current_user.id
        )
    )
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found"
        )
    
    await db.delete(source)
    await db.commit()
    
    return {"message": "Data source deleted successfully"}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file (CSV, Excel, JSON) as a data source."""
    import pandas as pd
    import tempfile
    import os
    
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Read file based on extension
        if file_ext == '.csv':
            df = pd.read_csv(tmp_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(tmp_path)
        elif file_ext == '.json':
            df = pd.read_json(tmp_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(tmp_path)
        
        # Convert to JSON
        data = df.to_dict(orient='records')
        
        # Get column info
        columns = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            columns.append({
                "name": col,
                "type": col_type,
                "sample": df[col].iloc[0] if len(df) > 0 else None
            })
        
        return {
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "preview": data[:100]  # First 100 rows
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading file: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)