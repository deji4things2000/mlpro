from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from ..database import get_db, Dashboard, User
from ..auth import get_current_user
from ..schemas import UserResponse

router = APIRouter()

@router.get("/", response_model=List[dict])
async def list_dashboards(
	skip: int = 0,
	limit: int = 100,
	current_user: User = Depends(get_current_user),
	db: AsyncSession = Depends(get_db)
):
	result = await db.execute(
		select(Dashboard).where(Dashboard.user_id == current_user.id).offset(skip).limit(limit)
	)
	return [d.__dict__ for d in result.scalars().all()]

@router.get("/{dashboard_id}", response_model=dict)
async def get_dashboard(
	dashboard_id: int,
	current_user: User = Depends(get_current_user),
	db: AsyncSession = Depends(get_db)
):
	result = await db.execute(
		select(Dashboard).where(Dashboard.id == dashboard_id, Dashboard.user_id == current_user.id)
	)
	dashboard = result.scalar_one_or_none()
	if not dashboard:
		raise HTTPException(status_code=404, detail="Dashboard not found")
	return dashboard.__dict__

@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_dashboard(
	dashboard: dict,
	current_user: User = Depends(get_current_user),
	db: AsyncSession = Depends(get_db)
):
	db_dashboard = Dashboard(
		name=dashboard["name"],
		description=dashboard.get("description"),
		user_id=current_user.id,
		layout=dashboard.get("layout", {}),
		is_public=dashboard.get("is_public", False),
		tags=dashboard.get("tags", [])
	)
	db.add(db_dashboard)
	await db.commit()
	await db.refresh(db_dashboard)
	return db_dashboard.__dict__

@router.delete("/{dashboard_id}")
async def delete_dashboard(
	dashboard_id: int,
	current_user: User = Depends(get_current_user),
	db: AsyncSession = Depends(get_db)
):
	result = await db.execute(
		select(Dashboard).where(Dashboard.id == dashboard_id, Dashboard.user_id == current_user.id)
	)
	dashboard = result.scalar_one_or_none()
	if not dashboard:
		raise HTTPException(status_code=404, detail="Dashboard not found")
	await db.delete(dashboard)
	await db.commit()
	return {"message": "Dashboard deleted successfully"}
