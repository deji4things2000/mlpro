from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db, User
from ..auth import get_current_user
from ..ai_service.insight_generator import InsightGenerator

router = APIRouter()

@router.post("/insight")
async def generate_insight(
	data: dict,
	current_user: User = Depends(get_current_user),
	db: AsyncSession = Depends(get_db)
):
	generator = InsightGenerator()
	try:
		insight = generator.generate_insight(data)
		return {"insight": insight}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
