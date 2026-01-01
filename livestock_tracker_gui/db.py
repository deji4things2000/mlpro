import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# 1) Load env and DB URL (supports MySQL or SQL Server)
load_dotenv()
DB_URL = os.getenv("DB_URL", "mysql+pymysql://user:password@localhost:3306/livestock")

class Base(DeclarativeBase):
    pass

# 2) Create engine/session
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
