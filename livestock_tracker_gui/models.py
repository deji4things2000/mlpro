from sqlalchemy import String, Integer, Float, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base

# Animal table
class Animal(Base):
    __tablename__ = "animals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    species: Mapped[str] = mapped_column(String(32))
    breed: Mapped[str] = mapped_column(String(64), default="")
    sex: Mapped[str] = mapped_column(String(8), default="")
    dob: Mapped[str] = mapped_column(String(10), default="")      # YYYY-MM-DD
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    location: Mapped[str] = mapped_column(String(64), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    events: Mapped[list["Event"]] = relationship(back_populates="animal", cascade="all, delete-orphan")

# Event table (e.g., vaccination, weight update)
class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag_id: Mapped[str] = mapped_column(String(32), ForeignKey("animals.tag_id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    event_date: Mapped[str] = mapped_column(String(10))            # YYYY-MM-DD
    details: Mapped[str] = mapped_column(Text, default="")
    animal: Mapped[Animal] = relationship(back_populates="events")
