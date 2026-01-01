import os
from typing import Dict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session
from .models import Animal, Event

REPORTS_PATH = "reports/summary.pdf"

def summary(session: Session) -> Dict:
    total = session.query(Animal).count()
    by_species = {}
    for (sp,) in session.query(Animal.species).distinct().all():
        by_species[sp] = session.query(Animal).filter(Animal.species == sp).count()
    vaccinations = session.query(Event).filter(Event.event_type == "vaccination").count()
    return {"total_animals": total, "by_species": by_species, "vaccinations": vaccinations}

def create_pdf(session: Session) -> str:
    os.makedirs("reports", exist_ok=True)
    stats = summary(session)
    c = canvas.Canvas(REPORTS_PATH, pagesize=A4)
    c.setFont("Helvetica-Bold", 16); c.drawString(50, 800, "Farm Livestock Summary Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Total animals: {stats['total_animals']}")
    c.drawString(50, 750, f"Total vaccinations: {stats['vaccinations']}")
    y = 730; c.drawString(50, y, "By species:"); y -= 20
    for sp, cnt in stats["by_species"].items():
        c.drawString(70, y, f"- {sp}: {cnt}"); y -= 18
    c.showPage(); c.save()
    return REPORTS_PATH
