# Create tables and seed sample animals
import os
from livestock_tracker_gui.db import Base, engine, SessionLocal
from livestock_tracker_gui.models import Animal

def main():
    os.makedirs("images/barcodes", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.query(Animal).count() > 0:
            print("DB already has data."); return
        samples = [
            {"tag_id": "A-COW00001", "species": "cow", "breed": "holstein", "sex": "F", "dob": "2024-04-12", "weight_kg": 450.0, "location": "paddock-1"},
            {"tag_id": "A-GOAT0001", "species": "goat", "breed": "boer", "sex": "M", "dob": "2025-02-01", "weight_kg": 35.2, "location": "pen-2"},
            {"tag_id": "A-CHIK0001", "species": "chicken", "breed": "rhode-island", "sex": "F", "dob": "2025-05-20", "weight_kg": 2.1, "location": "coop-3"},
        ]
        for s in samples:
            db.add(Animal(**s))
        db.commit()
    print("Seeded.")

if __name__ == "__main__":
    main()
