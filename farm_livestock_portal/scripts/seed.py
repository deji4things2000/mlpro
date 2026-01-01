# scripts/seed.py
"""Seed dummy data into the Farm Livestock Portal and Health Records.

Usage:
    python -m scripts.seed [--animals 20]

This will insert sample livestock rows and associated health records.
"""

import os
import sys
import csv
import random
import string
from datetime import date, timedelta

# Allow running as module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.db_service import insert_livestock_extended, ensure_livestock_table
from services.health_service import insert_health_record, ensure_health_table

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SPECIES_CSV_PATH = os.path.join(BASE_DIR, "fao_dad_list - Species.csv")

COLORS = ["Black", "White", "Brown", "Red", "Grey", "Tan", "Gold", "Speckled"]
TYPES = ["Dairy", "Beef", "Breeding", "Draft", "Layer", "Broiler", "Wool", "Pack", "Companion"]
HEALTH_STATUSES = ["Healthy", "Sick", "Recovering", "Under Treatment", "Injured", "Unknown"]


def load_species_breeds():
    mapping = {}
    try:
        with open(SPECIES_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = (row.get("Species") or "").strip()
                breed = (row.get("Breed") or "").strip()
                if species and breed:
                    mapping.setdefault(species, []).append(breed)
        if mapping:
            return mapping
    except Exception:
        pass
    return {
        "Cow": ["Holstein", "Jersey", "Angus", "Hereford"],
        "Goat": ["Boer", "Kiko", "Saanen", "Nubian"],
        "Sheep": ["Merino", "Suffolk", "Dorper", "Hampshire"],
        "Pig": ["Berkshire", "Yorkshire", "Landrace", "Duroc"],
        "Chicken": ["Leghorn", "Rhode Island Red", "Plymouth Rock", "Orpington"],
        "Bison": ["Plains", "Wood"],
    }


def rand_tag(species: str) -> str:
    return f"{species[:3].upper()}-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


def rand_date(past_days: int = 1000) -> date:
    return date.today() - timedelta(days=random.randint(0, past_days))


def seed_animals(n: int, mapping: dict):
    species_list = list(mapping.keys())
    animals = []
    for _ in range(n):
        species = random.choice(species_list)
        breed = random.choice(mapping.get(species, ["Mixed"]))
        tag = rand_tag(species)
        age_years = random.randint(1, 12)
        # Approximate DOB based on age_years, with random offset within the year
        dob = date.today() - timedelta(days=age_years * 365 + random.randint(0, 364))
        health = random.choice(HEALTH_STATUSES)
        purchase = rand_date(1800)
        livestock_type = random.choice(TYPES)
        color = random.choice(COLORS)
        data = (
            tag,
            species,
            breed,
            dob,
            health,
            purchase,
            livestock_type,
            color,
        )
        insert_livestock_extended(data)
        animals.append((tag, species, breed))
    return animals


def seed_health_for_animal(tag: str, species: str, diagnosis: str, severity: str = "Moderate"):
    record = (
        tag,
        species,
        rand_date(120),
        diagnosis,
        "Antibiotics",  # treatment
        "Amoxicillin",   # medication
        "500 mg",        # dosage
        "Dr. Vet",       # vet
        "Pending",       # lab_result
        severity,
        "Under observation",  # notes
        rand_date(60),    # next_check_date
        rand_date(45),    # withdrawal_end_date
    )
    insert_health_record(record)


def main():
    # Parse args
    n = 20
    args = sys.argv[1:]
    if "--animals" in args:
        try:
            n = int(args[args.index("--animals") + 1])
        except Exception:
            pass

    # Ensure tables
    ensure_livestock_table()
    ensure_health_table()
    mapping = load_species_breeds()
    animals = seed_animals(n, mapping)

    # Ensure a sick Bison example exists
    # If no Bison created, synthesize one
    bison = next(((t, s, b) for (t, s, b) in animals if s.lower() == "bison"), None)
    if not bison:
        species = "Bison"
        breed = random.choice(mapping.get(species, ["Plains"]))
        tag = rand_tag(species)
        bison_age_years = 6
        bison_dob = date.today() - timedelta(days=bison_age_years * 365 + random.randint(0, 364))
        insert_livestock_extended((tag, species, breed, bison_dob, "Healthy", rand_date(900), "Beef", random.choice(COLORS)))
        bison = (tag, species, breed)
    # Add a sick health record to validate auto-update
    seed_health_for_animal(bison[0], bison[1], "Pneumonia", severity="Severe")

    print(f"Seeded {n} animals and 1 sick Bison health record.")


if __name__ == "__main__":
    main()
