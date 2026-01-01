# scripts/backfill_dob.py

import os
import sys
import hashlib
import random
import calendar
from datetime import date, timedelta

# Ensure project root on path
try:
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
except Exception:
    pass

from services.db_service import fetch_all_livestock, update_date_of_birth_by_id


def seeded_dob_for(tag: str, age: int | None) -> date:
    today = date.today()
    # Deterministic seed from tag
    h = hashlib.sha256((tag or "").encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    rng = random.Random(seed)

    if age is None:
        # Choose a plausible age 1..7 deterministically
        age = rng.randint(1, 7)

    dob_year = today.year - int(age)
    # Pick month up to current month to keep age consistent
    month = rng.randint(1, today.month)
    days_in_month = calendar.monthrange(dob_year, month)[1]
    max_day = today.day if month == today.month else days_in_month
    day = rng.randint(1, max_day)
    return date(dob_year, month, day)


def main():
    rows = fetch_all_livestock()
    updated = 0
    skipped = 0
    for r in rows:
        # r: (id, tag, type, breed, age, date_of_birth, health_status, purchase_date)
        rid, tag, _type, _breed, age, dob, _health, _purchase = r
        if dob is not None:
            skipped += 1
            continue
        try:
            new_dob = seeded_dob_for(tag, age)
            update_date_of_birth_by_id(rid, new_dob)
            updated += 1
        except Exception as e:
            print(f"Failed to update DOB for id={rid} tag={tag}: {e}")
    print(f"DOB backfill complete: updated={updated}, skipped(existing)={skipped}")


if __name__ == "__main__":
    main()
