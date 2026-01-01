from services.db_service import fetch_all_livestock

def generate_report():
    data = fetch_all_livestock()
    report = []
    for row in data:
        report.append(
            f"ID: {row[0]} | Tag: {row[1]} | Type: {row[2]} | Breed: {row[3]} | Age: {row[4]} | Health: {row[5]}"
        )
    return report
