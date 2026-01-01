# Farmers Livestock Stock Portal

Python + MySQL desktop portal for farm livestock tracking, identification, and health records. Includes AI-assisted health inference and bulk barcode/QR workflows.

## Key Features
- **CSV-driven species/breeds**: Species and breed dropdowns sourced from FAO DAD-IS CSV.
- **Extended fields**: `livestock_type`, `color`, `age`, `purchase_date`, `health_status` (readonly dropdown).
- **Search/sort UI**: Filter livestock list, sortable columns, striped rows, status bar.
- **Bulk operations**: Generate barcodes and QR codes for multi-selected animals.
- **Previews**: In-portal preview for barcode/QR images (scaled to window).
- **Delete entries**: Remove selected livestock records from the database.
- **Health Records module**: Add/view/delete health entries; filter by species/tag.
- **AI health inference**: Auto-updates `health_status` after health record insert.
   - Heuristic fallback always available.
   - Optional ML model (scikit-learn) if installed.

## Requirements
- Python 3.10+
- MySQL server with a database you control
- Python packages (install via `requirements.txt`):
   - `mysql-connector-python`, `Pillow`, `python-barcode`, `qrcode`, `tkcalendar`, `joblib`, `scikit-learn`

Install:

```bash
pip install -r requirements.txt
```

## Configuration
- Update database credentials in [farm_livestock_portal/config/db_config.py](farm_livestock_portal/config/db_config.py).
- Tables are auto-created on first run via the app:
   - `livestock` (with sufficient column sizes, including `breed VARCHAR(255)`).
   - `health_records`.

## Running
Run from the portal folder to avoid import path issues:

```bash
cd farm_livestock_portal
python -m gui.main_app
```

## Seeding Dummy Data
Populate the database with N animals and a sample “sick bison” health record (demonstrates AI update):

```bash
cd farm_livestock_portal
python -m scripts.seed --animals 100
```

## AI Health Inference
- Heuristic engine in [farm_livestock_portal/services/ai_service.py](farm_livestock_portal/services/ai_service.py) maps diagnosis/treatment/severity/notes to a health status.
- Optional ML model in [farm_livestock_portal/services/health_ml.py](farm_livestock_portal/services/health_ml.py): TF‑IDF + LogisticRegression with OneHot features.
- Train the ML model (requires `scikit-learn` and `joblib`):

```bash
cd farm_livestock_portal
python -c "from services.health_ml import train_health_model; train_health_model()"
```

The trained model is saved to [farm_livestock_portal/models/health_model.pkl](farm_livestock_portal/models/health_model.pkl). When present, the portal uses it automatically; otherwise it falls back to heuristics.

## Barcode & QR Codes
- Barcodes (Code128) saved under [farm_livestock_portal/assets/barcodes](farm_livestock_portal/assets/barcodes).
- QR codes saved under [farm_livestock_portal/assets/qrcodes](farm_livestock_portal/assets/qrcodes).
- QR content includes a Google Images search link for the selected `species` + `breed`, aiding quick breed identification online.
- Preview buttons in the GUI open scaled image previews.

## Troubleshooting
- If you see “Data too long for column 'breed'”, rerun the app or seeder; the schema auto‑upgrade increases `breed` to `VARCHAR(255)`.
- Run from the portal folder with `python -m gui.main_app` to avoid import errors.
- If `scikit-learn` is missing, AI health inference uses heuristics automatically.

## Notes
- Tkinter is included with Python on macOS; we use ttk for styling.
- The FAO CSV is at [farm_livestock_portal/fao_dad_list - Species.csv](farm_livestock_portal/fao_dad_list%20-%20Species.csv).
