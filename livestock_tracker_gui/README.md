# Livestock Tracker (PyQt + SQL)

Track farm livestock with a desktop GUI:
- Add/update/list animals and events
- Generate Code128 barcode PNGs per tag
- Create a PDF summary report

## Setup (macOS)
```bash
cd /Users/user_1/mlpro/livestock_tracker_gui
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Database
- MySQL (easy on macOS):
```bash
brew install mysql
mysql -u root -p -e "CREATE DATABASE livestock;"
# edit .env DB_URL for your user/pass
```
- SQL Server (optional):
```bash
brew install unixodbc
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
ACCEPT_EULA=Y brew install msodbcsql17
# set DB_URL to mssql+pyodbc://... in .env
```

## Run
```bash
# create tables and seed sample data
python -m livestock_tracker_gui.scripts.seed
# start GUI
python -m livestock_tracker_gui.main
```

## Outputs
- images/barcodes/<TAG>.png
- reports/summary.pdf

## Notes
- tag IDs are short alphanumeric (e.g., A-ABC12345) and are barcode-ready.
- Extend models with parents, vaccinations schedule, weight history, etc.
