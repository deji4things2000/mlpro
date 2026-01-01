# Farmers Livestock Stock Portal

A Python + MySQL application for tracking livestock on a farm.

## Features
- Livestock registration
- Barcode generation per animal
- Health and age tracking
- Reports dashboard
- GUI-based interface

## Setup
1. Install MySQL and create the database using `database/schema.sql`
2. Update database credentials in `config/db_config.py`
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   python gui/main_app.py

## Barcodes
Generated barcodes are saved in `assets/barcodes/` and can be printed for tagging animals.
