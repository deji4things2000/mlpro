# services/health_service.py

import mysql.connector
from config.db_config import DB_CONFIG
from services.ai_service import infer_health_status
try:
    from services.health_ml import predict_health_status
except Exception:
    predict_health_status = None
from services.db_service import update_health_status_by_tag


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def ensure_health_table():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS health_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                animal_tag VARCHAR(64) NOT NULL,
                species VARCHAR(64) NOT NULL,
                record_date DATE NOT NULL,
                diagnosis VARCHAR(128) NULL,
                treatment VARCHAR(128) NULL,
                medication VARCHAR(128) NULL,
                dosage VARCHAR(64) NULL,
                vet VARCHAR(64) NULL,
                lab_result VARCHAR(128) NULL,
                severity VARCHAR(32) NULL,
                notes VARCHAR(255) NULL,
                next_check_date DATE NULL,
                withdrawal_end_date DATE NULL
            )
            """
        )
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def insert_health_record(data):
    """data: (animal_tag, species, record_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check_date, withdrawal_end_date)"""
    try:
        ensure_health_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO health_records (
                animal_tag, species, record_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check_date, withdrawal_end_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            data,
        )
        conn.commit()
        # After insert, infer health status and update portal
        try:
            (tag, species, rec_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check, withdrawal_end) = data
            # Try ML prediction first (if available); fallback to heuristic inference
            status = (predict_health_status(diagnosis, treatment, severity, lab_result, notes) if predict_health_status else None) or infer_health_status(
                diagnosis=diagnosis,
                treatment=treatment,
                severity=severity,
                lab_result=lab_result,
                notes=notes,
                next_check=next_check,
                withdrawal_end=withdrawal_end,
            )
            update_health_status_by_tag(tag, status)
        except Exception:
            # Non-fatal; ignore AI update errors
            pass
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def fetch_health_records(species=None, tag=None):
    try:
        ensure_health_table()
        conn = get_connection()
        cursor = conn.cursor()
        base = "SELECT id, animal_tag, species, record_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check_date, withdrawal_end_date FROM health_records"
        clauses = []
        params = []
        if species:
            clauses.append("species = %s")
            params.append(species)
        if tag:
            clauses.append("animal_tag = %s")
            params.append(tag)
        if clauses:
            base += " WHERE " + " AND ".join(clauses)
        base += " ORDER BY record_date DESC, id DESC"
        cursor.execute(base, tuple(params))
        return cursor.fetchall()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def delete_health_record(record_id):
    try:
        ensure_health_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM health_records WHERE id = %s", (record_id,))
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def update_health_record(record_id, data):
    """Update a health record and re-apply AI health inference for the animal.
    data: (animal_tag, species, record_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check_date, withdrawal_end_date)
    """
    try:
        ensure_health_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE health_records SET
                animal_tag = %s,
                species = %s,
                record_date = %s,
                diagnosis = %s,
                treatment = %s,
                medication = %s,
                dosage = %s,
                vet = %s,
                lab_result = %s,
                severity = %s,
                notes = %s,
                next_check_date = %s,
                withdrawal_end_date = %s
            WHERE id = %s
            """,
            data + (record_id,),
        )
        conn.commit()
        # After update, infer health status and update portal
        try:
            (tag, species, rec_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check, withdrawal_end) = data
            status = (predict_health_status(diagnosis, treatment, severity, lab_result, notes) if predict_health_status else None) or infer_health_status(
                diagnosis=diagnosis,
                treatment=treatment,
                severity=severity,
                lab_result=lab_result,
                notes=notes,
                next_check=next_check,
                withdrawal_end=withdrawal_end,
            )
            update_health_status_by_tag(tag, status)
        except Exception:
            pass
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
