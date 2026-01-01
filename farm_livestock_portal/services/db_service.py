# services/db_service.py

import mysql.connector
from config.db_config import DB_CONFIG

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def ensure_columns():
    """Ensure optional columns exist for extended attributes."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # Check existing columns
        cursor.execute("SHOW COLUMNS FROM livestock")
        columns = {row[0] for row in cursor.fetchall()}
        alters = []
        if 'livestock_type' not in columns:
            alters.append("ADD COLUMN livestock_type VARCHAR(64) NULL")
        if 'color' not in columns:
            alters.append("ADD COLUMN color VARCHAR(64) NULL")
        if alters:
            cursor.execute(f"ALTER TABLE livestock {', '.join(alters)}")
            conn.commit()
    except mysql.connector.Error as err:
        # Non-fatal; inserts will omit extended fields if schema can't be altered
        print(f"Schema ensure warning: {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def insert_livestock(data):
    """Insert new livestock record into MySQL"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO livestock 
            (animal_tag, animal_type, breed, age, health_status, purchase_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, data)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        cursor.close()
        conn.close()

def insert_livestock_extended(data):
    """Insert livestock including optional type/color. Ensures columns exist."""
    try:
        ensure_columns()
        conn = get_connection()
        cursor = conn.cursor()
        # Try extended insert first
        query_ext = """
            INSERT INTO livestock 
            (animal_tag, animal_type, breed, age, health_status, purchase_date, livestock_type, color)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(query_ext, data)
        except mysql.connector.Error:
            # Fallback to basic insert if extended columns not available
            query_basic = """
                INSERT INTO livestock 
                (animal_tag, animal_type, breed, age, health_status, purchase_date)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query_basic, data[:6])
        conn.commit()
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def fetch_all_livestock():
    """Fetch all livestock records"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM livestock")
        rows = cursor.fetchall()
        return rows
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

def delete_livestock_by_id(livestock_id):
    """Delete a livestock record by its ID"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM livestock WHERE id = %s", (livestock_id,))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        raise
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
