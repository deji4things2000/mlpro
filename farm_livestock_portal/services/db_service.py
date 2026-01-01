# services/db_service.py

import mysql.connector
from config.db_config import DB_CONFIG

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

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
