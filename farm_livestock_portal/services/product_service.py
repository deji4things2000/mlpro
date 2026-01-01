# services/product_service.py

import mysql.connector
from config.db_config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def ensure_products_table():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(128) NOT NULL,
                category VARCHAR(64) NULL,
                price DECIMAL(10,2) NOT NULL,
                stock INT NOT NULL DEFAULT 0,
                description VARCHAR(255) NULL
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


def insert_product(name, category, price, stock, description):
    ensure_products_table()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO products (name, category, price, stock, description) VALUES (%s, %s, %s, %s, %s)",
            (name, category, price, stock, description),
        )
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def fetch_products():
    ensure_products_table()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, category, price, stock, description FROM products ORDER BY name ASC")
        return cursor.fetchall()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def update_product(pid, name, category, price, stock, description):
    ensure_products_table()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE products SET name=%s, category=%s, price=%s, stock=%s, description=%s
            WHERE id=%s
            """,
            (name, category, price, stock, description, pid),
        )
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def delete_product(pid):
    ensure_products_table()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM products WHERE id=%s", (pid,))
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
