# services/db_service.py

import mysql.connector
from config.db_config import DB_CONFIG

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def ensure_livestock_table():
    """Create livestock table if it doesn't exist."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS livestock (
                id INT AUTO_INCREMENT PRIMARY KEY,
                animal_tag VARCHAR(64) NOT NULL,
                animal_type VARCHAR(64) NOT NULL,
                breed VARCHAR(128) NULL,
                age INT NULL,
                health_status VARCHAR(64) NULL,
                purchase_date DATE NULL,
                livestock_type VARCHAR(64) NULL,
                color VARCHAR(64) NULL
            )
            """
        )
        conn.commit()
    except mysql.connector.Error as err:
        print(f"MySQL Error (ensure table): {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
def ensure_columns():
    """Ensure optional columns exist for extended attributes."""
    try:
        ensure_livestock_table()
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
        if 'date_of_birth' not in columns:
            alters.append("ADD COLUMN date_of_birth DATE NULL")
        if 'created_by' not in columns:
            alters.append("ADD COLUMN created_by VARCHAR(150) NULL")
        if alters:
            cursor.execute(f"ALTER TABLE livestock {', '.join(alters)}")
            conn.commit()

        # Ensure breed column has sufficient length to accommodate FAO names
        try:
            cursor.execute(
                """
                SELECT CHARACTER_MAXIMUM_LENGTH
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'livestock'
                  AND COLUMN_NAME = 'breed'
                """
            )
            result = cursor.fetchone()
            if result and result[0] is not None and int(result[0]) < 255:
                cursor.execute("ALTER TABLE livestock MODIFY COLUMN breed VARCHAR(255) NULL")
                conn.commit()
        except mysql.connector.Error as err:
            # Non-fatal; continue even if information_schema query fails
            print(f"Schema length check warning: {err}")
    except mysql.connector.Error as err:
        # Non-fatal; inserts will omit extended fields if schema can't be altered
        print(f"Schema ensure warning: {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def ensure_indexes():
    """Ensure useful indexes exist for integrity and performance."""
    try:
        ensure_livestock_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT INDEX_NAME FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'livestock'
            """
        )
        existing = {row[0] for row in cursor.fetchall()}
        # Unique tag
        if 'ux_livestock_tag' not in existing:
            try:
                cursor.execute("CREATE UNIQUE INDEX ux_livestock_tag ON livestock(animal_tag)")
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Index create warning (tag): {err}")
        # Species/type index
        if 'idx_livestock_species' not in existing:
            try:
                cursor.execute("CREATE INDEX idx_livestock_species ON livestock(animal_type)")
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Index create warning (species): {err}")
        # Purchase date index
        if 'idx_livestock_purchase' not in existing:
            try:
                cursor.execute("CREATE INDEX idx_livestock_purchase ON livestock(purchase_date)")
                conn.commit()
            except mysql.connector.Error as err:
                print(f"Index create warning (purchase): {err}")
    except mysql.connector.Error as err:
        print(f"Index ensure warning: {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def insert_livestock(data):
    """Insert new livestock record into MySQL"""
    try:
        ensure_livestock_table()
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
    """Insert livestock including optional type/color and date_of_birth. Ensures columns exist."""
    try:
        ensure_columns()
        ensure_indexes()
        conn = get_connection()
        cursor = conn.cursor()
        # Try extended insert first
        query_ext = """
            INSERT INTO livestock 
            (animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date, livestock_type, color, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            # Map to basic fields: skip date_of_birth, optional type/color, and created_by
            basic = (data[0], data[1], data[2], data[3], data[6], data[7])
            cursor.execute(query_basic, basic)
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
    """Fetch all livestock records with explicit column order including date_of_birth."""
    try:
        ensure_columns()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date
            FROM livestock
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()
        return rows
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return []
    finally:
        cursor.close()
        conn.close()

def fetch_livestock_for_user(username: str | None, is_admin: bool = False):
    """Fetch livestock rows for UI. Admin sees all; users see only records they created."""
    try:
        ensure_columns()
        ensure_indexes()
        conn = get_connection()
        cursor = conn.cursor()
        if is_admin:
            cursor.execute(
                """
                SELECT id, animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date
                FROM livestock
                ORDER BY id ASC
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date
                FROM livestock
                WHERE created_by = %s
                ORDER BY id ASC
                """,
                (username or "",),
            )
        rows = cursor.fetchall()
        return rows
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return []
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

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

def get_livestock_by_id(livestock_id):
    """Fetch a single livestock record by ID, including optional columns if present."""
    try:
        ensure_columns()
        conn = get_connection()
        cursor = conn.cursor()
        # Attempt to fetch extended columns first
        try:
            cursor.execute(
                """
                SELECT id, animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date, livestock_type, color, created_by
                FROM livestock WHERE id = %s
                """,
                (livestock_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                return row
        except mysql.connector.Error:
            # Fallback to basic set of columns
            cursor.execute(
                "SELECT id, animal_tag, animal_type, breed, age, health_status, purchase_date FROM livestock WHERE id = %s",
                (livestock_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                # Pad missing optional fields with None (dob, livestock_type, color, created_by)
                return row + (None, None, None, None)
        return None
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return None
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

def update_livestock_extended(data_with_id):
    """Update livestock record by ID, including optional type/color and date_of_birth when available.
    data_with_id: (animal_tag, animal_type, breed, age, date_of_birth, health_status, purchase_date, livestock_type, color, id)
    """
    try:
        ensure_columns()
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE livestock SET
                    animal_tag = %s,
                    animal_type = %s,
                    breed = %s,
                    age = %s,
                    date_of_birth = %s,
                    health_status = %s,
                    purchase_date = %s,
                    livestock_type = %s,
                    color = %s
                WHERE id = %s
                """,
                data_with_id,
            )
        except mysql.connector.Error:
            # Fallback to basic update without optional fields
            cursor.execute(
                """
                UPDATE livestock SET
                    animal_tag = %s,
                    animal_type = %s,
                    breed = %s,
                    age = %s,
                    health_status = %s,
                    purchase_date = %s
                WHERE id = %s
                """,
                (data_with_id[0], data_with_id[1], data_with_id[2], data_with_id[3], data_with_id[5], data_with_id[6], data_with_id[-1]),
            )
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

def update_date_of_birth_by_id(livestock_id, dob):
    """Update only the date_of_birth field for a livestock record."""
    try:
        ensure_columns()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE livestock SET date_of_birth = %s WHERE id = %s",
            (dob, livestock_id),
        )
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
def update_health_status_by_tag(animal_tag, status):
    """Update health_status for a livestock entry by its tag."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE livestock SET health_status = %s WHERE animal_tag = %s",
            (status, animal_tag),
        )
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
