# services/user_service.py

import os
import binascii
import hashlib
import bcrypt
import random
import smtplib
from email.message import EmailMessage
import mysql.connector
from datetime import datetime, timedelta
from config.db_config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def ensure_users_table():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(64) NOT NULL UNIQUE,
                email VARCHAR(128) NOT NULL UNIQUE,
                phone VARCHAR(32) NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(32) NOT NULL DEFAULT 'user',
                is_verified TINYINT(1) NOT NULL DEFAULT 0,
                verification_code VARCHAR(16) NULL,
                verification_expires DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def ensure_user_columns():
    """Add missing columns to users table for verification features."""
    try:
        ensure_users_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM users")
        columns = {row[0] for row in cursor.fetchall()}
        alters = []
        if 'phone' not in columns:
            alters.append("ADD COLUMN phone VARCHAR(32) NULL")
        if 'is_verified' not in columns:
            alters.append("ADD COLUMN is_verified TINYINT(1) NOT NULL DEFAULT 0")
        if 'verification_code' not in columns:
            alters.append("ADD COLUMN verification_code VARCHAR(16) NULL")
        if 'verification_expires' not in columns:
            alters.append("ADD COLUMN verification_expires DATETIME NULL")
        if 'role' not in columns:
            alters.append("ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'user'")
        if alters:
            cursor.execute(f"ALTER TABLE users {', '.join(alters)}")
            conn.commit()
    except mysql.connector.Error as err:
        print(f"Users schema ensure warning: {err}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def _hash_password(password: str, salt: bytes | None = None) -> str:
    # Use bcrypt for secure hashing with built-in salt
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(stored: str, password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return False


def insert_user(username: str, email: str, password: str, phone: str | None = None):
    ensure_user_columns()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        ph = _hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, phone, password_hash) VALUES (%s, %s, %s, %s)",
            (username, email, phone, ph),
        )
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return False
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def verify_user(identifier: str, password: str):
    ensure_user_columns()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # Lookup by username or email
        cursor.execute(
            "SELECT id, username, email, phone, password_hash, role, is_verified FROM users WHERE username = %s OR email = %s",
            (identifier, identifier),
        )
        row = cursor.fetchone()
        if not row:
            return None
        uid, username, email, phone, stored, role, is_verified = row
        if _verify_password(stored, password):
            return {"id": uid, "username": username, "email": email, "phone": phone, "role": role, "is_verified": bool(is_verified)}
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

def ensure_default_admin():
    """Create a default admin user if missing."""
    ensure_user_columns()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=%s", ("admin",))
        exists = cursor.fetchone()
        if not exists:
            # Default password: admin123
            ph = _hash_password("admin123")
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role, is_verified) VALUES (%s, %s, %s, %s, %s)",
                ("admin", "admin@example.com", ph, "admin", 1),
            )
            conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def _generate_code(length: int = 6) -> str:
    digits = "0123456789"
    return "".join(random.choice(digits) for _ in range(length))


def _send_verification_email(to_email: str, code: str) -> bool:
    try:
        host = os.environ.get("SMTP_HOST")
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER")
        pwd = os.environ.get("SMTP_PASS")
        from_email = os.environ.get("FROM_EMAIL", user or "noreply@example.com")
        if not host or not user or not pwd:
            return False
        msg = EmailMessage()
        msg["Subject"] = "Your Verification Code"
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(f"Your verification code is: {code}\nIt expires in 10 minutes.")
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        return True
    except Exception:
        return False


def start_verification(user_id: int, method: str, email: str | None, phone: str | None) -> dict:
    """Create a verification code and attempt delivery.
    method: 'email' or 'sms' (sms sending not implemented; will fallback).
    Returns dict with keys: code_visible (bool), code (str) when fallback is used.
    """
    ensure_user_columns()
    code = _generate_code(6)
    expires = datetime.utcnow() + timedelta(minutes=10)
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET verification_code=%s, verification_expires=%s WHERE id=%s",
            (code, expires, user_id),
        )
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
    delivered = False
    if method == 'email' and email:
        delivered = _send_verification_email(email, code)
    elif method == 'sms' and phone:
        # Placeholder: Integrate an SMS provider (e.g., Twilio). Fallback to visible code.
        delivered = False
    return {"code_visible": not delivered, "code": code}


def verify_code(user_id: int, code: str) -> bool:
    ensure_user_columns()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verification_code, verification_expires FROM users WHERE id=%s",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return False
        vcode, vexp = row
        if not vcode or not vexp:
            return False
        if code.strip() != vcode.strip():
            return False
        if datetime.utcnow() > vexp:
            return False
        cursor.execute(
            "UPDATE users SET is_verified=1, verification_code=NULL, verification_expires=NULL WHERE id=%s",
            (user_id,),
        )
        conn.commit()
        return True
    except mysql.connector.Error:
        return False
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
