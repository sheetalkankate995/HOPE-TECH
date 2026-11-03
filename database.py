
import sqlite3
import os
import re
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "users.db")


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
                CHECK(role IN ('admin', 'faculty', 'student')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    columns = [
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(users)"
        ).fetchall()
    ]

    if "username" not in columns:
        connection.execute(
            "ALTER TABLE users ADD COLUMN username TEXT"
        )

    if "mobile_number" not in columns:
        connection.execute(
            "ALTER TABLE users ADD COLUMN mobile_number TEXT"
        )

    connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS unique_username
        ON users(username)
        WHERE username IS NOT NULL
    """)

    connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS unique_mobile
        ON users(mobile_number)
        WHERE mobile_number IS NOT NULL
    """)

    connection.commit()
    connection.close()


def create_user(full_name, age, email, password, role):

    allowed_roles = ("admin", "faculty", "student")

    if role not in allowed_roles:
        raise ValueError("Invalid user role")

    connection = get_connection()

    try:
        password_hash = generate_password_hash(password)

        connection.execute("""
            INSERT INTO users
            (full_name, age, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?)
        """, (
            full_name,
            age,
            email.lower().strip(),
            password_hash,
            role
        ))

        connection.commit()
        return True

    finally:
        connection.close()


def register_user(username, mobile_number, full_name,
                  age, email, password, role):

    username = username.strip()
    mobile_number = mobile_number.strip()
    full_name = full_name.strip()
    email = email.strip().lower()
    role = role.strip().lower()

    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return False, "Username must be 3-30 letters, numbers or underscores."

    if not re.fullmatch(r"[0-9]{10}", mobile_number):
        return False, "Enter a valid 10-digit mobile number."

    if not full_name:
        return False, "Full name is required."

    if not 1 <= age <= 120:
        return False, "Enter a valid age."

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if role not in ("student", "faculty"):
        return False, "Please select Student or Faculty."

    connection = get_connection()

    try:
        password_hash = generate_password_hash(password)

        connection.execute("""
            INSERT INTO users
            (username, mobile_number, full_name, age,
             email, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            mobile_number,
            full_name,
            age,
            email,
            password_hash,
            role
        ))

        connection.commit()
        return True, "Registration successful!"

    except sqlite3.IntegrityError:
        return False, "Username, mobile number or email already registered."

    finally:
        connection.close()


def verify_user(email, password, role):

    connection = get_connection()

    try:
        user = connection.execute("""
            SELECT * FROM users
            WHERE email = ? AND role = ?
        """, (email.lower().strip(), role)).fetchone()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            return dict(user)

        return None

    finally:
        connection.close()


def get_all_users():

    connection = get_connection()

    try:
        users = connection.execute("""
            SELECT id, full_name, age, email, role, created_at
            FROM users
            ORDER BY id DESC
        """).fetchall()

        return [dict(user) for user in users]

    finally:
        connection.close()
