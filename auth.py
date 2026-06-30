import hashlib
import streamlit as st
from database import get_connection


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hash_password(password))
        )
        conn.commit()
        return True, "Account created successfully."
    except Exception:
        return False, "Username or email already exists."
    finally:
        conn.close()


def login_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, hash_password(password))
    )

    user = cursor.fetchone()
    conn.close()

    if user:
        st.session_state["logged_in"] = True
        st.session_state["user_id"] = user["id"]
        st.session_state["username"] = user["username"]
        return True

    return False


def logout_user():
    st.session_state.clear()


def require_login():
    return st.session_state.get("logged_in", False)