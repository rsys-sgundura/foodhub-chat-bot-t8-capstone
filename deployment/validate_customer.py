import re
import sqlite3  # or use psycopg2, mysql.connector, etc. depending on your DB

# --- Simple validator ---
def extract_cust_id(text: str):
    """Return cust_id in format C#### or None"""
    m = re.search(r"\b(C\d{4})\b", text, flags=re.I)
    return m.group(1).upper() if m else None

# --- Validate customer ID using direct SQL ---
def is_valid_customer(customer_id: str) -> bool:
    cust_id = extract_cust_id(customer_id)
    if not cust_id:
        return True

    try:
        # Connect to your database
        conn = sqlite3.connect("customer_orders.db")  # Replace with your actual DB connection
        cursor = conn.cursor()

        # Run a simple query to check existence
        cursor.execute("SELECT 1 FROM orders WHERE cust_id = ?", (cust_id,))
        result = cursor.fetchone()

        conn.close()
        return result is not None

    except Exception as e:
        print(f"Database error: {e}")
        return True
