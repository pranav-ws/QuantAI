"""
create_admin.py
Creates the first admin user for QuantAI.
Run once after setup:  python create_admin.py
"""
from src.database import create_tables
from src.user_db import create_user_tables, create_user, get_user_by_username
from src.auth import hash_password
import getpass

create_tables()
create_user_tables()

print("\n" + "="*50)
print("  QuantAI — Create Admin Account")
print("="*50)
print("  This creates the first admin user.")
print("  The admin can see all users and portfolios.\n")

username = input("  Username: ").strip()
email    = input("  Email   : ").strip()
password = getpass.getpass("  Password: ")
confirm  = getpass.getpass("  Confirm : ")

if password != confirm:
    print("\n  ❌ Passwords do not match.")
    exit(1)

if len(password) < 6:
    print("\n  ❌ Password must be at least 6 characters.")
    exit(1)

result = create_user(username, email, hash_password(password))
if result["success"]:
    print(f"\n  ✅ Admin account created!")
    print(f"     Username : {username}")
    print(f"     Role     : {result['role']}")
    print(f"\n  You can now log in at the dashboard.")
else:
    print(f"\n  ❌ {result['error']}")
print("="*50 + "\n")
