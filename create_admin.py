"""
create_admin.py
Creates the first admin user for QuantAI.
Uses the same auth system as the API (src/auth.py).
Run once after setup:  python create_admin.py
"""
import getpass
from src.database import create_tables

create_tables()

print("\n" + "="*50)
print("  QuantAI — Create Admin Account")
print("="*50)
print("  This creates the first admin user.")
print("  The admin can manage users and view all data.\n")

username = input("  Username : ").strip()
email    = input("  Email    : ").strip()
password = getpass.getpass("  Password : ")
confirm  = getpass.getpass("  Confirm  : ")

if password != confirm:
    print("\n  ❌ Passwords do not match.")
    exit(1)

if len(password) < 6:
    print("\n  ❌ Password must be at least 6 characters.")
    exit(1)

if len(username) < 3:
    print("\n  ❌ Username must be at least 3 characters.")
    exit(1)

from src.auth import register_user, count_users

n_users  = count_users()
is_admin = (n_users == 0)   # first user always becomes admin

ok, result = register_user(username, email, password, is_admin=is_admin)

if ok:
    role = "Admin" if is_admin else "User"
    print(f"\n  ✅ Account created!")
    print(f"     Username : {username}")
    print(f"     Role     : {role}")
    if not is_admin:
        print(f"     Note     : An admin already exists. This account has user-level access.")
    print(f"\n  Open dashboard/login.html and sign in.")
else:
    print(f"\n  ❌ {result}")

print("="*50 + "\n")
