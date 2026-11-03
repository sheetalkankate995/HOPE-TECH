
from database import initialize_database, create_user

initialize_database()

print("HOPE TECH - Create Admin Account")

full_name = input("Enter Admin Name: ").strip()
age = int(input("Enter Admin Age: "))
email = input("Enter Admin Email: ").strip()
password = input("Enter Admin Password: ")

try:
    create_user(
        full_name,
        age,
        email,
        password,
        "admin"
    )

    print("\nAdmin account created successfully!")

except Exception as e:
    print("\nError creating Admin:", e)
