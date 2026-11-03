
import re
from database import initialize_database, create_user

# Create database and users table
initialize_database()

print("\n================================")
print("       HOPE TECH ADMIN")
print("================================\n")

# Get Admin details
full_name = input("Enter Admin Full Name: ").strip()
age_input = input("Enter Admin Age: ").strip()
email = input("Enter Admin Email: ").strip()
password = input("Create Admin Password: ")

# Validate all fields
if not full_name or not age_input or not email or not password:
    print("\nError: All fields are required.")

elif not re.fullmatch(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    email
):
    print("\nError: Invalid email address. Please try again.")

elif not age_input.isdigit():
    print("\nError: Age must be a number.")

elif int(age_input) < 18 or int(age_input) > 100:
    print("\nError: Age must be between 18 and 100.")

elif len(password) < 8:
    print("\nError: Password must contain at least 8 characters.")

else:
    try:
        create_user(
            full_name=full_name,
            age=int(age_input),
            email=email,
            password=password,
            role="admin"
        )

        print("\n================================")
        print(" Admin account created successfully!")
        print(" Welcome to HOPE TECH!")
        print("================================\n")

    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            print("\nError: This email is already registered.")
        else:
            print("\nError:", error)
