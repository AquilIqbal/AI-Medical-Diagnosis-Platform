"""Interactively create one active administrator account in MySQL.

Run this script manually from the project root only after the database schema
has been migrated. It does not create tables or run migration commands.
"""

from __future__ import annotations

import sys
from getpass import getpass
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError


# Allow ``python scripts/create_admin.py`` to import the project package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from backend.extensions import db  # noqa: E402
from backend.models.user import User, UserRole  # noqa: E402
from backend.utils.password import generate_password_hash  # noqa: E402


def prompt_for_admin_details() -> tuple[str, str, str, str]:
    """Prompt for and normalize the administrator's required details."""
    full_name = input("Full Name: ").strip()
    username = input("Username: ").strip()
    email = input("Email: ").strip().lower()
    password = getpass("Password: ")
    return full_name, username, email, password


def main() -> int:
    """Create an administrator account after validating uniqueness."""
    full_name, username, email, password = prompt_for_admin_details()

    if not all((full_name, username, email, password)):
        print("Error: Full Name, Username, Email, and Password are required.")
        return 1

    with app.app_context():
        try:
            existing_users = db.session.scalars(
                select(User).where(
                    or_(User.username == username, User.email == email)
                )
            ).all()

            if any(user.username == username for user in existing_users):
                print("Error: That username is already in use.")
                return 1

            if any(user.email == email for user in existing_users):
                print("Error: That email address is already in use.")
                return 1

            admin_user = User(
                full_name=full_name,
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.session.add(admin_user)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            print("Error: Username or email already exists. No user was created.")
            return 1
        except SQLAlchemyError as error:
            db.session.rollback()
            print(f"Database error: {error}")
            return 1

    print(f"Success: Administrator '{username}' was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

