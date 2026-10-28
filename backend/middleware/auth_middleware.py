"""Reusable authentication and authorization route middleware."""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from flask import flash, redirect, session, url_for
from flask.typing import ResponseReturnValue


P = ParamSpec("P")
R = TypeVar("R")


def login_required(
    view_function: Callable[P, ResponseReturnValue],
) -> Callable[P, ResponseReturnValue]:
    """Require an authenticated session before executing a view function.

    Future protected routes can use this decorator without duplicating session
    checks or redirect behavior.
    """

    @wraps(view_function)
    def wrapped_view(*args: P.args, **kwargs: P.kwargs) -> ResponseReturnValue:
        """Redirect unauthenticated requests to the login page."""
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("auth.login"))

        return view_function(*args, **kwargs)

    return wrapped_view


def admin_required(
    view_function: Callable[P, R],
) -> Callable[P, ResponseReturnValue]:
    """Block access until administrator authorization is implemented.

    The decorator deliberately returns HTTP 501 rather than allowing a route
    to execute without role-based access protection.
    """

    @wraps(view_function)
    def wrapped_view(*args: P.args, **kwargs: P.kwargs) -> ResponseReturnValue:
        """Return the explicit not-implemented authorization response."""
        _ = args, kwargs
        return "Administrator authorization is not implemented.", 501

    return wrapped_view
