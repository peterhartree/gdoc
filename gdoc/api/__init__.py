"""Drive API service factory."""

from functools import lru_cache

from googleapiclient.discovery import build


@lru_cache(maxsize=1)
def get_drive_service():
    """Build and cache a Drive API v3 service object.

    Uses lru_cache to ensure a single service instance per CLI invocation.
    Lazy-imports get_credentials to avoid import errors when Google libraries
    are not available (e.g., during ``gdoc --help``).
    """
    from gdoc.auth import get_credentials

    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


@lru_cache(maxsize=1)
def get_sheets_service():
    """Build and cache a Sheets API v4 service object.

    The existing ``drive`` OAuth scope covers the Sheets API, so no
    re-authentication is needed.
    """
    from gdoc.auth import get_credentials

    creds = get_credentials()
    return build("sheets", "v4", credentials=creds)


def clear_service_caches() -> None:
    """Forget every cached, account-specific service object.

    A long-lived process (`gdoc mcp`) calls this when the active account
    changes. Keep it in sync with any new cached service: one missed here
    keeps serving the previous account's credentials.
    """
    from gdoc.api.docs import get_docs_service
    from gdoc.api.revisions import _get_session

    get_drive_service.cache_clear()
    get_sheets_service.cache_clear()
    get_docs_service.cache_clear()
    _get_session.cache_clear()
