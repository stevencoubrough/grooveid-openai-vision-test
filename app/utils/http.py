import httpx

DEFAULT_TIMEOUT = httpx.Timeout(12.0, connect=4.0)
DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=20)

def get_client() -> httpx.AsyncClient:
    """
    Create a reusable HTTP client with sensible defaults.
    """
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS, headers={
        "User-Agent": "GrooveID/1.0 (+https://grooveid.app)"
    })
