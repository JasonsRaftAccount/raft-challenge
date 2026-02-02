# file: api_client.py
"""HTTP client for the dummy customer API."""
import requests
import config


class APIError(Exception):
    """Raised when API call fails."""

    pass


def fetch_orders() -> list[str]:
    """Fetch all orders. Returns list of raw order strings."""
    try:
        resp = requests.get(f"{config.DUMMY_API_URL}/api/orders", timeout=10)
        resp.raise_for_status()
        return resp.json().get("raw_orders", [])
    except requests.RequestException as e:
        raise APIError(f"Failed to fetch orders: {e}")


def fetch_order(order_id: str) -> str | None:
    """Fetch single order by ID. Returns raw order string or None if not found."""
    try:
        resp = requests.get(f"{config.DUMMY_API_URL}/api/order/{order_id}", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("raw_order")
    except requests.RequestException as e:
        raise APIError(f"Failed to fetch order {order_id}: {e}")


if __name__ == "__main__":
    # Quick test
    orders = fetch_orders()
    print(f"Fetched {len(orders)} orders")
    print(f"Sample: {orders[0][:80]}...")
