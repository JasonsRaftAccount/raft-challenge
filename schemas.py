# file: schemas.py
"""
Pydantic schemas for order data validation (structural validation).

These schemas serve two purposes:
1. Constrain LLM output to predictable structure (prevents hallucination)
2. Provide type validation for parsed order data

For factual validation (verifying against source data), see validation.py
"""

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def safe_parse_order(data: dict) -> tuple["Order | None", "str | None"]:
    """
    Safely parse order dict into Order model.

    Args:
        data: Dict from LLM output

    Returns:
        Tuple of (Order, None) on success or (None, error_message) on failure
    """
    try:
        return Order.model_validate(data), None
    except ValidationError as e:
        error_msg = f"Pydantic validation failed: {e.error_count()} errors"
        logger.warning(f"Order parse failed: {error_msg}")
        for error in e.errors():
            logger.debug(f"  - {error['loc']}: {error['msg']}")
        return None, error_msg


def safe_parse_response(data: dict) -> "QueryResponse":
    """
    Safely parse LLM response, returning error response on failure.

    Args:
        data: Dict from LLM output, expected to have 'orders' key

    Returns:
        QueryResponse (with orders or error message)
    """
    try:
        return QueryResponse.model_validate(data)
    except ValidationError as e:
        error_msg = f"Failed to parse LLM response: {e.error_count()} validation errors"
        logger.error(error_msg)
        return QueryResponse(orders=[], error=error_msg)


class OrderItem(BaseModel):
    """Individual item within an order."""

    name: str = Field(..., description="Product name")
    rating: float = Field(..., ge=1.0, le=5.0, description="Item rating 1-5 stars")


class Order(BaseModel):
    """
    Structured order matching challenge output requirements.

    Example output:
        { "orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.10 }
    """

    orderId: str = Field(..., description="Order ID (numeric string)")
    buyer: str = Field(..., description="Buyer full name")
    city: str = Field(..., description="City name")
    state: str = Field(
        ..., min_length=2, max_length=2, description="2-letter state code"
    )
    total: float = Field(..., gt=0, description="Order total in dollars")
    items: list[OrderItem] = Field(
        ..., min_length=1, description="List of items with ratings"
    )
    returned: bool = Field(..., description="Whether order was returned")

    @field_validator("state")
    @classmethod
    def state_must_be_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("orderId")
    @classmethod
    def order_id_must_be_numeric(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("orderId must contain only digits")
        return v

    @property
    def avg_rating(self) -> float:
        """Calculate average rating across all items."""
        if not self.items:
            return 0.0
        return sum(item.rating for item in self.items) / len(self.items)

    @property
    def item_count(self) -> int:
        """Number of items in order."""
        return len(self.items)

    def to_challenge_format(self) -> dict:
        """
        Return order in the exact format specified by challenge requirements.

        Returns:
            { "orderId": "...", "buyer": "...", "state": "OH", "total": 742.10 }
        """
        return {
            "orderId": self.orderId,
            "buyer": self.buyer,
            "state": self.state,
            "total": self.total,
        }


class QueryResponse(BaseModel):
    """
    Response structure for agent queries.

    Example:
        {
            "orders": [
                { "orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.10 }
            ],
            "error": null
        }
    """

    orders: list[Order] = Field(
        default_factory=list, description="List of matching orders"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if query failed"
    )

    def to_challenge_format(self) -> dict:
        """Return response in challenge-specified format."""
        return {"orders": [order.to_challenge_format() for order in self.orders]}


class AnalyticsFeatures(BaseModel):
    """
    Features extracted from orders for classification model.
    Used to predict return probability.
    """

    orderId: str
    avg_rating: float = Field(..., ge=1.0, le=5.0)
    order_total: float = Field(..., gt=0)
    item_count: int = Field(..., ge=1)
    returned: bool

    @classmethod
    def from_order(cls, order: Order) -> "AnalyticsFeatures":
        """Convert Order to analytics features."""
        return cls(
            orderId=order.orderId,
            avg_rating=order.avg_rating,
            order_total=order.total,
            item_count=order.item_count,
            returned=order.returned,
        )
