# schemas.py
"""
Pydantic models for the order parsing agent.

Defines all data structures for orders, failures, and responses used
throughout the pipeline.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    """Single item within an order."""

    name: str = Field(description="Product name")
    rating: float = Field(ge=1.0, le=5.0, description="Customer rating 1.0-5.0")


class Order(BaseModel):
    """Full parsed order with all fields."""

    orderId: str = Field(description="Unique order identifier")
    buyer: str = Field(description="Customer name")
    city: str = Field(description="City name")
    state: str = Field(min_length=2, max_length=2, description="Two-letter state code")
    total: float = Field(gt=0, description="Order total in dollars")
    items: list[OrderItem] = Field(description="List of items with ratings")
    returned: bool = Field(description="Whether order was returned")

    def to_summary(self) -> "OrderSummary":
        """
        Convert to minimal summary format for query responses.

        Returns:
            OrderSummary with orderId, buyer, state, and total only.
        """
        return OrderSummary(
            orderId=self.orderId,
            buyer=self.buyer,
            state=self.state,
            total=self.total,
        )


class OrderSummary(BaseModel):
    """Minimal order format for query responses."""

    orderId: str
    buyer: str
    state: str
    total: float


class RawOrderStore(BaseModel):
    """Storage for raw order strings from the API."""

    orders: list[str] = Field(default_factory=list, description="Raw order strings")
    fetched_at: datetime = Field(default_factory=datetime.now)

    def get_batch(self, batch_index: int, batch_size: int) -> list[str]:
        """
        Retrieve a batch of raw orders by index.

        Args:
            batch_index: Zero-based batch number.
            batch_size: Number of orders per batch.

        Returns:
            List of raw order strings for the specified batch.
        """
        start = batch_index * batch_size
        end = start + batch_size
        return self.orders[start:end]

    def total_batches(self, batch_size: int) -> int:
        """
        Calculate total number of batches.

        Args:
            batch_size: Number of orders per batch.

        Returns:
            Number of batches needed to process all orders.
        """
        return (len(self.orders) + batch_size - 1) // batch_size


class FailedBatch(BaseModel):
    """Record of a batch that failed structural validation."""

    batch_index: int = Field(description="Zero-based batch number")
    raw_orders: list[str] = Field(description="Raw orders in the failed batch")
    error: str = Field(description="Error message describing the failure")
    attempts: int = Field(default=1, description="Number of retry attempts made")
    failed_at: datetime = Field(default_factory=datetime.now)


class FailedRecord(BaseModel):
    """Record of an individual order that failed semantic validation."""

    orderId: Optional[str] = Field(
        default=None, description="Order ID for mismatch or hallucinated failures"
    )
    rawSnippet: Optional[str] = Field(
        default=None, description="First 100 chars of raw order for dropped failures"
    )
    failureType: Literal["mismatch", "hallucinated"] = Field(
        description="Category of validation failure"
    )
    reason: str = Field(description="Specific reason for failure")
    failed_at: datetime = Field(default_factory=datetime.now)


class DeadLetterQueue(BaseModel):
    """Aggregates all failures from parse and validation stages."""

    failed_batches: list[FailedBatch] = Field(default_factory=list)
    failed_records: list[FailedRecord] = Field(default_factory=list)

    @property
    def total_failures(self) -> int:
        """
        Count total failed orders across batches and individual records.

        Returns:
            Total number of orders that failed processing.
        """
        batch_orders = sum(len(b.raw_orders) for b in self.failed_batches)
        return batch_orders + len(self.failed_records)

    def add_batch_failure(
        self,
        batch_index: int,
        raw_orders: list[str],
        error: str,
        attempts: int = 1,
    ) -> None:
        """
        Record a batch that failed structural validation.

        Args:
            batch_index: Zero-based batch number.
            raw_orders: Raw order strings in the failed batch.
            error: Error message describing the failure.
            attempts: Number of retry attempts made.
        """
        self.failed_batches.append(
            FailedBatch(
                batch_index=batch_index,
                raw_orders=raw_orders,
                error=error,
                attempts=attempts,
            )
        )

    def add_record_failure(
        self,
        failure_type: Literal["mismatch", "dropped", "hallucinated"],
        reason: str,
        order_id: Optional[str] = None,
        raw_snippet: Optional[str] = None,
    ) -> None:
        """
        Record an individual order that failed semantic validation.

        Args:
            failure_type: Category of failure (mismatch, dropped, hallucinated).
            reason: Specific reason for failure.
            order_id: Order ID for mismatch/hallucinated failures.
            raw_snippet: Raw order snippet for dropped failures.
        """
        self.failed_records.append(
            FailedRecord(
                orderId=order_id,
                rawSnippet=raw_snippet[:100] if raw_snippet else None,
                failureType=failure_type,
                reason=reason,
            )
        )


class QueryMeta(BaseModel):
    """Metadata about query execution."""

    total_raw: int = Field(description="Orders fetched from API")
    total_parsed: int = Field(description="Orders successfully parsed")
    total_valid: int = Field(description="Orders that passed validation")
    total_failed: int = Field(description="Orders in dead letter queue")

    @property
    def success_rate(self) -> float:
        """
        Calculate percentage of successfully processed orders.

        Returns:
            Success rate as a decimal (0.0 to 1.0).
        """
        if self.total_raw == 0:
            return 1.0
        return self.total_valid / self.total_raw


class QueryResponse(BaseModel):
    """Response format for CLI and Query tab."""

    orders: list[OrderSummary] = Field(description="Validated order summaries")
    meta: QueryMeta = Field(description="Execution statistics")


class AnalyticsData(BaseModel):
    """Response format for Analytics tab with full order details."""

    orders: list[Order] = Field(description="Full validated orders for analysis")
    meta: QueryMeta = Field(description="Execution statistics")


class AgentResult(BaseModel):
    """Complete result from agent execution."""

    raw_store: RawOrderStore = Field(description="Raw orders from API")
    valid_orders: list[Order] = Field(description="Orders that passed all validation")
    dlq: DeadLetterQueue = Field(description="All processing failures")
    meta: QueryMeta = Field(description="Execution statistics")

    def to_query_response(self) -> QueryResponse:
        """
        Format result for CLI and Query tab.

        Returns:
            QueryResponse with order summaries and metadata.
        """
        return QueryResponse(
            orders=[o.to_summary() for o in self.valid_orders],
            meta=self.meta,
        )

    def to_analytics_data(self) -> AnalyticsData:
        """
        Format result for Analytics tab.

        Returns:
            AnalyticsData with full orders and metadata.
        """
        return AnalyticsData(
            orders=self.valid_orders,
            meta=self.meta,
        )
