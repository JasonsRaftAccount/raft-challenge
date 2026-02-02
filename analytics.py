# file: analytics.py
"""
Analytics module for order data.
- Summary statistics
- Logistic regression to predict order returns

Installation:
    pip install pandas scikit-learn
"""

import pandas as pd
import logging
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from validation import extract_anchors, AnchorExtractionError
from schemas import Order, OrderItem, AnalyticsFeatures

logger = logging.getLogger(__name__)


# TODO: Can fallback to regex after failed LLM call on analytics.
def parse_raw_order(raw: str) -> Order | None:
    """Parse raw order string into Order object for analytics."""
    try:
        anchor = extract_anchors(raw)
        buyer = re.search(r"Buyer=([^,]+),", raw).group(1)
        city = re.search(r"Location=([^,]+),", raw).group(1)
        items_match = re.findall(r"([^,]+?) \((\d+\.?\d*)\*\)", raw)

        if not items_match:
            return None

        return Order(
            orderId=anchor.orderId,
            buyer=buyer,
            city=city,
            state=anchor.state,
            total=anchor.total,
            items=[OrderItem(name=n.strip(), rating=float(r)) for n, r in items_match],
            returned=anchor.returned,
        )
    except (AttributeError, AnchorExtractionError):
        return None


def orders_to_dataframe(orders: list[Order]) -> pd.DataFrame:
    """Convert list of Order objects to DataFrame with ML features."""
    features = [AnalyticsFeatures.from_order(o) for o in orders]
    return pd.DataFrame([f.model_dump() for f in features])


def summary_stats(df: pd.DataFrame) -> dict:
    """Calculate basic summary statistics."""
    return {
        "total_orders": len(df),
        "total_revenue": round(df["order_total"].sum(), 2),
        "avg_order_value": round(df["order_total"].mean(), 2),
        "return_rate": round(df["returned"].mean() * 100, 1),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "avg_items_per_order": round(df["item_count"].mean(), 2),
    }


def train_return_model(df: pd.DataFrame) -> dict:
    """
    Train logistic regression to predict order returns.

    Features: avg_rating, order_total, item_count
    Target: returned (True/False)

    Returns dict with model, metrics, and feature importance.
    """
    # Features and target
    X = df[["avg_rating", "order_total", "item_count"]]
    y = df["returned"]

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])

    logger.info(
        f"Trained model: accuracy={accuracy:.3f}, train_size={len(X_train)}, test_size={len(X_test)}"
    )

    # Feature importance (coefficients)
    importance = dict(zip(X.columns, model.coef_[0]))

    return {
        "model": model,
        "accuracy": round(accuracy, 3),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_return(
    model: LogisticRegression, avg_rating: float, order_total: float, item_count: int
) -> dict:
    """Predict return probability for a single order."""
    proba = model.predict_proba([[avg_rating, order_total, item_count]])[0]
    return {
        "will_return": bool(proba[1] > 0.5),
        "return_probability": round(proba[1], 3),
    }


if __name__ == "__main__":
    print("Analytics module loaded. Import and use in app.py")
