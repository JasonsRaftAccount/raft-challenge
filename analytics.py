# analytics.py
"""
Analytics module for order data.

Provides summary statistics and logistic regression model
to predict order returns based on rating, total, and item count.
"""

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from schemas import Order

logger = logging.getLogger(__name__)


def orders_to_dataframe(orders: list[Order]) -> pd.DataFrame:
    """
    Convert list of Order objects to DataFrame with ML features.

    Args:
        orders: List of validated Order objects.

    Returns:
        DataFrame with columns: avg_rating, order_total, item_count, returned.
    """
    rows = []
    for order in orders:
        avg_rating = sum(item.rating for item in order.items) / len(order.items)
        rows.append(
            {
                "order_id": order.orderId,
                "avg_rating": round(avg_rating, 2),
                "order_total": order.total,
                "item_count": len(order.items),
                "returned": order.returned,
            }
        )

    df = pd.DataFrame(rows)
    logger.debug("Created DataFrame with %d orders", len(df))
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics from order DataFrame.

    Args:
        df: DataFrame from orders_to_dataframe().

    Returns:
        Dict with total_orders, total_revenue, avg_order_value,
        return_rate, avg_rating, avg_items_per_order.
    """
    stats = {
        "total_orders": len(df),
        "total_revenue": round(df["order_total"].sum(), 2),
        "avg_order_value": round(df["order_total"].mean(), 2),
        "return_rate": round(df["returned"].mean() * 100, 1),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "avg_items_per_order": round(df["item_count"].mean(), 2),
    }
    logger.info(
        "Summary: %d orders, $%.2f revenue, %.1f%% return rate",
        stats["total_orders"],
        stats["total_revenue"],
        stats["return_rate"],
    )
    return stats


def train_return_model(df: pd.DataFrame) -> dict:
    """
    Train logistic regression to predict order returns.

    Args:
        df: DataFrame from orders_to_dataframe().

    Returns:
        Dict with model, accuracy, confusion_matrix, feature_importance,
        train_size, and test_size.
    """
    X = df[["avg_rating", "order_total", "item_count"]]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])

    importance = dict(zip(X.columns, model.coef_[0]))

    logger.info(
        "Model trained: accuracy=%.3f, train_size=%d, test_size=%d",
        accuracy,
        len(X_train),
        len(X_test),
    )

    return {
        "model": model,
        "accuracy": round(accuracy, 3),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_return(
    model: LogisticRegression,
    avg_rating: float,
    order_total: float,
    item_count: int,
) -> dict:
    """
    Predict return probability for a single order.

    Args:
        model: Trained LogisticRegression model.
        avg_rating: Average item rating (1.0-5.0).
        order_total: Order total in dollars.
        item_count: Number of items in order.

    Returns:
        Dict with will_return (bool) and return_probability (float).
    """
    proba = model.predict_proba([[avg_rating, order_total, item_count]])[0]
    result = {
        "will_return": bool(proba[1] > 0.5),
        "return_probability": round(proba[1], 3),
    }
    logger.debug(
        "Prediction: rating=%.1f, total=%.2f, items=%d -> %.1f%% return probability",
        avg_rating,
        order_total,
        item_count,
        result["return_probability"] * 100,
    )
    return result
