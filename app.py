# app.py
"""
Streamlit UI for the order parsing agent.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from agent import run_agent
from analytics import (
    orders_to_dataframe,
    predict_return,
    summary_stats,
    train_return_model,
)
from clients import APIError, fetch_order_async
from schemas import Order
from utils import setup_logging

import asyncio

setup_logging()

st.set_page_config(page_title="Order Parsing Agent", layout="wide")

header_left, header_right = st.columns([1, 20])
with header_left:
    st.image("assets/derp-face-open-mouth.png", width=80)
with header_right:
    st.title("Order Parsing Agent")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Query Orders", "Single Order", "Analytics", "Dead Letter Queue"]
)


with tab1:
    st.subheader("Natural Language Query")

    query = st.text_input(
        "Enter your query:",
        placeholder="Show me all orders from Ohio over $500",
    )

    if st.button("Run Query", type="primary"):
        if not query:
            st.warning("Please enter a query")
        else:
            with st.spinner("Processing..."):
                result = run_agent(query)
                st.session_state["last_result"] = result

            if result.meta.total_valid == 0:
                st.info("No orders matched your query")
            else:
                st.success(
                    f"Found {result.meta.total_valid} orders "
                    f"({result.meta.success_rate:.1%} success rate)"
                )

                response = result.to_query_response()
                st.dataframe(
                    pd.DataFrame([o.model_dump() for o in response.orders]),
                    width="stretch",
                )

                with st.expander("View JSON"):
                    st.json(response.model_dump())


with tab2:
    st.subheader("Single Order Lookup")

    order_id = st.text_input("Order ID:", placeholder="1001")

    if st.button("Fetch Order"):
        if not order_id:
            st.warning("Please enter an order ID")
        else:
            try:
                raw_order = asyncio.run(fetch_order_async(order_id))
                if raw_order is None:
                    st.error(f"Order {order_id} not found")
                else:
                    st.code(raw_order)
            except APIError as e:
                st.error(str(e))


with tab3:
    st.subheader("Analytics & Return Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Summary Statistics**")

        if st.button("Load Data & Train Model"):
            if "orders_cache" in st.session_state:
                orders = st.session_state["orders_cache"]
                st.info("Using cached orders")
            else:
                with st.spinner("Fetching and parsing orders via LLM..."):
                    result = run_agent("Show me all orders")

                    if result.meta.total_valid == 0:
                        st.error("No orders parsed")
                        orders = []
                    else:
                        orders = result.valid_orders
                        st.session_state["orders_cache"] = orders
                        st.session_state["last_result"] = result

            if orders:
                df = orders_to_dataframe(orders)
                stats = summary_stats(df)
                results = train_return_model(df)

                st.session_state["stats"] = stats
                st.session_state["model"] = results["model"]
                st.session_state["model_results"] = results

                st.success(f"Loaded {len(orders)} orders")

        if "stats" in st.session_state:
            stats = st.session_state["stats"]
            st.metric("Total Orders", stats["total_orders"])
            st.metric("Total Revenue", f"${stats['total_revenue']:,.2f}")
            st.metric("Avg Order Value", f"${stats['avg_order_value']:.2f}")
            st.metric("Return Rate", f"{stats['return_rate']}%")

    with col2:
        st.markdown("**Return Prediction Model**")

        if "model_results" in st.session_state:
            results = st.session_state["model_results"]
            st.metric("Model Accuracy", f"{results['accuracy']:.1%}")

            st.markdown("Feature Importance:")
            for feat, imp in results["feature_importance"].items():
                direction = "↑ returns" if imp > 0 else "↓ returns"
                st.text(f"  {feat}: {imp:.4f} ({direction})")

            st.markdown("---")
            st.markdown("**Predict Return Probability**")

            avg_rating = st.slider("Avg Rating", 1.0, 5.0, 3.0, 0.1)
            order_total = st.number_input("Order Total ($)", 10.0, 5000.0, 100.0)
            item_count = st.number_input("Item Count", 1, 10, 2)

            if st.button("Predict"):
                pred = predict_return(
                    st.session_state["model"],
                    avg_rating,
                    order_total,
                    item_count,
                )
                prob = pred["return_probability"]
                if prob > 0.5:
                    st.error(f"High return risk: {prob:.0%}")
                else:
                    st.success(f"Low return risk: {prob:.0%}")
        else:
            st.info("Click 'Load Data & Train Model' first")


with tab4:
    st.subheader("Dead Letter Queue")

    if "last_result" not in st.session_state:
        st.info("Run a query first to see failures")
    else:
        result = st.session_state["last_result"]
        dlq = result.dlq

        st.metric("Total Failures", dlq.total_failures)

        if dlq.failed_batches:
            st.markdown("**Failed Batches (Parse Stage)**")
            for batch in dlq.failed_batches:
                with st.expander(
                    f"Batch {batch.batch_index} - {len(batch.raw_orders)} orders"
                ):
                    st.text(f"Error: {batch.error}")
                    st.text(f"Attempts: {batch.attempts}")
                    st.text(f"Failed at: {batch.failed_at}")
                    st.markdown("**Raw orders:**")
                    for order in batch.raw_orders:
                        st.code(order)

        if dlq.failed_records:
            st.markdown("**Failed Records (Validation Stage)**")
            records_data = []
            for record in dlq.failed_records:
                records_data.append(
                    {
                        "Order ID": record.orderId or "N/A",
                        "Type": record.failureType,
                        "Reason": record.reason,
                        "Snippet": (
                            record.rawSnippet[:50] + "..."
                            if record.rawSnippet
                            else "N/A"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(records_data), width="stretch")

        if dlq.total_failures == 0:
            st.success("No failures recorded")
