# file: app.py
"""
Streamlit UI for the order parsing agent.
Run streamlit run app.py
"""

import streamlit as st
import pandas as pd

from config import setup_logging
from agent import run_agent
from api_client import fetch_order, APIError
from schemas import Order
from analytics import (
    orders_to_dataframe,
    summary_stats,
    train_return_model,
    predict_return,
)

# logs

setup_logging()

# Headers

st.set_page_config(page_title="OMG Tech Challenge", layout="wide")
header_left, header_right = st.columns([1, 20])
with header_left:
    st.image("assets/derp-face-open-mouth.png", width=80)
with header_right:
    st.title("Order Parsing Agent")

tab1, tab2, tab3 = st.tabs(["Query Orders", "Single Order", "Analytics"])

# TAB 1: Natural Language Query

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

            if result.get("error"):
                st.error(result["error"])
            elif not result["orders"]:
                st.info("No orders matched your query")
            else:
                st.success(f"Found {len(result['orders'])} orders")
                st.dataframe(pd.DataFrame(result["orders"]), width="stretch")

                with st.expander("View JSON"):
                    st.json(result)

# TAB 2: Single Order Lookup

with tab2:
    st.subheader("Single Order Lookup")

    order_id = st.text_input("Order ID:", placeholder="1001")

    if st.button("Fetch Order"):
        if not order_id:
            st.warning("Please enter an order ID")
        else:
            try:
                raw_order = fetch_order(order_id)
                if raw_order is None:
                    st.error(f"Order {order_id} not found")
                else:
                    st.code(raw_order)
            except APIError as e:
                st.error(str(e))

# TAB 3: Analytics

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
                    try:
                        result = run_agent("Show me all orders", full_output=True)
                        if result.get("error"):
                            st.error(result["error"])
                            orders = []
                        else:
                            orders = [Order(**o) for o in result["orders"]]
                            st.session_state["orders_cache"] = orders
                    except APIError as e:
                        st.error(str(e))
                        orders = []

            if orders:
                df = orders_to_dataframe(orders)
                stats = summary_stats(df)
                results = train_return_model(df)

                st.session_state["stats"] = stats
                st.session_state["model"] = results["model"]
                st.session_state["model_results"] = results

                st.success(f"Loaded {len(orders)} orders")
            else:
                st.error("No orders parsed")

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
            order_total = st.number_input("Order Total ($)", 10.0, 2000.0, 100.0)
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
                    st.error(f"PANIC!!! High return risk: {prob:.0%}")
                else:
                    st.success(f"CHILL... Low return risk: {prob:.0%}")
        else:
            st.info("Click 'Load Data & Train Model' first")
