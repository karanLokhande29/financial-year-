import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Page settings
st.set_page_config(page_title="Sales & Clustering Dashboard", layout="wide")
st.title("📊 Sales & Clustering Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your sales CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read and clean
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["Price_per_Unit"] = df["Value"] / df["Quantity"]
        df["Month"] = df["Date"].dt.strftime("%B")
        df["Year"] = df["Date"].dt.year

        df = df.dropna(subset=["Item Name", "Quantity", "Value", "Date"])

        # Debug info
        st.markdown("### 🔍 Preview")
        st.dataframe(df.head())
        st.markdown(f"**Date Range:** {df['Date'].min().date()} → {df['Date'].max().date()}")

        # Filter setup
        with st.sidebar:
            st.header("🔍 Filters")
            products = st.multiselect("Product", df["Item Name"].unique(), default=list(df["Item Name"].unique()))
            months = st.multiselect("Month", df["Month"].unique(), default=list(df["Month"].unique()))
            years = st.multiselect("Year", sorted(df["Year"].unique()), default=list(df["Year"].unique()))
            quantity_range = st.slider("Quantity Range", int(df["Quantity"].min()), int(df["Quantity"].max()),
                                       (int(df["Quantity"].min()), int(df["Quantity"].max())))
            value_range = st.slider("Value Range", int(df["Value"].min()), int(df["Value"].max()),
                                    (int(df["Value"].min()), int(df["Value"].max())))

        # Apply filters
        filtered = df[
            df["Item Name"].isin(products) &
            df["Month"].isin(months) &
            df["Year"].isin(years) &
            df["Quantity"].between(quantity_range[0], quantity_range[1]) &
            df["Value"].between(value_range[0], value_range[1])
        ]

        st.markdown(f"### ✅ Filtered Rows: `{filtered.shape[0]}`")
        if filtered.empty:
            st.warning("⚠️ No records match the current filters.")
            st.stop()

        # Total summary
        st.markdown(f"### 📦 Total Quantity: `{int(filtered['Quantity'].sum())}`")
        st.markdown(f"### 💰 Total Sales: ₹ `{filtered['Value'].sum():,.2f}`")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Top Products", "Clustering", "Filtered Data"])

        with tab1:
            st.subheader("🔝 Top 10 by Quantity")
            top_q = filtered.sort_values(by="Quantity", ascending=False).head(10)
            st.dataframe(top_q)
            fig1, ax1 = plt.subplots()
            sns.barplot(data=top_q, y="Item Name", x="Quantity", ax=ax1)
            st.pyplot(fig1)

            st.subheader("💰 Top 10 by Value")
            top_v = filtered.sort_values(by="Value", ascending=False).head(10)
            st.dataframe(top_v)
            fig2, ax2 = plt.subplots()
            sns.barplot(data=top_v, y="Item Name", x="Value", ax=ax2)
            st.pyplot(fig2)

        with tab2:
            st.subheader("🧠 Product Segmentation (K-Means Clustering)")
            if filtered.shape[0] >= 3:
                model = KMeans(n_clusters=3, random_state=42)
                filtered["Cluster"] = model.fit_predict(filtered[["Quantity", "Value"]])
                fig3, ax3 = plt.subplots()
                sns.scatterplot(data=filtered, x="Quantity", y="Value", hue="Cluster", palette="tab10", s=100)
                for i in range(filtered.shape[0]):
                    ax3.text(filtered["Quantity"].iloc[i], filtered["Value"].iloc[i], filtered["Item Name"].iloc[i], fontsize=8)
                ax3.set_title("Clusters by Quantity vs Value")
                st.pyplot(fig3)
            else:
                st.info("ℹ️ Need at least 3 data points to run clustering.")

        with tab3:
            st.subheader("📋 Filtered Data")
            st.dataframe(filtered)
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "filtered_sales.csv", "text/csv")

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("📤 Please upload a CSV with columns: `Item Name`, `Quantity`, `Value`, `Date`")
