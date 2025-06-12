import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Product Sales & Clustering Dashboard", layout="wide")
st.title("📊 Product Sales & Clustering Dashboard")

uploaded_file = st.file_uploader("📤 Upload your product sales CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Convert data types
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["Price_per_Unit"] = df["Value"] / df["Quantity"]
        df["Month"] = df["Date"].dt.strftime('%B')
        df["Year"] = df["Date"].dt.year

        # Drop rows with missing Quantity or Value for filters
        valid_data = df.dropna(subset=["Quantity", "Value"])

        # ✅ Safe min/max range setup
        if not valid_data.empty:
            quantity_min, quantity_max = int(valid_data["Quantity"].min()), int(valid_data["Quantity"].max())
            value_min, value_max = int(valid_data["Value"].min()), int(valid_data["Value"].max())
        else:
            quantity_min, quantity_max = 0, 100
            value_min, value_max = 0, 100000

        # Sidebar filters
        with st.sidebar:
            st.header("🔍 Filters")
            product_filter = st.multiselect("Select Product(s)", options=sorted(df["Item Name"].dropna().unique()))
            month_filter = st.multiselect("Select Month(s)", options=sorted(df["Month"].dropna().unique()))
            year_filter = st.multiselect("Select Year(s)", options=sorted(df["Year"].dropna().unique()))

            quantity_range = st.slider("Quantity Range", min_value=quantity_min, max_value=quantity_max, value=(quantity_min, quantity_max))
            value_range = st.slider("Value Range", min_value=value_min, max_value=value_max, value=(value_min, value_max))

        # Apply filters
        filtered_df = df.copy()
        if product_filter:
            filtered_df = filtered_df[filtered_df["Item Name"].isin(product_filter)]
        if month_filter:
            filtered_df = filtered_df[filtered_df["Month"].isin(month_filter)]
        if year_filter:
            filtered_df = filtered_df[filtered_df["Year"].isin(year_filter)]

        filtered_df = filtered_df[
            (filtered_df["Quantity"] >= quantity_range[0]) & (filtered_df["Quantity"] <= quantity_range[1]) &
            (filtered_df["Value"] >= value_range[0]) & (filtered_df["Value"] <= value_range[1])
        ]

        # Total Sales Summary
        total_quantity = filtered_df["Quantity"].sum()
        total_value = filtered_df["Value"].sum()
        st.markdown(f"### 📦 Total Quantity Sold: `{total_quantity}`")
        st.markdown(f"### 💵 Total Sales Value: ₹ `{total_value:,.2f}`")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Top Products", "Clustering", "Raw Data"])

        with tab1:
            st.subheader("🔝 Top 10 Products by Quantity")
            top_q = filtered_df.sort_values(by="Quantity", ascending=False).head(10)
            st.dataframe(top_q)
            fig1, ax1 = plt.subplots()
            sns.barplot(data=top_q, y="Item Name", x="Quantity", ax=ax1)
            ax1.set_title("Top 10 by Quantity")
            st.pyplot(fig1)

            st.subheader("💰 Top 10 Products by Value")
            top_v = filtered_df.sort_values(by="Value", ascending=False).head(10)
            st.dataframe(top_v)
            fig2, ax2 = plt.subplots()
            sns.barplot(data=top_v, y="Item Name", x="Value", ax=ax2)
            ax2.set_title("Top 10 by Value")
            st.pyplot(fig2)

        with tab2:
            st.subheader("🧠 Product Segmentation using K-Means")
            cluster_df = filtered_df.dropna(subset=["Quantity", "Value"])
            if len(cluster_df) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                cluster_df["Cluster"] = kmeans.fit_predict(cluster_df[["Quantity", "Value"]])
                fig3, ax3 = plt.subplots()
                sns.scatterplot(data=cluster_df, x="Quantity", y="Value", hue="Cluster", palette="tab10", s=100)
                for i in range(cluster_df.shape[0]):
                    ax3.text(cluster_df["Quantity"].iloc[i], cluster_df["Value"].iloc[i], cluster_df["Item Name"].iloc[i], fontsize=8)
                ax3.set_title("Clusters based on Quantity and Value")
                st.pyplot(fig3)
            else:
                st.warning("Not enough data for clustering (need at least 3 valid rows).")

        with tab3:
            st.subheader("📋 Filtered Dataset")
            st.dataframe(filtered_df)
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Data", csv, "filtered_product_data.csv", "text/csv")

    except Exception as e:
        st.error(f"🚨 An error occurred while processing your file:\n\n{e}")
else:
    st.info("📤 Please upload your product sales CSV to begin.")
