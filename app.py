import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Product Sales & Clustering Dashboard", layout="wide")
st.title("📊 Product Sales & Clustering Dashboard")

uploaded_file = st.file_uploader("📤 Upload your product sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Handle date column
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Month"] = df["Date"].dt.strftime('%B')
    df["Year"] = df["Date"].dt.year

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        product_filter = st.multiselect("Select Product(s)", options=df["Item Name"].dropna().unique(), default=None)
        month_filter = st.multiselect("Select Month(s)", options=df["Month"].dropna().unique(), default=None)
        year_filter = st.multiselect("Select Year(s)", options=sorted(df["Year"].dropna().unique()), default=None)

        quantity_min, quantity_max = int(df["Quantity"].min()), int(df["Quantity"].max())
        quantity_range = st.slider("Quantity Range", min_value=quantity_min, max_value=quantity_max, value=(quantity_min, quantity_max))

        value_min, value_max = int(df["Value"].min()), int(df["Value"].max())
        value_range = st.slider("Value Range", min_value=value_min, max_value=value_max, value=(value_min, value_max))

    # Apply filters
    filtered_df = df.dropna(subset=["Item Name", "Quantity", "Value"])
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
        top_quantity = filtered_df.sort_values(by="Quantity", ascending=False).head(10)
        st.dataframe(top_quantity)
        fig1, ax1 = plt.subplots()
        sns.barplot(data=top_quantity, y="Item Name", x="Quantity", ax=ax1)
        ax1.set_title("Top 10 by Quantity")
        st.pyplot(fig1)

        st.subheader("💰 Top 10 Products by Value")
        top_value = filtered_df.sort_values(by="Value", ascending=False).head(10)
        st.dataframe(top_value)
        fig2, ax2 = plt.subplots()
        sns.barplot(data=top_value, y="Item Name", x="Value", ax=ax2)
        ax2.set_title("Top 10 by Value")
        st.pyplot(fig2)

    with tab2:
        st.subheader("🧠 Product Segmentation using K-Means")
        from sklearn.cluster import KMeans
        if "Price_per_Unit" not in filtered_df.columns:
            filtered_df["Price_per_Unit"] = filtered_df["Value"] / filtered_df["Quantity"]
        kmeans = KMeans(n_clusters=3, random_state=42)
        filtered_df["Cluster"] = kmeans.fit_predict(filtered_df[["Quantity", "Value"]])
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=filtered_df, x="Quantity", y="Value", hue="Cluster", palette="tab10", s=100)
        for i in range(filtered_df.shape[0]):
            ax3.text(filtered_df["Quantity"].iloc[i], filtered_df["Value"].iloc[i], filtered_df["Item Name"].iloc[i], fontsize=8)
        ax3.set_title("Clusters based on Quantity and Value")
        st.pyplot(fig3)

    with tab3:
        st.subheader("📋 Full Filtered Dataset")
        st.dataframe(filtered_df)

        # Download filtered CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data", csv, "filtered_product_data.csv", "text/csv")
else:
    st.warning("👆 Please upload a valid product sales CSV file to continue.")
