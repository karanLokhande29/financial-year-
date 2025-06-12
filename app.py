
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Product Sales & Clustering Dashboard", layout="wide")
st.title("📊 Product Sales & Clustering Dashboard")

df = pd.read_csv("clustering_output.csv")

tab1, tab2, tab3 = st.tabs(["Top Products", "Clustering", "Raw Data"])

with tab1:
    st.subheader("🔝 Top 10 Products by Quantity")
    top_quantity = df.sort_values(by="Quantity", ascending=False).head(10)
    st.dataframe(top_quantity)
    fig1, ax1 = plt.subplots()
    sns.barplot(data=top_quantity, y="Item Name", x="Quantity", ax=ax1)
    ax1.set_title("Top 10 by Quantity")
    st.pyplot(fig1)

    st.subheader("💰 Top 10 Products by Value")
    top_value = df.sort_values(by="Value", ascending=False).head(10)
    st.dataframe(top_value)
    fig2, ax2 = plt.subplots()
    sns.barplot(data=top_value, y="Item Name", x="Value", ax=ax2)
    ax2.set_title("Top 10 by Value")
    st.pyplot(fig2)

with tab2:
    st.subheader("🧠 Product Segmentation using K-Means")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="Quantity", y="Value", hue="Cluster", palette="tab10", s=100)
    for i in range(df.shape[0]):
        ax3.text(df["Quantity"][i], df["Value"][i], df["Item Name"][i], fontsize=8)
    ax3.set_title("Clusters based on Quantity and Value")
    st.pyplot(fig3)

with tab3:
    st.subheader("📋 Full Dataset")
    st.dataframe(df)
