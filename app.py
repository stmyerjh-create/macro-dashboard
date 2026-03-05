import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Macro Dashboard")

st.write("Simple macroeconomic simulation dashboard")

# User inputs
alpha = st.slider("Capital Share (α)", 0.1, 0.9, 0.33)
beta = st.slider("Discount Factor (β)", 0.8, 0.99, 0.95)
delta = st.slider("Depreciation Rate (δ)", 0.01, 0.2, 0.05)

st.write("### Production Function")

# Generate capital values
k = np.linspace(0.1, 10, 100)

# Cobb-Douglas production function
y = k ** alpha

df = pd.DataFrame({
    "Capital": k,
    "Output": y
})

fig = px.line(df, x="Capital", y="Output", title="Production Function")

st.plotly_chart(fig)

st.write("### Model Parameters")

st.write(f"""
Capital Share α: {alpha}  
Discount Factor β: {beta}  
Depreciation δ: {delta}
""")