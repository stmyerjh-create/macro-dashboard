import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fredapi import Fred

fred = Fred(api_key="34f96e5bb448b23ea70f0f56e771a6a9")
st.header("FRED Economic Data")

series = st.selectbox(
    "Select an economic indicator",
    ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS"]
)

data = fred.get_series(series)

df = data.reset_index()
df.columns = ["Date", "Value"]

fig = px.line(df, x="Date", y="Value", title=series)

st.plotly_chart(fig)

st.set_page_config(page_title="Macroeconomic Dashboard", layout="wide")

st.title("Macroeconomic Simulation Dashboard")

st.sidebar.title("Model Navigation")

model_choice = st.sidebar.radio(
    "Choose Model",
    ("Model 1: Consumption-Savings",
     "Model 2: Robinson Crusoe",
     "Model 3: Endogenous Labor")
)

st.sidebar.header("Parameters")

beta = st.sidebar.slider("Discount Factor β", 0.85, 0.99, 0.95)
sigma = st.sidebar.slider("Risk Aversion σ", 1.0, 5.0, 2.0)
r = st.sidebar.slider("Interest Rate r", 0.01, 0.05, 0.03)

periods = st.sidebar.slider("Simulation Length", 100, 500, 200)

income_states = np.array([0.8, 1.2])

P = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])

grid_size = 150
a_grid = np.linspace(0,10,grid_size)

def utility(c):
    if c <= 0:
        return -1e10
    if sigma == 1:
        return np.log(c)
    return (c**(1-sigma))/(1-sigma)

@st.cache_data
def solve_vfi():

    V = np.zeros((grid_size,2))
    policy = np.zeros((grid_size,2))

    diff = 1
    tol = 1e-5

    while diff > tol:

        V_new = np.zeros_like(V)

        for s in range(2):

            y = income_states[s]

            for i,a in enumerate(a_grid):

                values = []

                for a_next in a_grid:

                    c = y + (1+r)*a - a_next

                    if c <= 0:
                        values.append(-1e10)
                    else:

                        EV = 0

                        for s_next in range(2):
                            EV += P[s,s_next]*V[np.argmin(abs(a_grid-a_next)),s_next]

                        values.append(utility(c) + beta*EV)

                best = np.argmax(values)

                V_new[i,s] = values[best]
                policy[i,s] = a_grid[best]

        diff = np.max(abs(V_new-V))
        V = V_new

    return policy

policy = solve_vfi()

def simulate_model():

    a = 1
    state = 0

    assets = []
    consumption = []
    income = []

    for t in range(periods):

        idx = np.argmin(abs(a_grid-a))

        a_next = policy[idx,state]

        y = income_states[state]

        c = y + (1+r)*a - a_next

        assets.append(a)
        consumption.append(c)
        income.append(y)

        a = a_next

        state = np.random.choice([0,1], p=P[state])

    return np.array(assets),np.array(consumption),np.array(income)

def simulate_crusoe():

    alpha = 0.35
    delta = 0.05

    k = 1

    capital = []
    consumption = []
    output = []

    for t in range(periods):

        z = np.random.choice([0.9,1.1])

        y = z*(k**alpha)

        c = 0.7*y
        k_next = y - c + (1-delta)*k

        capital.append(k)
        consumption.append(c)
        output.append(y)

        k = k_next

    return np.array(capital),np.array(consumption),np.array(output)

def simulate_labor():

    wage_states = [0.8,1.2]

    state = 0

    labor = []
    consumption = []
    wage_series = []

    for t in range(periods):

        w = wage_states[state]

        L = min(1,max(0,0.5 + 0.3*(w-1)))

        c = w*L

        labor.append(L)
        consumption.append(c)
        wage_series.append(w)

        state = np.random.choice([0,1],p=P[state])

    return np.array(labor),np.array(consumption),np.array(wage_series)

def show_moments(series1,series2):

    mean_val = np.mean(series1)
    var_val = np.var(series1)
    corr = np.corrcoef(series1,series2)[0,1]

    df = pd.DataFrame({
        "Statistic":["Mean","Variance","Correlation with Income/Output"],
        "Value":[mean_val,var_val,corr]
    })

    st.table(df)

if model_choice == "Model 1: Consumption-Savings":

    st.header("Stochastic Consumption-Savings Model")

    assets,consumption,income = simulate_model()

    fig,ax = plt.subplots()

    ax.plot(consumption,label="Consumption")
    ax.plot(assets,label="Assets")
    ax.plot(income,label="Income")

    ax.legend()

    st.pyplot(fig)

    show_moments(consumption,income)

    st.write(f"""
    With discount factor β = {beta}, households smooth consumption over time.
    The model generates an average consumption level of {np.mean(consumption):.2f}.
    Consumption is less volatile than income, demonstrating consumption smoothing.
    """)

elif model_choice == "Model 2: Robinson Crusoe":

    st.header("Robinson Crusoe Economy")

    capital,consumption,output = simulate_crusoe()

    fig,ax = plt.subplots()

    ax.plot(capital,label="Capital")
    ax.plot(consumption,label="Consumption")
    ax.plot(output,label="Output")

    ax.legend()

    st.pyplot(fig)

    show_moments(consumption,output)

    st.write(f"""
    In the Robinson Crusoe model, capital accumulation determines future output.
    Average consumption equals {np.mean(consumption):.2f} and fluctuates with productivity shocks.
    """)

elif model_choice == "Model 3: Endogenous Labor":

    st.header("Endogenous Labor Supply Model")

    labor,consumption,wages = simulate_labor()

    fig,ax = plt.subplots()

    ax.plot(labor,label="Labor")
    ax.plot(consumption,label="Consumption")
    ax.plot(wages,label="Wages")

    ax.legend()

    st.pyplot(fig)

    show_moments(consumption,wages)

    st.write(f"""
    Labor supply responds to wage shocks. When wages rise, households supply more labor.
    Average labor supply in the simulation is {np.mean(labor):.2f}.
    """)