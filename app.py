import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Title and description
st.title("Portfolio Management Dashboard")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Choose a section",
    ["Portfolio Settings", "Historical Prices", "Portfolio Performance", "Risk Analysis", "Correlation Analysis"]
)

# Sidebar for user inputs
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, GOOG, AMZN")
weights_input = st.sidebar.text_input("Enter portfolio weights (comma-separated)", "0.25, 0.25, 0.25, 0.25")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Convert user inputs
try:
    tickers_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    weights = [float(weight) for weight in weights_input.split(",") if weight.strip()]
except ValueError:
    st.sidebar.error("Please ensure weights are numeric and properly separated.")
    st.stop()

# Validate weights
if len(tickers_list) != len(weights):
    st.sidebar.error("Number of tickers and weights must match.")
    st.stop()

if not np.isclose(sum(weights), 1):
    st.sidebar.error("Portfolio weights must sum to 1.")
    st.stop()

# Download data
@st.cache
def fetch_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)["Adj Close"]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

data = fetch_data(tickers_list, start_date, end_date)

# Handle case where no data is retrieved
if data.empty:
    st.error("No data retrieved. Please check the tickers or date range.")
    st.stop()

# Calculate common metrics globally
returns = data.pct_change().dropna()  # Daily returns
portfolio_returns = returns @ np.array(weights)  # Weighted portfolio returns
cumulative_returns = (1 + portfolio_returns).cumprod() - 1  # Cumulative returns

# Portfolio Composition Section
if menu == "Portfolio Settings":
    st.subheader("Portfolio Composition")
    st.write("**Tickers**:", tickers_list)
    st.write("**Weights**:", weights)

# Historical Prices Section
if menu == "Historical Prices":
    st.subheader("Historical Prices")
    fig, ax = plt.subplots(figsize=(10, 5))
    for ticker in tickers_list:
        if ticker in data.columns:
            ax.plot(data.index, data[ticker], label=ticker)
    ax.set_title("Historical Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# Portfolio Performance Section
if menu == "Portfolio Performance":
    st.subheader("Portfolio Performance")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(cumulative_returns.index, cumulative_returns, label="Portfolio", color="purple")
    ax2.set_title("Cumulative Portfolio Returns")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Returns")
    ax2.legend()
    st.pyplot(fig2)

    # Display portfolio statistics
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    st.write("**Annualized Return:** {:.2%}".format(annualized_return))
    st.write("**Annualized Volatility:** {:.2%}".format(annualized_volatility))
    st.write("**Sharpe Ratio:** {:.2f}".format(sharpe_ratio))

# Risk Analysis Section
if menu == "Risk Analysis":
    st.subheader("Risk Analysis")

    # Value at Risk (VaR)
    confidence_level = 0.05
    VaR = np.percentile(portfolio_returns, confidence_level * 100)
    st.write(f"**Value at Risk (5%):** {VaR:.2%}")

    # Conditional Value at Risk (CVaR)
    CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
    st.write(f"**Conditional Value at Risk (5%):** {CVaR:.2%}")

    # Maximum Drawdown
    cumulative_returns_plus_one = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns_plus_one.cummax()
    drawdown = (cumulative_returns_plus_one - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")

    # Drawdown Visualization
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(drawdown.index, drawdown, label="Drawdown", color="red")
    ax3.set_title("Portfolio Drawdown")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Drawdown")
    ax3.legend()
    st.pyplot(fig3)

# Correlation Analysis Section
if menu == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    correlation_matrix = returns.corr()

    st.write("**Correlation Matrix:**")
    st.dataframe(correlation_matrix)

    # Heatmap for Correlation
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    cax = ax4.matshow(correlation_matrix, cmap="coolwarm")
    fig4.colorbar(cax)
    ax4.set_xticks(range(len(correlation_matrix.columns)))
    ax4.set_yticks(range(len(correlation_matrix.columns)))
    ax4.set_xticklabels(correlation_matrix.columns, rotation=45)
    ax4.set_yticklabels(correlation_matrix.columns)
    ax4.set_title("Stock Correlation Heatmap", pad=20)
    st.pyplot(fig4)
