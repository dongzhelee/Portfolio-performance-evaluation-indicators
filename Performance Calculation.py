#%%Import
import pandas as pd
import numpy as np
import scipy.stats as stats


#%%Indicator aggregation framework
def FactorPerformance(NAV):

    # NAV should be a pandas DataFrame
    # NAV = pd.DataFrame(NAV)

    print('Performance evaluation')

    # Find the row number of the first value in the first column that is not 1
    first_non_one_pos = NAV.iloc[:, 0].ne(1).idxmax()

    if NAV.iloc[:, 0].eq(1).all():
        start_index = 0
    else:
        pos = NAV.index.get_loc(first_non_one_pos)
        start_index = max(pos - 1, 0)

    # Price series
    filtered_df = NAV.iloc[start_index:]
    normalized_df = filtered_df.apply(lambda x: x / x.iloc[0])

    # Return series
    returns_df = normalized_df.pct_change().dropna()

    results_df = pd.concat([annual_growth_rate(filtered_df)
                             , annual_volatility(returns_df)
                             , annual_Sharpe_ratio(filtered_df, returns_df)
                             , max_drawdown(returns_df)
                             , calmar_ratio(filtered_df, returns_df)
                             , sortino_ratio(returns_df)
                             , skewness(returns_df)
                             , kurtosis(returns_df)])

    print(results_df)

    return results_df


#%% Annualized return
def annual_growth_rate(filtered_df):
    result_df = ((filtered_df.iloc[-1] - filtered_df.iloc[0]) / filtered_df.iloc[0]).to_frame().T
    result_df = ((1 + result_df) ** (252 / len(filtered_df)) - 1) * 100
    result_df = result_df.applymap(lambda x: f"{x:.4f}%")

    result_df.index = ['Annualized return']

    return result_df


#%% Annulized volatility
def annual_volatility(returns_df):
    result_df = (returns_df.std() * (252 ** 0.5) * 100).to_frame().T
    result_df = result_df.applymap(lambda x: f"{x:.4f}%")

    result_df.index = ['Annulized volatility']
    return result_df


#%% Annulized Sharpe ratio
def annual_Sharpe_ratio(filtered_df,returns_df):
    ar = annual_growth_rate(filtered_df)
    av = annual_volatility(returns_df)
    ar = ar.apply(lambda x: x.str.replace('%', '').astype(float))
    av = av.apply(lambda x: x.str.replace('%', '').astype(float))
    # Unify the labels
    ar.index = ['Annulized Sharpe ratio']
    av.index = ['Annulized Sharpe ratio']
    result_df = ar / av
    return result_df


#%% Max drawdown
def max_drawdown(returns_df):
    # Convert daily returns to cumulative returns
    cumulative_returns = (1 + returns_df).cumprod()

    max_drawdowns = {}

    for column in cumulative_returns.columns:
        # Reset maximum peak and maximum drawdown
        max_peak = float('-inf')
        max_drawdown = 0

        for value in cumulative_returns[column]:
            if value > max_peak:
                max_peak = value  # Update maximum peak
            # Calculate maximum drawdown
            drawdown = (max_peak - value) / max_peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown  # Update maximum drawdown

        max_drawdowns[column] = max_drawdown

    result_df = pd.DataFrame(max_drawdowns, index=['Max drawdown'])
    result_df = result_df.applymap(lambda x: f"{x:.4f}%")

    return result_df


#%% Calmar ratio
def calmar_ratio(filtered_df, returns_df):
    annual_growth = annual_growth_rate(filtered_df)
    annual_growth = annual_growth.apply(lambda x: x.str.replace('%', '').astype(float))
    annual_growth.index = ['Calmar ratio']

    max_drawdown1 = max_drawdown(returns_df).apply(lambda x: x.str.replace('%', '').astype(float))
    max_drawdown1.index = ['Calmar ratio']
    max_drawdown1 = max_drawdown1.abs()

    result_df = annual_growth / max_drawdown1

    return result_df


#%% Sortino ratio
def sortino_ratio(returns_df):
    # Set 0 to be the risk-free annual rate of return

    returns_df1 = returns_df.copy()
    returns_df1[returns_df1 >= 0] = 0
    below_risk_free = returns_df[returns_df < 0]

    if len(below_risk_free) <= 1:
        print("The downside risk range is too small to calculate the Sortino ratio")
        result_df = pd.DataFrame({col: [np.nan] for col in returns_df.columns}, index=['Sortino ratio'])
        return result_df

    down_risk = returns_df1.apply(lambda x: ((x - 0) ** 2).sum() / (x.count() - 1)) ** 0.5

    sortino_ratio = (returns_df.mean() - 0) / down_risk
    yearly_sortino = sortino_ratio * np.sqrt(252)

    result_df = yearly_sortino.to_frame().T
    result_df.index = ['Sortino ratio']

    return result_df


#%% NAV growth rate skewness
def skewness(returns_df):
    skew_values = stats.skew(returns_df)
    result_df = pd.DataFrame([skew_values], columns=returns_df.columns, index=['Skewness'])

    return result_df


#%% NAV growth rate kurtosis
def kurtosis(returns_df):
    kurto_values = stats.kurtosis(returns_df)
    result_df = pd.DataFrame([kurto_values], columns=returns_df.columns, index=['kurtosis'])


    return result_df
