import glob
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    data_df = pd.read_csv('VehicleData-1.csv')
    data_df['DATE'] = pd.to_datetime(data_df['DATE'], format='%m/%d/%y')
    return data_df

def load_additional_data(data_df, files):

    for file_path in files:
        temp = pd.read_csv(file_path)
        temp['DATE'] = pd.to_datetime(temp['observation_date'])
        temp = temp.drop(columns= ['observation_date'])    
        data_df = pd.merge(data_df, temp, on='DATE', how='left')
        
    # Set date as index
    data_df = data_df.set_index('DATE').sort_index()

    if 'TERMCBAUTO48NS' in data_df.columns:
        data_df['TERMCBAUTO48NS'] = data_df['TERMCBAUTO48NS'].ffill()
        data_df['TERMCBAUTO48NS'] = data_df['TERMCBAUTO48NS'].bfill()
    return data_df


def plot_cols(df, col_list):
    fig, host = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(right=0.75)  # make space for extra y-axes

    axes = [host]
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Compute symmetric limits around zero for each column
    y_limits = {}
    for col in col_list:
        max_abs = max(abs(df[col].min()), abs(df[col].max()))
        y_limits[col] = (-max_abs, max_abs)

    # Plot the first column
    host.plot(df.index, df[col_list[0]], linestyle='--', color=colors[0], label=col_list[0])
    host.set_ylabel(col_list[0], color=colors[0])
    host.tick_params(axis='y', labelcolor=colors[0])
    host.set_ylim(y_limits[col_list[0]])

    # Additional axes
    for i, col in enumerate(col_list[1:]):
        ax = host.twinx()
        ax.spines["right"].set_position(("axes", 1 + 0.1*i))
        ax.plot(df.index, df[col], linestyle='--', color=colors[(i+1)%10], label=col)
        ax.set_ylabel(col, color=colors[(i+1)%10])
        ax.tick_params(axis='y', labelcolor=colors[(i+1)%10])
        ax.set_ylim(y_limits[col])
        axes.append(ax)

    host.set_xlabel('Date')
    host.grid(True)
    plt.xticks(rotation=45)

    # Combine legends
    lines, labels = [], []
    for ax in axes:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    host.legend(lines, labels, loc='upper left')

    plt.show()


def extract_granger_results(data, maxlag, alpha=0.05):
    results = grangercausalitytests(data, maxlag, verbose=False)
    out = []
    cols = list(data.columns)
    for lag, test_dict in results.items():
        # SSR-based F test p-value
        pval = test_dict[0]['ssr_ftest'][1]
        decision = "Reject H0 (Granger Causality)" if pval < alpha else "Fail to Reject H0"

        out.append({
            'result': cols[0],
            'exog': cols[1],
            "lag": lag,
            "p_value": pval,
            "decision": decision
        })

    return out

def granger_results(df, result_cols, exog_cols):
    granger_results = []
    for result_col in result_cols:
        for exog_col in exog_cols:
            granger_data =df[[result_col, exog_col]]
            granger_results += extract_granger_results(granger_data, 5)
    granger_df = pd.DataFrame(granger_results)
    granger_df['exog_flag'] = granger_df['decision'] == 'Reject H0 (Granger Causality)'
    return granger_df

def interpret_corr(corr):
    if abs(corr) < 0.2:
        return "negligible"
    elif abs(corr) < 0.5:
        return "moderate"
    else:
        return "strong"
    

import statsmodels.api as sm

def granger_effect_direction(df, causal_pairs):
    """
    causal_pairs: list of tuples (x_col, y_col, lag)
    returns: DataFrame with coefficients and direction
    """
    results = []

    for x_col, y_col, lag in causal_pairs:
        # create lagged X columns
        df_lagged = df[[y_col]].copy()
        df_lagged[f'{x_col}_lagged'] = df[x_col].shift(lag)
        df_lagged = df_lagged.dropna()
        
        X = df_lagged[f'{x_col}_lagged']
        X = sm.add_constant(X)
        y = df_lagged[y_col]
        
        model = sm.OLS(y, X).fit()
        coef = model.params[1:]  # exclude constant
        
        # Determine majority sign among coefficients
        if all(c > 0 for c in coef):
            direction = "positive"
        elif all(c < 0 for c in coef):
            direction = "negative"
        else:
            direction = "mixed"
        
        results.append({
            'X': x_col,
            'Y': y_col,
            'lag': lag,
            'coef': coef.values,
            'direction': direction
        })
    
    return pd.DataFrame(results)


def plot_df_subplots(df, overlay=False, figsize=(14, 6),  title=None):

    if overlay:
        # --- Overlay Plot ---
        plt.figure(figsize=figsize)
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)
        plt.legend()
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Value")
        if title:
            plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    else:
        # --- Subplots ---
        n = df.shape[1]  # number of columns
        fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True)

        # Ensure axes is iterable
        if n == 1:
            axes = [axes]

        for i, col in enumerate(df.columns):
            axes[i].plot(df.index, df[col], label=col)
            axes[i].set_title(col)
            axes[i].set_ylabel("Value")
            axes[i].grid(True)

        axes[-1].set_xlabel("Date")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.show()




from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_var(var_data, test_horizon, exog_data=None):
 

    # -------------------------
    # Train/Test Split
    # -------------------------
    train_data = var_data.iloc[:-test_horizon]
    test_data  = var_data.iloc[-test_horizon:]

    if exog_data is not None:
        exog_train = exog_data.iloc[:-test_horizon]
        exog_test  = exog_data.iloc[-test_horizon:]
    else:
        exog_train = exog_test = None

    history_y = train_data.copy()
    history_exog = exog_train.copy() if exog_train is not None else None

    rolling_forecasts = []


    if exog_data is None:
        model = VAR(history_y)

        order_selection = model.select_order(maxlags=10)
        best_order = order_selection.bic 
        print(f'Selected Order: {best_order}')
        print(order_selection.summary())

    else:
        best_bic = np.inf
        order_results = []
        for p in range(0, 3):
            for q in range(0, 3):
                print(p, ',', q)
                try:
                    test_model = VARMAX(history_y, exog=history_exog, order=(p, q))
                    test_res   = test_model.fit(disp=False)

                    order_results.append({"p": p, "q": q, "bic": test_res.bic, "aic": test_res.aic})

                    if test_res.bic < best_bic:
                        best_bic = test_res.bic
                        best_order = (p, q)

                except Exception as e:
                    order_results.append({"p": p, "q": q, "bic": np.inf, "aic": np.inf})
                    print(f"VARMAX order ({p},{q}) failed with error: {e}")
                    pass

        # Convert to DataFrame summary table
        varmax_order_selection = pd.DataFrame(order_results)
        display(varmax_order_selection)
        model = VARMAX(history_y, exog=history_exog, order=best_order)

    # Rolling Forecast Loop
    for t in range(test_horizon):

        if exog_data is None:
            # --------------------
            # VAR (no exogenous)
            # --------------------
            model = VAR(history_y)
            result = model.fit(best_order, trend='n')

            yhat = result.forecast(history_y.values[-best_order:], steps=1)[0]

        else:
            # --------------------
            # VARX using VARMAX
            # --------------------
            model = VARMAX(history_y, exog=history_exog, order=best_order)

            result = model.fit(disp=False)

            next_exog = exog_test.iloc[t:t+1]

            yhat = result.forecast(steps=1, exog=next_exog).values[0]

        rolling_forecasts.append(yhat)

        # Expand history with actual next point
        next_actual_y = test_data.iloc[t:t+1]
        history_y = pd.concat([history_y, next_actual_y])

        if exog_data is not None:
            next_actual_exog = exog_test.iloc[t:t+1]
            history_exog = pd.concat([history_exog, next_actual_exog])


    forecast_df = pd.DataFrame(
        np.array(rolling_forecasts),
        index=test_data.index,
        columns=test_data.columns
    )


    metrics = {}

    for col in test_data.columns:
        actual = test_data[col]
        pred = forecast_df[col]

        rmse = mean_squared_error(actual, pred)
        mae  = mean_absolute_error(actual, pred)

        metrics[col] = {"RMSE": rmse, "MAE": mae}

    metrics_df = pd.DataFrame(metrics).T

    return {
        "metrics": metrics_df,
        "actual": test_data,
        "pred": forecast_df,
        "model": result,
        "order": best_order
    }

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
def rolling_arimax_forecast(y, exog, horizon, p=3, q=0):
    d = 0
    # Train/test split
    y_train = y.iloc[:-horizon]
    y_test  = y.iloc[-horizon:]
    x_train = exog.iloc[:-horizon]
    x_test  = exog.iloc[-horizon:]

    preds = []
    hist_y = y_train.copy()
    hist_x = x_train.copy()

    # Rolling one-step loop
    for i in range(horizon):
        model = SARIMAX(hist_y, exog=hist_x, order=(p, d, q), trend='n')
        result = model.fit(disp=False, warn_convergence=False)

        # forecast uses NEXT exog row
        next_x = x_test.iloc[i:i+1]
        fc = result.forecast(steps=1, exog=next_x)
        preds.append(fc.iloc[0])

        # update history
        hist_y = pd.concat([hist_y, y_test.iloc[i:i+1]])
        hist_x = pd.concat([hist_x, next_x])

    pred_series = pd.Series(preds, index=y_test.index, name="Prediction")

    forecast = pd.DataFrame({'actuals': y_test, 'predictions': pred_series})

    metrics = {
        "RMSE": mean_squared_error(y_test, pred_series),
        "MAE":  mean_absolute_error(y_test, pred_series)
    }
    return {'model':result,
        "forecast": forecast,
        "metrics": metrics
    }

def rolling_by_dates_test(
    y,
    exog,
    rolling_func,
    cutoff_dates,
    forecast_horizon=1,
    model_params=None
):
    results_dict = {}
    forecasts_dict = {}

    for dt in cutoff_dates:
        if dt not in y.index:
            raise ValueError(f"Cutoff date {dt} not in y index.")
        else: print(f"Predicting based on {dt}")

        window_index = y.index[:y.index.get_loc(dt) + forecast_horizon]
        y_train = y.loc[window_index]
        exog_train = exog.loc[window_index] if exog is not None else None

        # Run your rolling forecast function
        out = rolling_func(
            y_train,
            exog_train,
            horizon=forecast_horizon,
            **(model_params or {})
        )

        forecast = out["forecast"]

        forecasts_dict[dt] = forecast

        results_dict[dt] = out['metrics']

    return pd.DataFrame.from_dict(results_dict, orient="index"), forecasts_dict, out['model']

