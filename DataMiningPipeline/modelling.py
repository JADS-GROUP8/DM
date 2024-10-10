import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


class Modelling:

    def __init__(self):
        pass

    def load_and_split_data(self, data, target_variable="Median_House_Value", test_size=0.2, random_state=42):
        X = data.drop(columns=[target_variable, "Latitude", "Longitude"])
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_linear_regression(self, X_train_scaled, y_train):
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        return model
    
    def evaluate_model(self, model, X_test_scaled, y_test):
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        return mse, r2, y_pred
    
    def summarize_regression_model(self, model, X, feature_names):
        print("Linear Regression Model Summary")
        print("--------------------------------")
        print(f"Intercept: {model.intercept_}")
        print("Coefficients:")
        for feature, coef in zip(feature_names, model.coef_):
            print(f"  {feature}: {coef}")

    def statsmodels_summary(self, X_train_scaled, y_train):
        X_train_const = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_train_const).fit()
        print(model_sm.summary())
        return model_sm
    
    def plot_residuals(self, y_test, y_pred):
        residuals = y_test - y_pred

        plt.figure(figsize=(15, 6))

        # Residuals vs Fitted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')

        # Distribution of Residuals
        plt.subplot(1, 2, 2)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Distribution of Residuals')

        plt.tight_layout()
        plt.show()

    def train_random_forest_regressor(self, X_train_scaled, y_train, n_estimators=100, random_state=42):
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X_train_scaled, y_train)
        return rf_model

    def calculate_aic_bic(self, y_test, y_pred, X_test_scaled, model_name):
        n = len(y_test)
        mse = mean_squared_error(y_test, y_pred)
        p = X_test_scaled.shape[1]  # Number of features

        # Calculate AIC
        aic = n * np.log(mse) + 2 * p

        # Calculate BIC
        bic = n * np.log(mse) + p * np.log(n)

        print(f"Model: {model_name}")
        print(f"AIC: {aic}")
        print(f"BIC: {bic}")
        return aic, bic
    
    def calculate_mean_absolute_percentage_error(self, y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f"Mean Absolute Percentage Error: {mape}")
        return mape
    
    def plot_model_comparison(self, mse_lr, mse_rf, mape_lr, mape_rf):
        models = ['Linear Regression', 'Random Forest Regression']
        rmse_values = [np.sqrt(mse_lr), np.sqrt(mse_rf)]
        mape_values = [mape_lr, mape_rf]

        plt.figure(figsize=(12, 6))

        # Plot RMSE
        plt.subplot(1, 2, 1)
        plt.bar(models, rmse_values, color=['cyan', 'salmon'])
        plt.ylabel('RMSE')
        plt.title('RMSE Comparison for Models')

        # Plot MAPE
        plt.subplot(1, 2, 2)
        plt.bar(models, mape_values, color=['cyan', 'salmon'])
        plt.ylabel('MAPE')
        plt.title('MAPE Comparison for Models')

        plt.tight_layout()
        plt.show()