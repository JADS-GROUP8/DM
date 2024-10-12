import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
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
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        return lr_model
    
    def evaluate_model(self, model_name, model, X_test_scaled, y_test, X_train_scaled, y_train, y_pred):
        print(model_name)
        print("--------------------------------")
        # Calculate RMSE
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        # Calculate RMSE on training set too to account for overfitting 
        rmse_train = root_mean_squared_error(y_train, model.predict(X_train_scaled))
        print(f'Root Mean Squared Error (RMSE) on training set: {rmse_train}')

        # Calculate MSE (We square the RMSE as mse func is deprecated)
        mse = rmse**2
        print(f'Mean Squared Error (MSE): {mse}')

        # Calculate MAE
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Mean Absolute Error (MAE): {mae}')

        # Calculate AIC and BIC
        n = len(y_test)
        p = X_test_scaled.shape[1]
        aic = n * np.log(mse) + 2 * p
        bic = n * np.log(mse) + p * np.log(n)

        print(f'Akaike Information Criterion (AIC): {aic}')
        print(f'Bayesian Information Criterion (BIC): {bic}')
        return rmse, mse, mae, aic, bic
    
    def display_qq_plot_residuals(self, y_test, y_pred):
        plt.figure(figsize=(12, 6)) 

        # Generate QQ plot
        plt.subplot(1, 2, 1)
        sm.qqplot(y_test - y_pred, line='s', ax=plt.gca())
        plt.title('QQ Plot')

        # Generate residuals vs fitted values plot
        plt.subplot(1, 2, 2)
        sns.regplot(x=y_pred, y=y_test - y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1}, scatter_kws={'alpha': 0.5})
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')

        plt.tight_layout()
        plt.show()

    def train_random_forest_regressor(self, X_train_scaled, y_train, n_estimators=100, random_state=42):
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X_train_scaled, y_train)
        return rf_model
    
    def plot_model_comparison(self, rmse_lr, rmse_rf ,mse_lr, mse_rf, mae_lr, mae_rf, aic_lr, aic_rf, bic_lr, bic_rf):
        # Define the models and their metrics
        models = ['Linear Regression', 'Random Forest Regression']
        metrics = ['RMSE', 'MSE', 'MAE', 'AIC', 'BIC']

        # Define the values for each metric
        rmse_values = [rmse_lr, rmse_rf]
        mse_values = [mse_lr, mse_rf]
        mae_values = [mae_lr, mae_rf]
        aic_values = [aic_lr, aic_rf]
        bic_values = [bic_lr, bic_rf]

        # Create a dictionary to hold the metric values
        metric_values = {
            'RMSE': rmse_values,
            'MSE': mse_values,
            'MAE': mae_values,
            'AIC': aic_values,
            'BIC': bic_values
        }

        # Plot each metric
        plt.figure(figsize=(12, 10))

        for i, metric in enumerate(metrics):
            plt.subplot(3, 2, i + 1)
            plt.bar(models, metric_values[metric], color=['lightcoral', 'lightblue'])
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison for Models')

        plt.tight_layout()
        plt.show()

    def compute_confidence_interval(self, model_name, predictions, y_test):
        # Calculate the prediction interval (e.g., 95% prediction interval)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)

        # Calculate the percentage of the actual test data that falls within the interval
        within_interval = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100

        print(f'Percentage of test data within the 95% prediction interval ({model_name}): {within_interval:.2f}%')