import pandas as pd
import eda
import modelling as md
from sklearn.metrics import mean_absolute_percentage_error

# Import the data
data = pd.read_csv("../California_Houses.csv")

# Call the function to plot the graph
if __name__ == "__main__":
    print('Explaroatory Data Analysis', "\n")
    data_analysis = eda.EDA(data)
    data_analysis.display_statistics()
    data_analysis.rename_column()
    data_analysis.display_null_values()
    print(data_analysis.return_geographical_columns())
    #data_analysis.display_pairplot()
    #data_analysis.display_boxplots()
    data_analysis.display_grouped_counts_median_income()
    data_analysis.display_grouped_median_house_value()
    #data_analysis.display_density_plots()
    #data_analysis.display_correlation_matrix()
    data_analysis.apply_pca()
    data_analysis.calculate_average_population_per_household()
    #data_analysis.display_correlation_matrix()
    #data_analysis.display_map_visualization()
    data_analysis.transform_location_data()
    #data_analysis.display_improved_map_visualization()
    #data_analysis.display_distribution_plots_based_on_closest_city()
    data = data_analysis.transform_categorical_string_column_into_dummies("Closest_City")

    print("\n", 'Modelling', "\n")
    modelling = md.Modelling()
    x_train, x_test, y_train, y_test = modelling.load_and_split_data(data)
    x_train_scaled, x_test_scaled = modelling.scale_data(x_train, x_test)

    print("\n", 'Linear Regression', "\n")
    linear_model = modelling.train_linear_regression(x_train_scaled, y_train)
    mse_lr, r2_lr, y_pred_lr = modelling.evaluate_model(linear_model, x_test_scaled, y_test)
    modelling.summarize_regression_model(linear_model, x_train, x_train.columns)
    modelling.statsmodels_summary(x_train_scaled, y_train)
    mape_lr = modelling.calculate_mean_absolute_percentage_error(y_test, y_pred_lr)
    modelling.plot_residuals(y_test, y_pred_lr)
    aic_lr, bic_lr =  modelling.calculate_aic_bic(y_test, y_pred_lr, x_train_scaled, 'Linear Regression')

    print("\n", 'Random Forest', "\n")
    random_forest_model = modelling.train_random_forest_regressor(x_train_scaled, y_train)
    mse_rf, r2_rf, y_pred_rf = modelling.evaluate_model(random_forest_model, x_test_scaled, y_test)
    mape_rf = modelling.calculate_mean_absolute_percentage_error(y_test, y_pred_rf)
    aic_rf, bic_rf = modelling.calculate_aic_bic(y_test, y_pred_rf, x_train_scaled, 'Random Forest')

    modelling.plot_model_comparison(mse_lr, mse_rf, mape_lr, mape_rf)

