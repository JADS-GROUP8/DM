import pandas as pd
import eda
import modelling as md
import numpy as np

if __name__ == "__main__":
    # Import the data
    data = pd.read_csv("../California_Houses.csv")

    print('Exploratory Data Analysis', "\n")
    data_analysis = eda.EDA(data)
    data_analysis.display_statistics()
    data_analysis.rename_column()
    data_analysis.display_null_values()
    print(data_analysis.return_geographical_columns())
    data_analysis.display_pairplot()
    data_analysis.display_boxplots()
    data_analysis.display_grouped_counts_median_income()
    data_analysis.display_grouped_median_house_value()
    data_analysis.remove_outliers()
    data_analysis.display_density_plots()
    data_analysis.display_correlation_matrix()
    data_analysis.apply_pca()
    data_analysis.calculate_average_population_per_household()
    data_analysis.display_correlation_matrix()
    data_analysis.display_map_visualization()
    data_analysis.transform_location_data()
    data_analysis.display_improved_map_visualization()
    data_analysis.display_distribution_plots_based_on_closest_city()
    data = data_analysis.transform_categorical_string_column_into_dummies("Closest_City")

    print("\n", 'Modelling', "\n")
    modelling = md.Modelling()
    x_train, x_test, y_train, y_test = modelling.load_and_split_data(data)
    x_train_scaled, x_test_scaled = modelling.scale_data(x_train, x_test)

    print("\n", 'Linear Regression', "\n")
    linear_model = modelling.train_linear_regression(x_train_scaled, y_train)
    y_pred_lr = linear_model.predict(x_test_scaled)
    rmse_lr, mse_lr, mae_lr, aic_lr, bic_lr = modelling.evaluate_model('Linear Regression', linear_model, x_test_scaled, y_test, x_train_scaled, y_train, y_pred_lr)
    modelling.display_qq_plot_residuals(y_test, y_pred_lr)

    print("\n", 'Random Forest', "\n")
    random_forest_model = modelling.train_random_forest_regressor(x_train_scaled, y_train)
    y_pred_rf = random_forest_model.predict(x_test_scaled)
    all_tree_predictions = np.array([tree.predict(x_test_scaled) for tree in random_forest_model.estimators_])
    rmse_rf, mse_rf, mae_rf, aic_rf, bic_rf = modelling.evaluate_model('Random Forest', random_forest_model, x_test_scaled, y_test, x_train_scaled, y_train, y_pred_rf)

    print("\n", 'Evaluation', "\n")
    modelling.plot_model_comparison(rmse_lr, rmse_rf ,mse_lr, mse_rf, mae_lr, mae_rf, aic_lr, aic_rf, bic_lr, bic_rf)
    modelling.compute_confidence_interval('Linear Regression', y_pred_lr, y_test)
    modelling.compute_confidence_interval('Random Forest', all_tree_predictions, y_test)