import pandas as pd
import eda

# Import the data
data = pd.read_csv("../California_Houses.csv")

# Call the function to plot the graph
if __name__ == "__main__":
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
    print(data.columns)