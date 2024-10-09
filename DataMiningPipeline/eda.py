import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopandas as gpd
from geodatasets import get_path


class EDA:

    def __init__(self, data):
        self.data = data

    def display_statistics(self):
        print("Data Statistics:", "\n")
        print(self.data.describe().transpose(), "\n")
    
    def rename_column(self):
        self.data.rename(columns={'Distance_to_LA':'Distance_to_LosAngeles'}, inplace=True) 

    def display_null_values(self):
        print("Null Values in Data:", "\n")
        print(self.data.isna().sum(), "\n")

    def return_geographical_columns(self):
        columns_geographical = self.data.columns[self.data.columns.str.contains('Distance|Latitude|Longitude')]
        return columns_geographical
    
    def display_pairplot(self):
        sns.pairplot(self.data.drop(columns=self.return_geographical_columns()), plot_kws={'alpha':0.1})  # Set opacity to 10%)
        plt.show()

    def display_boxplots(self):
        cols_per_row = 4
        num_rows = int(np.ceil(len(self.data.columns) / cols_per_row))
        fig, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
        axes = axes.flatten()
        flierprops = dict(marker='o', markerfacecolor='white', markersize=6, linestyle='none', alpha=0.1)  # Set opacity to 10%

        for index, column in enumerate(self.data.columns):
            ax = axes[index]
            self.data.boxplot(column=column, ax=ax, vert=True, flierprops=flierprops)
            ax.set_title(f'Boxplot of {column}')

        for j in range(index + 1, len(axes)):
            fig.delaxes(axes[j]) 
                
        plt.tight_layout()
        plt.show()

    def display_grouped_counts_median_income(self):
        print(self.data.groupby("Median_Income").count().sort_values(by="Median_Income", ascending=False)["Median_House_Value"].rename("Count of rows"))

    def display_grouped_median_house_value(self):
        print(self.data.groupby("Median_House_Value").count().sort_values(by="Median_House_Value", ascending=False)["Median_Income"].rename("Count of rows"))

    def display_density_plots(self):
         # Plot density plots for all variables in a single figure
        fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20, 15))
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            sns.histplot(self.data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')

        plt.tight_layout()
        plt.show()

    def display_correlation_matrix(self):
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.show()

    def apply_pca(self):
        # Extract the relevant columns
        bedrooms_rooms = self.data[['Tot_Bedrooms', 'Tot_Rooms']]

        # Standardize the data
        scaler = StandardScaler()
        bedrooms_rooms_scaled = scaler.fit_transform(bedrooms_rooms)

        # Apply PCA
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(bedrooms_rooms_scaled)

        # Add the principal component back to the dataframe
        self.data['Bedrooms_Rooms_block_PCA'] = principal_components

    def calculate_average_population_per_household(self):
        self.data['Avg_Pop_Per_Household'] = self.data['Population'] / self.data['Households']
        self.data.drop(columns=['Households', 'Tot_Bedrooms', 'Tot_Rooms'], inplace=True)  

    def display_map_visualization(self):
        gdf = gpd.GeoDataFrame(
            self.data, geometry=gpd.points_from_xy(self.data.Longitude, self.data.Latitude), crs="EPSG:4326"
        )

        world = gpd.read_file(get_path("naturalearth.land"))

        # Create a DataFrame for the cities with their coordinates
        gdf_cities = gpd.GeoDataFrame({
            'City': ['SanJose', 'SanFrancisco', 'LosAngeles', 'SanDiego'],
            'Latitude': [37.3382, 37.7749, 34.0522, 32.7157],
            'Longitude': [-121.8863, -122.4194, -118.2437, -117.1611]
        }, geometry=gpd.points_from_xy([-121.8863, -122.4194, -118.2437, -117.1611], [37.3382, 37.7749, 34.0522, 32.7157]), crs="EPSG:4326")

        # Plot the cities on the map
        ax = world.clip([-130,32.5,-114,42]).plot(color="white", edgecolor="black")
        gdf.plot(ax=ax, color="red", markersize=2)
        gdf_cities.plot(ax=ax, color="blue", markersize=100, marker='o', label='Major Cities')

        # Add labels for the cities with an offset to the left and a line connecting to the point
        for x, y, label in zip(gdf_cities.geometry.x, gdf_cities.geometry.y, gdf_cities['City']):
            ax.annotate(label, xy=(x, y), xytext=(x - 1.5, y - 0.2),
                        textcoords='data', fontsize=12, ha='right',
                        arrowprops=dict(arrowstyle="-", color='black'))

        ax.axis('off')
        plt.legend(loc='upper right')
        plt.legend()
        plt.show()

    def transform_location_data(self):
        # Create new Closest City column for Los Angeles and San Francisco San Jose and San Diego using distance columns
        self.data['Closest_Distance_to_city'] = self.data[[column for column in self.data.columns if column.startswith('Distance_')]].min(axis=1)

        self.data['Closest_Distance_to_city'] = self.data[['Distance_to_LosAngeles', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']].min(axis=1)

        self.data['Closest_City'] = np.where(self.data['Distance_to_LosAngeles'] == self.data['Closest_Distance_to_city'], 'LosAngeles',
                            np.where(self.data['Distance_to_SanDiego'] == self.data['Closest_Distance_to_city'], 'SanDiego',
                            np.where(self.data['Distance_to_SanJose'] == self.data['Closest_Distance_to_city'], 'SanJose', 'SanFrancisco')))

        self.data.drop(columns=['Distance_to_LosAngeles', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco'], inplace=True)

    def display_improved_map_visualization(self):
        gdf = gpd.GeoDataFrame(
            self.data, geometry=gpd.points_from_xy(self.data.Longitude, self.data.Latitude), crs="EPSG:4326"
        )
        
        world = gpd.read_file(get_path("naturalearth.land"))

        # Create a DataFrame for the cities with their coordinates
        gdf_cities = gpd.GeoDataFrame({
            'City': ['SanJose', 'SanFrancisco', 'LosAngeles', 'SanDiego'],
            'Latitude': [37.3382, 37.7749, 34.0522, 32.7157],
            'Longitude': [-121.8863, -122.4194, -118.2437, -117.1611]
        }, geometry=gpd.points_from_xy([-121.8863, -122.4194, -118.2437, -117.1611], [37.3382, 37.7749, 34.0522, 32.7157]), crs="EPSG:4326")

        # Define colors for each city
        colors = {'LosAngeles': 'red', 'SanDiego': 'green', 'SanJose': 'blue', 'SanFrancisco': 'purple'}

        # Create a GeoDataFrame for the points with the closest city
        gdf['Closest_City'] = self.data['Closest_City']

        # Plot the world map restricted to California
        ax = world.clip([-130, 32.5, -114, 42]).plot(color="white", edgecolor="black")

        # Plot the points colored by the closest city
        for city, color in colors.items():
            gdf[gdf['Closest_City'] == city].plot(ax=ax, color=color, markersize=10, label=city)

        # Plot the cities on the map
        gdf_cities.plot(ax=ax, color="black", markersize=100, marker='o')

        # Add labels for the cities with an offset to the left and a line connecting to the point
        for x, y, label in zip(gdf_cities.geometry.x, gdf_cities.geometry.y, gdf_cities['City']):
            ax.annotate(label, xy=(x, y), xytext=(x - 1.5, y - 0.2),
            textcoords='data', fontsize=12, ha='right',
            arrowprops=dict(arrowstyle="-", color='black'))

        # Remove axes completely
        ax.axis('off')

        # Move the legend to the right
        plt.legend(loc='upper right')
        plt.show()

    def display_distribution_plots_based_on_closest_city(self):
        colors = {'LosAngeles': 'red', 'SanDiego': 'green', 'SanJose': 'blue', 'SanFrancisco': 'purple'}

        columns_to_plot = [col for col in self.data.columns if col not in ['Longitude', 'Latitude',
                                                            'Distance_to_LosAngeles', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco',
                                                            'Closest_Distance_to_city', 'Closest_City']]

        fig, axes = plt.subplots(nrows=len(columns_to_plot), ncols=1, figsize=(15, 30))
        axes = axes.flatten()

        for i, col in enumerate(columns_to_plot):
            sns.kdeplot(data=self.data, x=col, hue='Closest_City', common_norm=False, palette=colors, ax=axes[i])
            axes[i].set_title(f'Distribution of {col} by Closest City')

        plt.tight_layout()
        plt.show()

    def transform_categorical_string_column_into_dummies(self, column):
        if column not in self.data:
            return self.data
        self.data = pd.get_dummies(self.data, columns=[column])
        return self.data
