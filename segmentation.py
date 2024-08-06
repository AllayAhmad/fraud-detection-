import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
file_path = 'supermarket_sales.csv'
df = pd.read_csv(file_path)

# Select relevant attributes
attributes = ['Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Total', 'Payment', 'Rating']
df_selected = df[attributes]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Unit price', 'Quantity', 'Total', 'Rating']),
        ('cat', OneHotEncoder(), ['Customer type', 'Gender', 'Product line', 'Payment'])
    ])

# Apply preprocessing
data_preprocessed = preprocessor.fit_transform(df_selected)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_preprocessed)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Display the first few rows of the dataframe with cluster labels
print(df.head())

# Optional: Save the dataframe with cluster labels to a new CSV file
df.to_csv('supermarket_sales_with_clusters.csv', index=False)
