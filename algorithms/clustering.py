from statistics import stdev
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler 
from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer
from utils import encode_states, encode_loss, adjust_dollars


def k_means(data, k, max_iterations=1000):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    numerical_data = data[numerical_columns].values

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(numerical_data)

    n_samples, n_numerical_features = data_normalized.shape
    centroids = kmeans_plusplus_init(data_normalized, k)

    for _ in range(max_iterations):
        labels, _ = pairwise_distances_argmin_min(data_normalized, centroids)
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = np.mean(data_normalized[labels == i], axis=0)

    return centroids, labels

def kmeans_plusplus_init(data, k):
    centroids = [data[np.random.choice(len(data))]]
    for _ in range(1, k):
        distances = pairwise_distances_argmin_min(data, centroids)[1]
        probabilities = distances / np.sum(distances)
        new_centroid_index = np.random.choice(len(data), p=probabilities)
        centroids.append(data[new_centroid_index])
    return np.array(centroids)


def cluster_statistics(data, labels):
    # Add the assigned clusters to the df
    stats_data = data
    stats_data['cluster'] = labels

    # Dataframe to store cluster stats
    cluster_stats = pd.DataFrame(index=range(len(np.unique(labels))), columns=['Cluster', 'Mode_State', 'Mean_Mag', 'Mean_Loss', 'Mean_Area', 'Mean_Cas_Ratio'])

    for cluster in np.unique(labels):
        cluster_data = stats_data[stats_data['cluster'] == cluster]
        cluster_stats.at[cluster, 'Cluster'] = cluster
        cluster_stats.at[cluster, 'Mode_State'] = cluster_data['st'].mode().iloc[0]  
        cluster_stats.at[cluster, 'Mean_Mag'] = cluster_data['mag'].mean()
        cluster_stats.at[cluster, 'Mean_Loss'] = cluster_data['loss'].mean()
        cluster_stats.at[cluster, 'Mean_Area'] = cluster_data['area'].mean()
        cluster_stats.at[cluster, 'Mean_Cas_Ratio'] = cluster_data['cas_ratio'].mean()

    return cluster_stats


def calculate_ssd(data, centroids, labels):
    ssd = 0
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data_normalized = StandardScaler().fit_transform(data[numerical_columns].values)

    for i in range(len(ata_normalized)):
        ssd += np.sum((data_normalized[i] - centroids[labels[i]])**2)

    return ssd

def process_csv(csv_path):
    df = pd.read_csv(csv_path)

    # preprocessing stuff, drop mag == -9 rows, replace gust NaNs with 0
    na_cols = ["max_gust","min_gust","mean_gust","sd_gust","median_gust"]
    df = df[df.mag != -9]
    df[na_cols] = df[na_cols].fillna(0)

    df["cas"] = df["inj"] + df["fat"]           # casualty target column
                    
    df = encode_states(df)                      # encoding state column such tha it becomes the state tornado rank

    df = encode_loss(df)                        # encode loss column as per NOAA instructions (see function for details)

    df = adjust_dollars(df, ["loss","closs"])   # adjust money for inflation so that it's all in current year dollars
    df["dmg"] = df["loss"] + df["closs"]        # damages target column
    # remove outliers based on user-provided outlier cutoff
    nrow_before = df.shape[0]


    return df

# uncomment this code to run this algorithm by itself as well as the elbow method. 
'''
datapath = '/Users/cobeyweemes/Desktop/CS 5593/Data-Mining-Project/data/tornado_wind_data.csv'

data = process_csv(datapath, 'cas_ratio', None)

exclude_columns = ['date', 'yr', 'mo', 'dy', 'time']
data = data.drop(columns=exclude_columns)

# Apply k-means algorithm
k = 4
centroids, labels = k_means(data, k)

# print(centroids)
ssd = calculate_ssd(data, centroids, labels)
print(f"Sum of Squared Differences: {ssd}")

stats = cluster_statistics(data, labels)
print(stats)


# Elbow method
model = KMeans()
visualizer = KElbowVisualizer(model)

numeric_cols = data.select_dtypes(include=[np.number]).columns
data_numeric = data[numeric_cols]

# Fit the visualizer with numeric data
visualizer.fit(data_numeric)
visualizer.show()
'''
