# iris_kmeans.py

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Load Iris dataset and rename features for customer segmentation
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Annual Income', 'Spending Score', 'Feature3', 'Feature4'])

# Step 2: Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Elbow Method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Save Elbow plot as PNG
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("ElbowPlot.png")  # saves the plot
plt.close()  # close figure

# Step 4: Apply KMeans (choose k=3 for Iris)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 5: Add cluster labels to dataframe
X['Cluster'] = y_kmeans

# Step 6: Plot clusters (first two features renamed) and save as PNG
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_kmeans, cmap='rainbow', s=60)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', marker='X', edgecolor='black', label='Centroids')
plt.title("K-means Clustering (Annual Income vs Spending Score)")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.savefig("ClusterPlot.png")
plt.close()  # close figure

# Step 7: Save clustered data to CSV
X.to_csv("Iris_Clustered.csv", index=False, encoding='utf-8-sig')
print("Clustering complete! Results saved as 'Iris_Clustered.csv'")
print("Elbow plot saved as 'ElbowPlot.png'")
print("Cluster plot saved as 'ClusterPlot.png'")



