# visualize_clusters.py

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load clustered data
data = pd.read_csv("Iris_Clustered.csv")

# Step 2: Calculate centroids for each cluster
centroids = data.groupby('Cluster')[['Annual Income', 'Spending Score']].mean().reset_index()

# Step 3: Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(data['Annual Income'], data['Spending Score'], 
            c=data['Cluster'], cmap='rainbow', s=60)

# Step 4: Plot centroids
plt.scatter(centroids['Annual Income'], centroids['Spending Score'],
            s=200, c='black', marker='X', label='Centroids')

# Step 5: Add labels and title
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments with Centroids")
plt.legend()

# Step 6: Save the plot as PNG instead of showing it
plt.savefig("Clustered_Customers.png")
plt.close()  # close figure to free memory

print("Clustered customer plot saved as 'Clustered_Customers.png'")


