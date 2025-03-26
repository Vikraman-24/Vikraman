import pandas as pd
import zipfile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
zip_path = r"c:\Users\balaj\Downloads\alexa.com_site_info.csv.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('alexa.com_site_info.csv') as f:
        df = pd.read_csv(f)

# Select relevant columns for clustering
columns = ["This_site_rank_in_global_internet_engagement", "Daily_time_on_site"]
df_filtered = df[columns].dropna()

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)

# Determine optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply K-Means clustering with optimal K (e.g., 3 for illustration)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered['Cluster'] = kmeans.fit_predict(df_scaled)

# Show cluster distribution
print(df_filtered.groupby('Cluster').mean())

# Save results
df_filtered.to_csv("university_clusters.csv", index=False)
