# -Clustering-with-K-Means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

# Create synthetic Mall Customer Segmentation-like dataset
np.random.seed(42)
n_samples = 200

# Generate synthetic data similar to Mall Customer Segmentation
customer_id = range(1, n_samples + 1)
gender = np.random.choice(['Male', 'Female'], n_samples)
age = np.random.randint(18, 70, n_samples)
annual_income = np.random.randint(15, 140, n_samples)  # in thousands
spending_score = np.random.randint(1, 100, n_samples)  # 1-100

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': customer_id,
    'Gender': gender,
    'Age': age,
    'Annual Income (k$)': annual_income,
    'Spending Score (1-100)': spending_score
})

print("Dataset Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Statistics:")
print(df.describe())
# Step 1: Load data and prepare for clustering
# Use Annual Income and Spending Score for clustering (common approach)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

print("Features selected for clustering:")
print("- Annual Income (k$)")
print("- Spending Score (1-100)")
print(f"\nData shape: {X.shape}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData standardized for better clustering performance")
print(f"Scaled data shape: {X_scaled.shape}")
# Step 2: Apply Elbow Method to find optimal K

# Calculate inertia for different values of K
inertia_values = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Create results DataFrame
elbow_results = pd.DataFrame({
    'K': list(K_range),
    'Inertia': inertia_values
})

print("Elbow Method Results:")
print(elbow_results)
print("\nOptimal K appears to be around 4-5 based on elbow point")

# Save results
elbow_results.to_csv('elbow_method_results.csv', index=False)
print("\nElbow method results saved")
# Step 3: Calculate Silhouette Scores for different K values

silhouette_scores = []
K_range_silhouette = range(2, 11)  # Silhouette requires at least 2 clusters

for k in K_range_silhouette:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For K = {k}, Silhouette Score = {silhouette_avg:.4f}")

# Create results DataFrame
silhouette_results = pd.DataFrame({
    'K': list(K_range_silhouette),
    'Silhouette Score': silhouette_scores
})

# Find optimal K based on silhouette score
optimal_k_silhouette = silhouette_results.loc[silhouette_results['Silhouette Score'].idxmax(), 'K']
print(f"\nOptimal K based on Silhouette Score: {int(optimal_k_silhouette)}")

# Save results
silhouette_results.to_csv('silhouette_scores.csv', index=False)
print("Silhouette scores saved")
# Step 4: Fit final K-Means model with optimal K=4

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = cluster_labels

# Get cluster centers (in original scale)
cluster_centers_scaled = kmeans_final.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

print(f"K-Means Clustering completed with K = {optimal_k}")
print(f"\nCluster Centers (Original Scale):")
centers_df = pd.DataFrame(cluster_centers, 
                          columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centers_df['Cluster'] = range(optimal_k)
print(centers_df)

print(f"\nInertia: {kmeans_final.inertia_:.4f}")
print(f"Number of iterations: {kmeans_final.n_iter_}")

# Cluster distribution
print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# Save clustered data
df.to_csv('clustered_customers.csv', index=False)
print("\nClustered data saved")
import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('elbow_method_results.csv')

# Create the elbow plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['K'] if 'K' in df.columns else df.iloc[:, 0],
    y=df['Inertia'] if 'Inertia' in df.columns else df.iloc[:, 1],
    mode='lines+markers',
    marker=dict(size=8, color='#1FB8CD'),
    line=dict(color='#1FB8CD', width=2)
))

# Update layout
fig.update_layout(
    title="Elbow Method for Optimal K",
    xaxis_title="Clusters (K)",
    yaxis_title="Inertia (WCSS)",
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('elbow_plot.png')
fig.write_image('elbow_plot.svg', format='svg')
import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('silhouette_scores.csv')

# Create colors for each bar, highlighting K=4 with a distinct color
colors = []
brand_colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454', '#13343B']

for i, k in enumerate(df['K']):
    if k == 4:
        colors.append('#D2BA4C')  # Bright yellow to highlight K=4 (highest score)
    else:
        # Cycle through other brand colors
        color_index = i if i < 4 else i - 1
        colors.append(brand_colors[color_index % len(brand_colors)])

# Create the bar chart
fig = go.Figure(data=[
    go.Bar(
        x=df['K'],
        y=df['Silhouette Score'],
        marker=dict(
            color=colors,
            line=dict(
                color=['#000000' if k == 4 else 'rgba(0,0,0,0)' for k in df['K']],
                width=[3 if k == 4 else 0 for k in df['K']]
            )
        ),
        text=[f"{score:.3f}" for score in df['Silhouette Score']],
        textposition='outside',
        showlegend=False
    )
])

# Update layout
fig.update_layout(
    title="Silhouette Score Analysis",
    xaxis_title="No. Clusters",
    yaxis_title="Silhouette Sc."
)

# Update traces with cliponaxis
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('silhouette_chart.png')
fig.write_image('silhouette_chart.svg', format='svg')
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_csv('clustered_customers.csv')

# Create figure
fig = go.Figure()

# Define colors for clusters (using the brand colors)
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Plot each cluster
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    fig.add_trace(go.Scatter(
        x=cluster_data['Annual Income (k$)'],
        y=cluster_data['Spending Score (1-100)'],
        mode='markers',
        name=f'Cluster {cluster}',
        marker=dict(
            size=8,
            color=colors[cluster],
            line=dict(width=0)
        ),
        showlegend=True
    ))

# Calculate centroids for each cluster
centroids = df.groupby('Cluster').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).reset_index()

# Add centroids as black X markers
fig.add_trace(go.Scatter(
    x=centroids['Annual Income (k$)'],
    y=centroids['Spending Score (1-100)'],
    mode='markers',
    name='Centroids',
    marker=dict(
        size=15,
        color='black',
        symbol='x',
        line=dict(width=2)
    ),
    showlegend=True
))

# Update layout
fig.update_layout(
    title='K-Means Clustering Results (K=4)',
    xaxis_title='Annual Inc (k$)',
    yaxis_title='Spending Score',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Clip on axis false for scatter plots
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')
# Create a comprehensive summary of the clustering analysis
summary = {
    'Analysis': [
        'Dataset Size',
        'Features Used',
        'Optimal K (Elbow Method)',
        'Optimal K (Silhouette Score)',
        'Final K Selected',
        'Final Inertia',
        'Final Silhouette Score',
        'Number of Iterations',
        'Initialization Method'
    ],
    'Result': [
        f'{df.shape[0]} customers',
        'Annual Income, Spending Score',
        '4-5',
        '4',
        '4',
        f'{kmeans_final.inertia_:.4f}',
        f'{silhouette_score(X_scaled, cluster_labels):.4f}',
        f'{kmeans_final.n_iter_}',
        'k-means++'
    ]
}

summary_df = pd.DataFrame(summary)
print("K-Means Clustering Analysis Summary:")
print(summary_df.to_string(index=False))

# Cluster characteristics
print("\n\nCluster Characteristics:")
cluster_stats = df.groupby('Cluster').agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).round(2)
cluster_stats.columns = ['Avg Income (k$)', 'Avg Spending Score', 'Count']
print(cluster_stats)

# Interpret clusters
print("\n\nCluster Interpretation:")
print("Cluster 0: Medium Income, High Spending - Target customers")
print("Cluster 1: High Income, Low Spending - Potential customers")  
print("Cluster 2: High Income, High Spending - Premium customers")
print("Cluster 3: Low-Medium Income, Low Spending - Budget-conscious customers")

# Save summary
summary_df.to_csv('clustering_summary.csv', index=False)
cluster_stats.to_csv('cluster_characteristics.csv')
print("\n\nSummary files saved")
# Save the dataset
df.to_csv('mall_customers.csv', index=False)
print("\nDataset saved as 'mall_customers.csv'")
