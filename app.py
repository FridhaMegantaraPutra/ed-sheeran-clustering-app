import streamlit as st
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Title of the app
st.title('K-means Clustering with PCA Visualization')
st.write('This app performs K-means clustering on tweets about Ed Sheeran.')
st.image('ed.png')

# Define model files based on number of clusters
model_files = {
    2: 'models2.pkl',
    3: 'models3.pkl',
    4: 'models4.pkl',
    5: 'models5.pkl'
}

# Sidebar for user inputs
st.header('User Input Parameters')
num_clusters = st.slider(
    'Select number of clusters:', min_value=2, max_value=5, value=3, step=1)
# Load model and components from the selected .pkl file
selected_model_file = model_files[num_clusters]
with open(selected_model_file, 'rb') as f:
    kmeans, pca, df, pca_components = pickle.load(f)

# Apply KMeans with the selected number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(pca_components)

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(num_clusters):
    indices = df['cluster'] == cluster
    ax.scatter(pca_components[indices, 0], pca_components[indices, 1],
               pca_components[indices, 2], label=f'Cluster {cluster}', alpha=0.5)  # Set alpha value for transparency

ax.set_title('3D Scatter Plot of K-means Clusters')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)

# Cluster Results
st.header('Cluster Results')
st.write(f"Number of clusters selected: {num_clusters}")
st.write(df[['full_text', 'cluster']])  # Show dataframe

# Add footer or additional information
st.markdown('---')
st.write('Fridha Megantara Putra')
