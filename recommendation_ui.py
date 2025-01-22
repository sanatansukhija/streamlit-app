import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif

# Enable holoviews for chord diagram
hv.extension('bokeh')

# Set the app title
st.title("Facebook Network Explorer and Recommendation System")

# Sidebar instructions
st.sidebar.header("Upload Dataset")
st.sidebar.write("Upload a CSV file containing features for nodes (individuals).")

# File upload for node features
uploaded_file = st.sidebar.file_uploader("Choose a CSV file for features", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Display basic information
    st.subheader("Dataset Overview")
    st.write("Here’s a preview of the uploaded dataset:")
    st.dataframe(df.head())
    
    st.write("Summary statistics of the dataset:")
    st.write(df.describe())
    
    # Extract feature columns (assuming first column is ID or name)
    feature_columns = df.columns[1:]
    
    # --- Feature Distribution ---
    st.subheader("Feature Distributions")
    st.write("View the distributions of different features.")
    
    feature_for_dist = st.selectbox(
        "Select a feature to visualize its distribution",
        feature_columns
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature_for_dist], kde=True, ax=ax, color='cornflowerblue')
    st.pyplot(fig)

    # --- Pairwise Plot ---
    st.subheader("Pairwise Feature Relationships")
    st.write("Select features for pairwise plotting. If no features are selected, we’ll randomly choose 3 features.")
    
    # Allow user to modify the selected features
    selected_plot_features = st.multiselect(
        "Select features for pairwise plotting (up to 10 features recommended)",
        feature_columns,
        default=[],
        max_selections=10
    )
    
    # If no features are selected, pick 3 random features
    if len(selected_plot_features) == 0:
        selected_plot_features = random.sample(list(feature_columns), 3)
    
    if len(selected_plot_features) > 1:
        fig = px.scatter_matrix(df, dimensions=selected_plot_features, title="Pairplot of Selected Features", 
                                color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig)
    else:
        st.write("Please select at least 2 features to display the pairwise plot.")

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    st.write("This heatmap shows correlations between the first 20 features, with annotations.")
    
    # Limit to the first 20 features (or fewer if the dataset has fewer features)
    heatmap_features = feature_columns[:min(20, len(feature_columns))]
    
    # Calculate the correlation matrix for the selected features
    correlation_matrix = df[heatmap_features].corr()
    
    # Plot the heatmap using Seaborn with annotations
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', cbar=True, ax=ax, annot_kws={'size': 10}, linewidths=0.5)
    st.pyplot(fig)

    # --- Clustering: K-Means ---
    st.subheader("User Clustering")
    st.write("We will apply K-means clustering to group users based on their features. You can adjust the number of clusters.")

    num_clusters = st.slider("Select the number of clusters", 2, 10, 3)
    
    # Standardize the feature matrix to normalize the scales
    features_matrix = df[feature_columns]
    scaler = StandardScaler()
    features_matrix_scaled = scaler.fit_transform(features_matrix)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_matrix_scaled)
    
    # --- Dimensionality Reduction for 2D and 3D Visualization ---
    st.subheader("Dimensionality Reduction for Clustering Visualization")
    st.write("Using PCA or t-SNE to reduce the feature space to 2D or 3D for visualizing clusters.")
    
    view_option = st.radio("Choose visualization view", ("2D View", "3D View"))
    reduction_method = st.radio("Choose dimensionality reduction method", ("PCA", "t-SNE"))
    
    if reduction_method == "PCA":
        if view_option == "3D View":
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(features_matrix_scaled)
        else:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(features_matrix_scaled)
    else:
        if view_option == "3D View":
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(features_matrix_scaled)
        else:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(features_matrix_scaled)
    
    # Create a DataFrame for the projection
    if view_option == "3D View":
        reduced_df = pd.DataFrame(reduced_data, columns=["Dim1", "Dim2", "Dim3"])
    else:
        reduced_df = pd.DataFrame(reduced_data, columns=["Dim1", "Dim2"])

    reduced_df['Cluster'] = df['Cluster']
    reduced_df['User'] = df.iloc[:, 0]  # Include user ID for hover data
    
    # Plot the clusters interactively using Plotly
    if view_option == "3D View":
        cluster_colors = sns.color_palette("husl", n_colors=num_clusters)  # Using a diverse color palette for 3D visualization
        fig = px.scatter_3d(
            reduced_df, x="Dim1", y="Dim2", z="Dim3", color="Cluster", title="3D Clustering Visualization", 
            hover_data=["User", "Cluster"], color_discrete_sequence=cluster_colors
        )
    else:
        cluster_colors = sns.color_palette("husl", n_colors=num_clusters)  # Using a diverse color palette for 2D visualization
        fig = px.scatter(
            reduced_df, x="Dim1", y="Dim2", color="Cluster", title="2D Clustering Visualization", 
            hover_data=["User", "Cluster"], color_discrete_sequence=cluster_colors
        )
    
    st.plotly_chart(fig)

    # --- User Profile Visualizations ---
    st.subheader("User Profile Details")
    st.write("Click on the visualization points to see detailed user profiles.")
    
    # User selection for more information
    selected_user = st.selectbox("Select a user for detailed view", df.iloc[:, 0].tolist())
    user_details = df[df.iloc[:, 0] == selected_user].drop(columns=[df.columns[0]])  # Remove ID column
    
    st.write(f"User Profile for {selected_user}:")
    st.dataframe(user_details)

    # --- Feature Importance Based on Clustering ---
    st.subheader("Feature Importance Based on Clustering")
    st.write("View the importance of each feature in determining cluster assignments.")
    
    # Use feature selection (ANOVA F-test) to rank features
    selector = SelectKBest(f_classif, k='all')
    selector.fit(features_matrix_scaled, df['Cluster'])
    
    feature_scores = pd.DataFrame({
        "Feature": feature_columns,
        "Score": selector.scores_
    })
    
    # Sort the features by score and only take the top 10
    feature_scores_sorted = feature_scores.sort_values(by="Score", ascending=False).head(10)
    
    # Plot the feature importance (top 10 features)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Score", y="Feature", data=feature_scores_sorted, ax=ax, palette="coolwarm")
    st.pyplot(fig)

    # --- Find Similar Users Based on Features ---
    st.subheader("Find Similar Users Based on Features")
    
    # Allow the user to select a user from the list
    users = df.iloc[:, 0].tolist()  # Assuming the first column is the user ID
    selected_user = st.selectbox("Select a user", users)
    
    # Find the index of the selected user
    selected_user_idx = df[df.iloc[:, 0] == selected_user].index[0]
    
    # Compute pairwise Euclidean distances
    distances = euclidean_distances(features_matrix_scaled)
    
    # Get distances for the selected user and create a DataFrame for easy viewing
    selected_user_distances = distances[selected_user_idx]
    
    # Create a DataFrame of users and their distances from the selected user
    distance_df = pd.DataFrame({
        "User": df.iloc[:, 0],
        "Distance": selected_user_distances,
        "Cluster": df['Cluster']
    })
    
    # Remove the selected user from the results
    distance_df = distance_df[distance_df['User'] != selected_user]
    
    # Sort the DataFrame by distance (ascending)
    distance_df_sorted = distance_df.sort_values(by="Distance")
    
    # Allow the user to filter the number of similar users to show
    num_similar_users = st.slider("Select number of similar users to show", min_value=1, max_value=20, value=5)
    
    # Show the top N similar users within the distance threshold
    top_similar_users = distance_df_sorted.head(num_similar_users)
    
    # Display the top N similar users
    st.write(f"Top {num_similar_users} similar users to {selected_user}:")
    st.dataframe(top_similar_users.style.background_gradient(cmap='Blues'))  # Adding background color gradient to make it more visually appealing.

else:
    st.write("Please upload a CSV file to get started.")

