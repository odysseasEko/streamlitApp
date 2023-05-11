# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to run k-means clustering and calculate the silhouette score
def run_kmeans(data, k):
    kmeans = KMeans(n_clusters=k)  # Create a KMeans instance with the specified number of clusters
    kmeans.fit(data)               # Fit the KMeans model to the data
    labels = kmeans.labels_        # Get the cluster labels for each data point
    score = silhouette_score(data, labels)  # Calculate the silhouette score for the clustering
    return labels, score

# Create a Streamlit title
st.title("the two pipes")

# Create a file uploader widget for uploading a tab-separated TXT file
uploaded_file = st.file_uploader("Upload a comma seperated TXT file (no header)", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file as a DataFrame using pandas, assuming no header
    data = pd.read_csv(uploaded_file, sep=',', header=None)
    # Display a preview of the data
    st.write("Data preview:")
    st.write(data.head())

    # Create a number input widget for specifying the number of clusters for k-means clustering
    n_clusters = st.number_input("Enter the number of clusters for KMeans:", min_value=2, value=2)

    # Create a button to start the analysis
    if st.button("Start Analysis"):
        # Separate the features (K-1 columns) and the target (last column) from the data
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]

        # Run k-means clustering and decision tree classification on the data
        kmeans_labels, kmeans_score = run_kmeans(features, n_clusters)
        

        # Display the evaluation results in a table
        st.write("Evaluation Results:")
        st.write(pd.DataFrame({"Method": ["KMeans (Silhouette Score)"],
                               "Score": [kmeans_score]}))