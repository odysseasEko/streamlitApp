import streamlit as st
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score


def load_data(file):
    data = pd.read_csv(file)
    return data

def run_qda(data):
    # Split the data into features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Fit the QDA model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    # Make predictions on the test data
    y_pred = qda.predict(X)

    # Calculate the accuracy
    accuracy = accuracy_score(y, y_pred)

    # Calculate the cluster purity index
    cluster_purity = calculate_purity(X, y_pred)                                                                     

    return accuracy, cluster_purity

def run_spectral_clustering(data, n_clusters):
    # Split the data into features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Fit the Spectral Clustering model
    sc = SpectralClustering(n_clusters=n_clusters)
    sc.fit(X)

    # Make predictions on the test data
    y_pred = sc.labels_

    # Calculate the cluster purity index
    cluster_purity = calculate_purity(X, y_pred)

    return cluster_purity

def calculate_purity(data, y_pred):
    # Get the unique cluster labels
    unique_labels = np.unique(y_pred)

    # Calculate the cluster purity index
    purity_index = 0
    for label in unique_labels:
        indices = np.where(y_pred == label)[0]
        class_counts = np.bincount(data[indices, -1].astype(int))
        majority_class = np.argmax(class_counts)
        majority_count = class_counts[majority_class]
        purity_index += majority_count / indices.shape[0]

    return purity_index

# Create the Streamlit app
st.title("Machine Learning App")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Load the data when the file is uploaded
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Run the QDA model
    if st.button("Run QDA"):
        accuracy, purity = run_qda(data)
        st.write(f"QDA Accuracy: {accuracy:.2f}")
        st.write(f"QDA Purity Index: {purity:.2f}")

    # Run the Spectral Clustering model
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=2, step=1)
    if st.button("Run Spectral Clustering"):
        purity = run_spectral_clustering(data, n_clusters)
        st.write(f"Spectral Clustering Purity Index: {purity:.2f}")
