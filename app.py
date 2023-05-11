import streamlit as st
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

#QDA
def qda(df, X, y):
# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a QuadraticDiscriminantAnalysis object and fit the model on the training data
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    # Make predictions on the test data and calculate the accuracy of the model
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)

    return accuracy


#streamlit starting page

# Set page icon & title
st.set_page_config(page_title="Streamlit Classification & Clustering", page_icon=":guardsman:", layout="wide")

# Add a title
st.title("Upload a csv file")

# Add a file uploader component
uploaded_file = st.file_uploader("Choose a file", type = "csv")

# Add a button to submit the uploaded file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=',', header=None)
    st.write("You selected the following file:")
    st.write(df.head())

# koubi analisis
if st.button("Start Analysis"):
    # Split the data into features and target variable
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    accuracy = qda(df,X,y)
    st.write(f"Accuracy: {accuracy:.2f}")