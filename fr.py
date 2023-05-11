import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a random forest classifier
rf = RandomForestClassifier()
rf.fit(X, y)

# Display the feature importances
st.write("Feature importances:")
for i in range(len(iris.feature_names)):
    st.write(f"{iris.feature_names[i]}: {rf.feature_importances_[i]}")

# Display a prediction form
st.write("")
st.write("Make a prediction:")
sepal_length = st.slider("Sepal length", 0.0, 10.0, 5.0, 0.1)
sepal_width = st.slider("Sepal width", 0.0, 10.0, 5.0, 0.1)
petal_length = st.slider("Petal length", 0.0, 10.0, 5.0, 0.1)
petal_width = st.slider("Petal width", 0.0, 10.0, 5.0, 0.1)
prediction = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write(f"Prediction: {iris.target_names[prediction[0]]}")
