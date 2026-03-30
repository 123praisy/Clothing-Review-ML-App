import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Clothing Review Recommendation Predictor")

st.write("Enter a product review to check whether it is recommended or not.")

# Text input
review = st.text_area("Enter your review:")

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Predict button
if st.button("Predict"):
    
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        # Clean and transform input
        cleaned_review = clean_text(review)
        review_vec = vectorizer.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_vec)

        # Output
        if prediction[0] == 1:
            st.success("Recommended ✅")
        else:
            st.error("Not Recommended ❌")