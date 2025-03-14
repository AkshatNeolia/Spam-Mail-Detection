import streamlit as st
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")

# Custom CSS for background and styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stTextArea textarea {
            background-color: #f8f9fa;
            color: #333;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff2020;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("📧 Spam Mail Detector")
st.subheader("🔍 Detect whether an email is **Spam or Ham**")

# Load trained model and vectorizer
try:
    with open('spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# Input text field
user_input = st.text_area("✉️ Type your email message here:", height=150)

# Button to check spam
if st.button("🚀 Check Spam Status"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before checking!")
    else:
        # Process input & predict
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)

        # Display results with custom styling
        if prediction[0] == 1:
            st.success("✅ **Ham (Not Spam)** - Your email is safe!")
        else:
            st.error("🚨 **Spam!** - This email looks suspicious!")

# Footer
st.markdown("---")
st.caption("🔹 **Built with Streamlit** | 💻 **By Akshat Neolia** | 📩 **Spam Detection AI**")
