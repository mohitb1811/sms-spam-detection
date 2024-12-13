import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon=":warning:",
    layout="centered",  # Center the layout
    initial_sidebar_state="expanded"
)

# Add custom CSS to center content, adjust header placement, and style button
st.write("""
    <style>
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 50px 0;
        }
        .stHeader {
            color: #1e3a8a; /* dark blue text */
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px; /* Space between header and input box */
            text-align: center;
        }
        .stText {
            color: #4b5563; /* gray text */
            font-size: 16px;
            text-align: center;
        }
        .stButton button {
            background-color: #ef4444 !important; /* red button */
            color: #ffffff !important;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Define function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        i = re.sub(r'[^a-zA-Z0-9]', '', i)
        if i:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Page header
st.markdown("<h1 class='stHeader'>SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Input section
input_sms = st.text_area("Enter the message", "")
if st.button('Predict'):
    if not input_sms.strip():  # Check if the input is empty or only whitespace
        st.warning("**Please provide a message.**")  # Display warning message
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorizer
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.error("⚠️ SPAM ⚠️")
        else:
            st.success("NOT SPAM")

