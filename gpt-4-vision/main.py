import streamlit as st
from dotenv import load_dotenv
from utils import describe_image

# Load environment variables
load_dotenv()


# Streamlit App
st.title("🖼️ - GPT-4 Vision")  # Add a title
st.divider()  # Add a divider   

# Custom style for blue button
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: transparent;
            border: 1px solid #3498db;
            float: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input
with st.form("user_form", clear_on_submit=True):
    text_input = st.text_input("Enter image url")
    submit_button = st.form_submit_button(label="Submit")

# Process the uploaded file
if submit_button and text_input is not None:
    with st.spinner("Generating..."):
        description = describe_image(text_input)
        # text in blue color
        st.markdown(f'<p style="color:lightblue;font-size:1.2rem">{description}</p>', unsafe_allow_html=True)
    