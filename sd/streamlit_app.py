import streamlit as st
from demo import methodName

# Revised CSS to adjust logo size, center elements, and set the text area background
CSS = """
html,body {
background: url('https://i.imgur.com/xDeaqpt.jpg') no-repeat center center fixed;    color: white !important;
background-size: cover;
}


.streamlit-container, .main, .block-container, .stApp {
    background-color: transparent !important;
}


div.stApp {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    flex-direction: column;
}
div.block-container {
    padding-top: 20vh;  # adjust this to set the vertical starting position of your logo
    width: 100%;
}
img {
    width: 60%;  # adjusted size of your logo to be three times larger
    margin-bottom: 2rem;  # space between logo and the first input element
    display: block;
    margin-left: auto;
    margin-right: auto;
}
/* Text area and input field adjustments */
.stTextInput input, .stTextArea textarea {
    background-color: white !important;
    color: black !important;
}

/* Button customization */
.stButton > button {
    margin: 20px auto;
    background-color: #4CAF50 !important;
    color: white !important;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    line-height: 20px;
    border-radius: 4px;
    box-shadow: none !important;
    text-shadow: none !important;
}

/* Fixing the width of the button to auto */
button {
    width: auto !important;
    display: block !important;
}

/* Adjusting placeholder color specifically */
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: white !important;
    opacity: 1; /* Ensures the placeholder is fully visible */
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: white !important;
}

/* Additional CSS for a larger textarea */
.stTextArea textarea {
    height: 300px !important; /* This height can be adjusted as needed */
}

label {
    color: white !important; /* This will change all labels to white */
    font-size: 90px;
}

/* Direct tag targeting if under a specific class container */
.stApp h2, .stMarkdown span {
    color: white !important; /* Sets color to white */
    font-size: 24px; /* Optional: Adjust font size if needed */
    margin-bottom: 0px !important;
}

.stMarkdown p {
   color: red !important;
}

"""

# Inject custom CSS
st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)

# Display logo
st.image('color-transperant-logo.png', use_column_width=False)

# Input form
with st.form(key='input_form'):
    st.subheader('Enter your prompt')
    prompt_text = st.text_area("promptTextArea", key='prompt', label_visibility="hidden")
    st.subheader('Enter URL:')
    url_text = st.text_input("urlTextInput", key='url', label_visibility="hidden")

    submitted = st.form_submit_button("Submit")


# Outside the form, check if it was submitted
if submitted:
    if prompt_text and url_text:
        methodName(prompt_text,url_text)
        print("I COMPLETED")
        st.image('combined_image.png', caption='Your Image', use_column_width=True)
    else:
        st.write("Please fill in both fields to enable the submission.")
