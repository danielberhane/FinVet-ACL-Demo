import streamlit as st

# Configure the page
st.set_page_config(
    page_title="FinVet | Financial Misinfo Detector",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Display a simple message
st.title("FinVet: Financial Misinformation Detector")
st.write("This is a minimal version of the app to test deployment.")

# Basic UI elements
st.markdown("## Enter a claim to verify")
claim = st.text_area(
    "Enter financial claim:",
    placeholder="Example: Tesla's stock price doubled in 2023"
)

if st.button("Verify Claim"):
    st.info("This is a test version. Verification is not implemented yet.")
    
# Display version
st.markdown("---")
st.write("FinVet v0.1.4 - Minimal Test Version")