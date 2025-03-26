import streamlit as st
import sys
import os

def main():
    st.title("FinVet Deployment Diagnostics")
    
    # System Information
    st.header("System Details")
    st.write(f"Python Version: {sys.version}")
    st.write(f"Current Working Directory: {os.getcwd()}")
    
    # Python Path
    st.header("Python Path")
    for path in sys.path:
        st.write(path)
    
    # Project Structure
    st.header("Project Structure")
    try:
        root_contents = os.listdir('.')
        st.write("Root Directory Contents:")
        st.code('\n'.join(root_contents))
        
        if 'src' in root_contents:
            src_contents = os.listdir('src')
            st.write("src Directory Contents:")
            st.code('\n'.join(src_contents))
    except Exception as e:
        st.error(f"Error listing directory: {e}")
    
    # Simple Interaction
    st.header("Basic Interaction Test")
    name = st.text_input("Enter your name:")
    if name:
        st.write(f"Hello, {name}!")

if __name__ == "__main__":
    main()