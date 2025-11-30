import streamlit as st
from config import EXERCISES

def sidebar():
    st.sidebar.title("Settings")
    
    # Exercise selection dropdown
    st.sidebar.subheader("Recovery Exercise")
    selected_exercise = st.sidebar.selectbox(
        "Choose an exercise:",
        options=EXERCISES,
        index=0,
        help="Select a recovery exercise to perform"
    )
    
    # Store selected exercise in session state
    st.session_state.selected_exercise = selected_exercise
    
    st.sidebar.divider()
    st.sidebar.subheader("Camera Settings")
    
    return selected_exercise