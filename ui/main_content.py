import streamlit as st
from config import EXERCISE_INSTRUCTIONS

def main_content(selected_exercise):
    st.title("Posture Perfect")
    
    # Display selected exercise in main area
    if selected_exercise and selected_exercise != "Select an exercise...":
        st.info(f"📋 **Current Exercise:** {selected_exercise}")
        
        # Show exercise instructions
        with st.expander("📖 How to perform this exercise", expanded=True):
            if selected_exercise in EXERCISE_INSTRUCTIONS:
                st.write(EXERCISE_INSTRUCTIONS[selected_exercise])
            else:
                st.write("Instructions for this exercise are being updated.")
    else:
        st.warning("⚠️ Please select an exercise from the sidebar to begin.")
    
    st.write("Live camera feed and FPS display.")