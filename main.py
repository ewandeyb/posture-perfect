import streamlit as st
import cv2
import time
import pandas as pd
from exercises.exercise_factory import ExerciseFactory
from pose_detector import PoseDetector

# 1. Add some CSS to reduce whitespace and stabilize the layout
def local_css():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stMetric { background-color: #1e293b; padding: 10px; border-radius: 5px; text-align: center; color: white;}
        .stMetric label { color: #cbd5e1 !important; }
        .stMetric [data-testid="stMetricValue"] { color: white !important; }
        .stMetric [data-testid="stMetricDelta"] { color: #94a3b8 !important; }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Posture Perfect", layout="wide")
    local_css() # Apply CSS
    
    st.title("Posture Perfect")
    
    # --- Sidebar (kept mostly the same) ---
    with st.sidebar:
        st.header("Exercise Selection")
        categories = ExerciseFactory.get_exercises_by_category()
        
        category = st.selectbox("Category", options=list(categories.keys()), format_func=lambda x: x.capitalize())
        exercise_ids = categories[category]
        exercise_names = {eid: ExerciseFactory.create(eid).get_name() for eid in exercise_ids}
        
        selected_id = st.selectbox("Exercise", options=exercise_ids, format_func=lambda x: exercise_names[x])
        
        # Initialize exercise state
        if 'current_exercise' not in st.session_state or st.session_state.get('exercise_id') != selected_id:
            st.session_state.current_exercise = ExerciseFactory.create(selected_id)
            st.session_state.exercise_id = selected_id
        
        exercise = st.session_state.current_exercise
        
        st.divider()
        st.subheader("📋 Instructions")
        for i, instruction in enumerate(exercise.get_instructions(), 1):
            st.write(f"{i}. {instruction}")
            
        if st.button("🔄 Reset Exercise", use_container_width=True):
            exercise.reset()
            st.rerun()

    # --- Main Dashboard Layout ---
    # We allocate more space to video (3 parts) and use a distinct 'Dashboard' column (1 part)
    col_video, col_stats = st.columns([3, 1.2])
    
    with col_video:
        st.subheader("📹 Live Feed")
        # Define a fixed aspect ratio or height to prevent video jumping
        video_placeholder = st.empty()

    with col_stats:
        st.subheader("📊 Performance")
        
        # 2. STABILITY FIX: Define placeholders in a specific order.
        # Score is at the top (fixed height) so it never jumps.
        score_container = st.container()
        st.divider() 
        
        # Metrics in the middle
        metrics_container = st.container()
        
        # Variable feedback at the bottom
        feedback_container = st.container()

    # Control buttons (Placed below video for cleaner look)
    with col_video:
        col_start, col_stop = st.columns(2)
        start = col_start.button("▶️ Start", use_container_width=True, type="primary")
        stop = col_stop.button("⏹️ Stop", use_container_width=True)

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False
        exercise.reset()

    if st.session_state.get('running', False):
        run_session(
            exercise,
            video_placeholder,
            score_container,
            metrics_container,
            feedback_container
        )

def run_session(exercise, video_ph, score_container, metrics_container, feedback_container):
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    # 3. THROTTLING LOGIC: Initialize timer
    last_ui_update_time = time.time()
    UI_UPDATE_INTERVAL = 0.3  # Update text only every 0.3 seconds

    # Create specific elements inside the containers ONCE
    with score_container:
        score_metric = st.empty()
    with metrics_container:
        angle_chart = st.empty()
    with feedback_container:
        feedback_alert = st.empty()

    stframe = video_ph.empty()
    
    while st.session_state.get('running', False):
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detector.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = detector.extract_landmarks(results, exercise.get_required_landmarks())
            metrics = exercise.calculate_metrics(landmarks)
            validation = exercise.validate_form(metrics, frame_count)
            
            # Draw Skeleton (Happens every frame for smoothness)
            frame_rgb = detector.draw_landmarks(frame_rgb, results, exercise.get_visualization_points())
            frame_rgb = draw_overlay_feedback(frame_rgb, validation)

            # Update UI (Throttled for readability)
            current_time = time.time()
            if current_time - last_ui_update_time > UI_UPDATE_INTERVAL:
                update_dashboard(validation, score_metric, angle_chart, feedback_alert)
                last_ui_update_time = current_time
        
        # Video updates every frame (30fps)
        stframe.image(frame_rgb, channels="RGB")
        frame_count += 1
    
    cap.release()

def update_dashboard(validation, score_metric, angle_chart, feedback_alert):
    """Updates the side dashboard. Separated for cleanliness."""
    
    # 1. Update Score
    score = validation.score
    delta_color = "normal"
    if score >= 80: delta_color = "normal" # Greenish in Streamlit
    elif score < 50: delta_color = "inverse" # Reddish
    
    score_metric.metric(
        label="Form Score", 
        value=f"{score:.0f}/100", 
        delta=f"{'Keep it up!' if score > 80 else 'Check form'}",
        delta_color=delta_color
    )

    # 2. Update Angles (Using a small dataframe is cleaner than text lists)
    if validation.angles:
        df = pd.DataFrame(list(validation.angles.items()), columns=["Metric", "Angle"])
        # Format the angle to 1 decimal place
        df['Angle'] = df['Angle'].apply(lambda x: f"{x:.1f}°")
        angle_chart.dataframe(df, hide_index=True, use_container_width=True)

    # 3. Update Feedback Messages
    if validation.feedback_messages:
        # Join messages with bullets
        msg_text = "\n".join([f"• {msg}" for msg in validation.feedback_messages])
        if validation.is_correct:
            feedback_alert.success(msg_text, icon="✅")
        else:
            feedback_alert.warning(msg_text, icon="⚠️")
    else:
        feedback_alert.info("Perfect form!", icon="✨")

def draw_overlay_feedback(frame, validation):
    """Draw simple status on video frame only"""
    # Keep this minimal so the video isn't cluttered
    color = (0, 255, 0) if validation.is_correct else (255, 0, 0)
    
    # Simple top bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 10), color, -1)
    
    return frame

if __name__ == "__main__":
    main()