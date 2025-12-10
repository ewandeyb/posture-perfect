import streamlit as st
import cv2
import time
import pandas as pd
from urllib.parse import urlparse, parse_qs
from exercises.exercise_factory import ExerciseFactory
from pose_detector import PoseDetector


def _parse_youtube(video_url: str) -> tuple[str | None, int | None]:
    """
    Extract YouTube video ID and optional start time (seconds) from URL.
    Supports:
      - https://www.youtube.com/watch?v=ID&t=156
      - https://youtu.be/ID?t=156
      - start= / t= with h/m/s (e.g., 2m36s)
    """
    parsed = urlparse(video_url)
    query = parse_qs(parsed.query)

    # Video ID
    video_id = None
    if "youtube.com" in parsed.netloc:
        video_id = query.get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/").split("/")[0] or None

    # Start time
    def _parse_time(value: str) -> int | None:
        if value.isdigit():
            return int(value)
        total = 0
        num = ""
        for ch in value:
            if ch.isdigit():
                num += ch
            else:
                if not num:
                    continue
                if ch == "h":
                    total += int(num) * 3600
                elif ch == "m":
                    total += int(num) * 60
                elif ch == "s":
                    total += int(num)
                num = ""
        if num:
            total += int(num)
        return total if total > 0 else None

    start = None
    if "t" in query:
        start = _parse_time(query["t"][0])
    elif "start" in query:
        start = _parse_time(query["start"][0])

    return video_id, start

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
        
        # Rep counter widget
        rep_container = st.container()
        
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

        # Demonstration video below controls (collapsible) to keep top aligned with metrics
        video_url = exercise.get_video_url()
        if video_url:
            with st.expander("🎥 Demonstration video", expanded=False):
                if "youtube.com" in video_url or "youtu.be" in video_url:
                    video_id, start_time = _parse_youtube(video_url)
                    if video_id:
                        params = f"?start={start_time}" if start_time else ""
                        youtube_embed = f"https://www.youtube.com/embed/{video_id}{params}"
                        st.markdown(
                            f'<iframe width="100%" height="230" src="{youtube_embed}" '
                            'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                            'gyroscope; picture-in-picture" allowfullscreen></iframe>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"[▶️ Watch Video: {video_url}]({video_url})")
                else:
                    try:
                        st.video(video_url, format="video/mp4")
                    except:
                        st.markdown(f"[▶️ Watch Video: {video_url}]({video_url})")
                st.markdown(f"🔗 [Open in new tab]({video_url})")

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
            rep_container,
            metrics_container,
            feedback_container
        )

def run_session(exercise, video_ph, score_container, rep_container, metrics_container, feedback_container):
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    # 3. THROTTLING LOGIC: Initialize timer
    last_ui_update_time = time.time()
    UI_UPDATE_INTERVAL = 0.3  # Update text only every 0.3 seconds

    # Create specific elements inside the containers ONCE
    with score_container:
        score_metric = st.empty()
    with rep_container:
        rep_metric = st.empty()
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
            # Check visibility of required landmarks
            required_landmarks = exercise.get_required_landmarks()
            all_visible, missing_landmarks = detector.check_visibility(
                results, 
                required_landmarks,
                min_visibility=0.5
            )
            
            # Only extract landmarks that are visible
            landmarks = detector.extract_landmarks(results, required_landmarks)
            
            # If not all visible, add missing info to landmarks dict
            if not all_visible:
                landmarks['_missing_points'] = missing_landmarks
                landmarks['_all_visible'] = False
            else:
                landmarks['_all_visible'] = True
            
            metrics = exercise.calculate_metrics(landmarks)
            validation = exercise.validate_form(metrics, frame_count)
            
            # Draw Skeleton (Happens every frame for smoothness)
            frame_rgb = detector.draw_landmarks(frame_rgb, results, exercise.get_visualization_points())
            frame_rgb = draw_overlay_feedback(frame_rgb, validation, exercise)

            # Update UI (Throttled for readability)
            current_time = time.time()
            if current_time - last_ui_update_time > UI_UPDATE_INTERVAL:
                update_dashboard(validation, score_metric, rep_metric, angle_chart, feedback_alert)
                last_ui_update_time = current_time
        
        # Video updates every frame (30fps)
        stframe.image(frame_rgb, channels="RGB")
        frame_count += 1
    
    cap.release()

def update_dashboard(validation, score_metric, rep_metric, angle_chart, feedback_alert):
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
    
    # 2. Update Rep Counter
    details = validation.details
    total_reps = details.get('reps', 0)
    left_reps = details.get('left_reps', 0)
    right_reps = details.get('right_reps', 0)
    
    # Show rep count with breakdown if available
    if left_reps > 0 or right_reps > 0:
        rep_text = f"L: {left_reps} | R: {right_reps}"
        delta_text = f"Total: {total_reps}"
    else:
        rep_text = f"{total_reps}"
        delta_text = "Keep going!"
    
    rep_metric.metric(
        label="Reps",
        value=rep_text,
        delta=delta_text,
        delta_color="normal"
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

def draw_overlay_feedback(frame, validation, exercise):
    """Draw simple status on video frame only"""
    # Keep this minimal so the video isn't cluttered
    color = (0, 255, 0) if validation.is_correct else (255, 0, 0)
    
    # Simple top bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 10), color, -1)
    
    # Draw midpoint line and indicators for neck_side_to_side exercise
    if exercise.get_name() == "Neck Side to Side":
        h, w = frame.shape[:2]
        midpoint_x = int(w * 0.5)  # Midpoint at 50% of frame width
        
        # Draw vertical line at midpoint (yellow color for visibility)
        cv2.line(frame, (midpoint_x, 0), (midpoint_x, h), (0, 255, 255), 2)
        
        # Add label at top
        cv2.putText(frame, "MIDPOINT", (midpoint_x - 50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw indicators showing which side is which
        cv2.putText(frame, "LEFT", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "RIGHT", (w - 80, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

if __name__ == "__main__":
    main()