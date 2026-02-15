import streamlit as st
import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import numpy as np
import time
import plotly.graph_objects as go
# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="🤖 Robotic Arm Control",
    page_icon="🤖",
    layout="wide"
)
# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .connected { background: #00c853; color: white; }
    .disconnected { background: #ff1744; color: white; }
    .tracking { background: #00b0ff; color: white; }
    .no-hand { background: #ff9100; color: white; }
    .grip-open { background: #00e676; color: black; }
    .grip-closed { background: #ff5252; color: white; }
</style>
""", unsafe_allow_html=True)
# ===== CALIBRATED SERVO RANGES =====
GRIPPER_CLOSED, GRIPPER_OPEN = 0, 20
BASE_MIN, BASE_MAX = 0, 180
SHOULDER_MIN, SHOULDER_MAX = 0, 50
ELBOW_MIN, ELBOW_MAX = 90, 180
# Stability settings
SMOOTHING = 0.7
DEAD_ZONE = 15
# ===== SESSION STATE =====
defaults = {
    'ser': None, 'base': 90, 'shoulder': 25, 'elbow': 135, 'gripper': GRIPPER_OPEN,
    'hand_detected': False, 'grip_closed': False, 'gesture_mode': False,
    'prev_wrist_x': None, 'prev_wrist_y': None, 'prev_middle_y': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
# ===== HELPER FUNCTIONS =====
def get_ports():
    return [p.device for p in serial.tools.list_ports.comports()] or ["No ports"]
def connect(port):
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        return ser
    except:
        return None
def send_cmd(b, s, e, g):
    if st.session_state.ser and st.session_state.ser.is_open:
        st.session_state.ser.write(f"{b},{s},{e},{g}\n".encode())
def map_val(x, in_min, in_max, out_min, out_max):
    x = max(in_min, min(in_max, x))
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
def smooth(current, target):
    return int(current * SMOOTHING + target * (1 - SMOOTHING))
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])*2 + (p1[1]-p2[1])*2)
def is_closed(lms, w):
    pinch = distance(lms[4], lms[8]) < (w * 0.08)
    curled = sum([lms[8][1]>lms[5][1], lms[12][1]>lms[9][1], lms[16][1]>lms[13][1], lms[20][1]>lms[17][1]])
    return pinch or curled >= 3
def create_gauge(value, min_v, max_v, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 14, 'color': 'white'}},
        number={'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [min_v, max_v], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': '#1a1a2e',
            'bordercolor': '#333'
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=180, margin=dict(l=20,r=20,t=40,b=10))
    return fig
# ===== HEADER =====
st.markdown('<h1 class="main-header">🤖 Robotic Arm Control Dashboard</h1>', unsafe_allow_html=True)
# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## ⚙️ Connection")
    ports = get_ports()
    port = st.selectbox("COM Port", ports)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔗 Connect", use_container_width=True):
            st.session_state.ser = connect(port)
    with c2:
        if st.button("❌ Disconnect", use_container_width=True):
            if st.session_state.ser:
                st.session_state.ser.close()
                st.session_state.ser = None
    
    connected = st.session_state.ser and st.session_state.ser.is_open
    st.markdown(f'<div class="status-box {"connected" if connected else "disconnected"}">{"✅ Connected" if connected else "❌ Disconnected"}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("## 🎮 Quick Actions")
    if st.button("🏠 Home Position", use_container_width=True):
        st.session_state.base, st.session_state.shoulder = 90, 25
        st.session_state.elbow, st.session_state.gripper = 135, GRIPPER_OPEN
        send_cmd(90, 25, 135, GRIPPER_OPEN)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✊ Close", use_container_width=True):
            st.session_state.gripper = GRIPPER_CLOSED
            send_cmd(st.session_state.base, st.session_state.shoulder, st.session_state.elbow, GRIPPER_CLOSED)
    with c2:
        if st.button("🖐️ Open", use_container_width=True):
            st.session_state.gripper = GRIPPER_OPEN
            send_cmd(st.session_state.base, st.session_state.shoulder, st.session_state.elbow, GRIPPER_OPEN)
# ===== MAIN LAYOUT =====
col_cam, col_ctrl = st.columns([2, 1])
# ===== CAMERA & GESTURE CONTROL =====
with col_cam:
    st.markdown("### 📹 Camera & Gesture Control")
    
    gesture_on = st.toggle("🖐️ Enable Gesture Control", value=st.session_state.gesture_mode)
    st.session_state.gesture_mode = gesture_on
    
    # Camera placeholder
    FRAME_WINDOW = st.empty()
    status_placeholder = st.empty()
    
    if gesture_on:
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        stop_btn = st.button("⏹️ Stop Camera")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            hand_detected = False
            grip_closed = False
            
            if result.multi_hand_landmarks:
                hand_detected = True
                lms = [(int(l.x*w), int(l.y*h)) for l in result.multi_hand_landmarks[0].landmark]
                
                # Get positions
                wrist_x, wrist_y = lms[0]
                middle_y = lms[12][1]
                
                # Base control
                if st.session_state.prev_wrist_x is None or abs(wrist_x - st.session_state.prev_wrist_x) > DEAD_ZONE:
                    target_base = map_val(wrist_x, 50, w-50, BASE_MAX, BASE_MIN)
                    st.session_state.base = smooth(st.session_state.base, target_base)
                    st.session_state.prev_wrist_x = wrist_x
                
                # Shoulder control
                if st.session_state.prev_wrist_y is None or abs(wrist_y - st.session_state.prev_wrist_y) > DEAD_ZONE:
                    target_shoulder = map_val(wrist_y, 80, h-80, SHOULDER_MIN, SHOULDER_MAX)
                    st.session_state.shoulder = smooth(st.session_state.shoulder, target_shoulder)
                    st.session_state.prev_wrist_y = wrist_y
                
                # Elbow control
                if st.session_state.prev_middle_y is None or abs(middle_y - st.session_state.prev_middle_y) > DEAD_ZONE:
                    target_elbow = map_val(middle_y, 80, h-80, ELBOW_MIN, ELBOW_MAX)
                    st.session_state.elbow = smooth(st.session_state.elbow, target_elbow)
                    st.session_state.prev_middle_y = middle_y
                
                # Gripper control
                grip_closed = is_closed(lms, w)
                st.session_state.gripper = GRIPPER_CLOSED if grip_closed else GRIPPER_OPEN
                st.session_state.grip_closed = grip_closed
                
                # Draw landmarks
                mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                color = (0, 255, 0) if grip_closed else (0, 0, 255)
                cv2.circle(frame, lms[4], 10, color, -1)
                cv2.circle(frame, lms[8], 10, color, -1)
                cv2.line(frame, lms[4], lms[8], color, 2)
            else:
                st.session_state.prev_wrist_x = None
                st.session_state.prev_wrist_y = None
                st.session_state.prev_middle_y = None
            
            st.session_state.hand_detected = hand_detected
            
            # Send to Arduino
            send_cmd(st.session_state.base, st.session_state.shoulder, st.session_state.elbow, st.session_state.gripper)
            
            # Draw info on frame
            cv2.rectangle(frame, (5,5), (180,100), (0,0,0), -1)
            cv2.putText(frame, f"Base: {st.session_state.base}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Shoulder: {st.session_state.shoulder}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Elbow: {st.session_state.elbow}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Gripper: {'CLOSED' if grip_closed else 'OPEN'}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if grip_closed else (0,0,255), 1)
            
            # Show frame
            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            
            # Status
            if hand_detected:
                status_placeholder.markdown('<div class="status-box tracking">🖐️ Hand Tracking Active</div>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<div class="status-box no-hand">⚠️ No Hand Detected</div>', unsafe_allow_html=True)
            
            time.sleep(0.03)
        
        cap.release()
    else:
        st.info("👆 Enable gesture control to start camera and hand tracking")
# ===== MANUAL CONTROLS =====
with col_ctrl:
    st.markdown("### 🎛️ Manual Control")
    
    # Base
    st.markdown("*🔄 Base*")
    new_base = st.slider("Base", BASE_MIN, BASE_MAX, st.session_state.base, key="b_slider", disabled=gesture_on)
    if not gesture_on and new_base != st.session_state.base:
        st.session_state.base = new_base
        send_cmd(new_base, st.session_state.shoulder, st.session_state.elbow, st.session_state.gripper)
    
    # Shoulder
    st.markdown("*💪 Shoulder*")
    new_sh = st.slider("Shoulder", SHOULDER_MIN, SHOULDER_MAX, st.session_state.shoulder, key="s_slider", disabled=gesture_on)
    if not gesture_on and new_sh != st.session_state.shoulder:
        st.session_state.shoulder = new_sh
        send_cmd(st.session_state.base, new_sh, st.session_state.elbow, st.session_state.gripper)
    
    # Elbow
    st.markdown("*🦾 Elbow*")
    new_el = st.slider("Elbow", ELBOW_MIN, ELBOW_MAX, st.session_state.elbow, key="e_slider", disabled=gesture_on)
    if not gesture_on and new_el != st.session_state.elbow:
        st.session_state.elbow = new_el
        send_cmd(st.session_state.base, st.session_state.shoulder, new_el, st.session_state.gripper)
    
    # Gripper
    st.markdown("*🤏 Gripper*")
    new_gr = st.slider("Gripper", GRIPPER_CLOSED, GRIPPER_OPEN, st.session_state.gripper, key="g_slider", disabled=gesture_on)
    if not gesture_on and new_gr != st.session_state.gripper:
        st.session_state.gripper = new_gr
        send_cmd(st.session_state.base, st.session_state.shoulder, st.session_state.elbow, new_gr)
    
    st.divider()
    
    # Gauges
    st.markdown("### 📊 Servo Positions")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_gauge(st.session_state.base, BASE_MIN, BASE_MAX, "Base", "#00d4ff"), use_container_width=True)
        st.plotly_chart(create_gauge(st.session_state.elbow, ELBOW_MIN, ELBOW_MAX, "Elbow", "#ff6b6b"), use_container_width=True)
    with c2:
        st.plotly_chart(create_gauge(st.session_state.shoulder, SHOULDER_MIN, SHOULDER_MAX, "Shoulder", "#7b2ff7"), use_container_width=True)
        st.plotly_chart(create_gauge(st.session_state.gripper, GRIPPER_CLOSED, GRIPPER_OPEN, "Gripper", "#00e676"), use_container_width=True)
# ===== FOOTER =====
st.divider()
st.markdown('<div style="text-align:center;color:#666;">🤖 Hand Gesture Robotic Arm | Streamlit + OpenCV + MediaPipe + Arduino</div>', unsafe_allow_html=True)