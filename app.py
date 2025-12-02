import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="Shaka Shot (자동촬영)", layout="centered")

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

st.title("🤙 Shaka Shot 자동 촬영기")
st.info("카메라를 보고 얼굴과 함께 **'샤카(Shaka) 포즈'**를 취하면 3초 뒤 찍어줍니다!")
st.markdown("*(샤카 포즈: 엄지와 새끼손가락만 펴고 나머지 세 손가락은 접는 하와이 인사법)*")

# ---------------- 2. Mediapipe 초기화 ----------------
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- 3. 헬퍼 함수: Shaka 판별 ----------------
def is_shaka(hand_landmarks, w, h):
    """
    엄지와 새끼는 펴져 있고(Up/Out), 검지/중지/약지는 접혀 있는지 확인
    """
    def c(i):
        lm = hand_landmarks.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    # 손가락 끝(tip)과 마디(ip/knuckle) 좌표
    thumb_tip = c(4); thumb_ip = c(3)
    index_tip = c(8); index_kn = c(5)
    middle_tip = c(12); middle_kn = c(9)
    ring_tip = c(16); ring_kn = c(13)
    pinky_tip = c(20); pinky_kn = c(17)

    # 판별 로직 (화면 좌표계: y는 아래로 갈수록 커짐)
    # 엄지와 새끼는 펴짐 (Tip이 관절보다 바깥쪽/위쪽) - 손 방향에 따라 다르지만 단순화
    # 여기서는 "접힘" 여부를 확실히 체크하는 것이 중요
    
    # 나머지 손가락(검지, 중지, 약지)은 확실히 접혀야 함 (Tip이 관절보다 아래/안쪽)
    # y좌표 기준: 접히면 Tip의 y가 Knuckle의 y보다 커야 함 (손을 위로 들었을 때 기준)
    # 하지만 손을 옆으로 들 수도 있으니 거리 기반이나 벡터가 정확하나, 
    # 간단하게 "나머지 세 손가락이 접혔는가"를 봅니다.
    
    index_folded = index_tip[1] > index_kn[1]
    middle_folded = middle_tip[1] > middle_kn[1]
    ring_folded = ring_tip[1] > ring_kn[1]
    
    # 엄지와 새끼는 펴져있어야 함 (반대 조건)
    thumb_extended = thumb_tip[1] < thumb_ip[1] or abs(thumb_tip[0] - thumb_ip[0]) > 20
    pinky_extended = pinky_tip[1] < pinky_kn[1]
    
    return index_folded and middle_folded and ring_folded and (thumb_extended or pinky_extended)

# ---------------- 4. 영상 처리 클래스 ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 모델 로드
        self.face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        
        # 상태 변수
        self.enter_time = None
        self.capture_triggered = False
        self.flash_frame = 0
        self.result_queue = queue.Queue() # 데이터 전송 통로

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. 감지 수행
        face_res = self.face_detector.process(rgb_img)
        hand_res = self.hand_detector.process(rgb_img)
        
        face_detected = face_res.detections is not None
        shaka_detected = False
        
        status_msg = "Show Face & Shaka"
        border_color = (0, 0, 255) # 빨강

        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        # 2. 손 인식 및 로직 확인
        if hand_res.multi_hand_landmarks:
            for hand_lms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                if is_shaka(hand_lms, w, h):
                    shaka_detected = True
        
        # 3. 얼굴 그리기
        if face_detected:
            for d in face_res.detections:
                mp_draw.draw_detection(img, d)

        # 4. 촬영 조건 확인 (얼굴 + 샤카)
        if face_detected and shaka_detected:
            status_msg = "HOLD ON! (3s)"
            border_color = (0, 255, 0) # 초록
            
            # 카운트다운 시작
            if self.enter_time is None:
                self.enter_time = time.time()
            
            elapsed = time.time() - self.enter_time
            countdown = 3.0 - elapsed
            
            # 화면 표시
            cv2.rectangle(img, (0,0), (w,h), border_color, 20)
            
            if countdown > 0:
                cv2.putText(img, f"{countdown:.1f}", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
            else:
                # ★ 촬영 시점 ★
                if not self.capture_triggered:
                    save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.result_queue.put(save_img) # 큐에 전송
                    self.capture_triggered = True
                    self.flash_frame = 5
        else:
            # 조건이 깨지면 타이머 리셋
            self.enter_time = None
            self.capture_triggered = False

        # 상태 텍스트 출력
        cv2.putText(img, status_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color, 2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- 5. UI 로직 ----------------

# 결과 화면
if st.session_state.snapshot is not None:
    st.success("📸 촬영 성공!")
    st.image(st.session_state.snapshot, caption="Shaka Shot Result", use_container_width=True)
    
    # 다운로드 버튼
    img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    if is_success:
        st.download_button(
            label="📥 사진 저장하기",
            data=buffer.tobytes(),
            file_name=f"Shaka_Shot_{int(time.time())}.jpg",
            mime="image/jpeg",
            type="primary",
            use_container_width=True
        )
    st.warning("🔄 다시 촬영하시려면 웹페이지를 새로고침 해주세요.")

# 촬영 화면
else:
    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    ctx = webrtc_streamer(
        key="shaka-camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    )

    # 큐 수신 루프
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result_img = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result_img is not None:
                        st.session_state.snapshot = result_img
                        st.rerun()
                except queue.Empty:
                    pass
            time.sleep(0.1)
