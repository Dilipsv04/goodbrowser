import time
import cv2
import pyautogui
import mediapipe as mp
from typing import Tuple, Optional

SCREEN_SCALE_X = 1.5
SCREEN_SCALE_Y = 1.5
BLINK_TIME = 2.0

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_gaze_time = time.time()
        self.last_gaze_position = None
        self.camera = None
        self.is_running = False
        self.frame_width = 1280
        self.frame_height = 720
        self.blink_start_time = None
        self.eyes_closed = False
        self.enabled = True
        
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

    def set_enabled(self, enabled):
        self.enabled = enabled

    def detect_blink(self, landmarks) -> bool:
        left_eye = [landmarks[362], landmarks[386]]
        right_eye = [landmarks[33], landmarks[160]]
        
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2
        
        if ear < 0.2:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.blink_start_time = time.time()
        else:
            if self.eyes_closed:
                if self.blink_start_time and (time.time() - self.blink_start_time) >= BLINK_TIME:
                    pyautogui.click()
                self.eyes_closed = False
                self.blink_start_time = None
        
        return self.eyes_closed

    def get_eye_aspect_ratio(self, eye):
        vertical = abs(eye[0].y - eye[1].y)
        return vertical

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.is_running = True

    def stop_camera(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def get_pupil_coordinates(self, frame) -> Optional[Tuple[int, int]]:
        if not self.enabled:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        
        self.detect_blink(landmarks)
        
        left_eye = landmarks[448]
        right_eye = landmarks[473]
        
        pupil_x = int(((left_eye.x + right_eye.x) / 2) * frame.shape[1])
        pupil_y = int(((left_eye.y + right_eye.y) / 2) * frame.shape[0])
        
        screen_x = int((pupil_x / self.frame_width) * self.screen_width * SCREEN_SCALE_X)
        screen_y = int((pupil_y / self.frame_height) * self.screen_height * SCREEN_SCALE_Y)
        
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        
        return (pupil_x, pupil_y)

    def process_frame(self, frame):
        pupil_coords = self.get_pupil_coordinates(frame)
        if pupil_coords is None:
            return frame

        pupil_x, pupil_y = pupil_coords

        cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (pupil_x, pupil_y), 10, (255, 255, 0), 2)

        status = "Eyes Closed (Clicking...)" if self.eyes_closed else "Eyes Open"
        tracking_status = "Enabled" if self.enabled else "Disabled"
        cv2.putText(frame, f"Status: {status} - Tracking: {tracking_status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
