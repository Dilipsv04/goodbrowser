'''import sys
import time
import threading
import logging
from typing import Tuple, Optional
from queue import Queue

import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLineEdit, QVBoxLayout, 
                           QWidget, QPushButton, QHBoxLayout, QLabel)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject
from PyQt5.QtGui import QIcon
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DWELL_TIME = 1.5
BLINK_TIME = 2.0  # Time in seconds for blink detection
GAZE_STABILITY_THRESHOLD = 15
SCREEN_SCALE_X = 1.5
SCREEN_SCALE_Y = 1.5

class Signals(QObject):
    navigate = pyqtSignal(str)
    click = pyqtSignal()
    scroll_up = pyqtSignal()
    scroll_down = pyqtSignal()
    type_text = pyqtSignal(str)
    search = pyqtSignal(str)
    click_nth_link = pyqtSignal(int)
    click_nth_video = pyqtSignal(int)

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
        
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

    def detect_blink(self, landmarks) -> bool:
        # Get eye landmarks
        left_eye = [landmarks[362], landmarks[386]]  # Upper and lower landmarks
        right_eye = [landmarks[33], landmarks[160]]   # Upper and lower landmarks
        
        # Calculate eye aspect ratio
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2
        
        # Threshold for closed eyes
        if ear < 0.2:  # Adjust this threshold as needed
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
        # Calculate vertical distance
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Check for blink
        self.detect_blink(landmarks)
        
        # Get both eyes' landmarks
        left_eye = landmarks[468]
        right_eye = landmarks[473]
        
        # Average the positions
        pupil_x = int(((left_eye.x + right_eye.x) / 2) * frame.shape[1])
        pupil_y = int(((left_eye.y + right_eye.y) / 2) * frame.shape[0])
        
        # Map camera coordinates to screen coordinates
        screen_x = int((pupil_x / self.frame_width) * self.screen_width * SCREEN_SCALE_X)
        screen_y = int((pupil_y / self.frame_height) * self.screen_height * SCREEN_SCALE_Y)
        
        # Move mouse pointer smoothly
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        
        return (pupil_x, pupil_y)

    def process_frame(self, frame):
        pupil_coords = self.get_pupil_coordinates(frame)
        if pupil_coords is None:
            return frame

        pupil_x, pupil_y = pupil_coords

        # Draw visualization
        cv2.circle(frame, (pupil_x, pupil_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (pupil_x, pupil_y), 10, (255, 255, 0), 2)

        # Add status text
        status = "Eyes Closed (Clicking...)" if self.eyes_closed else "Eyes Open"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

class VoiceController:
    def __init__(self, signals):
        self.recognizer = sr.Recognizer()
        self.is_running = False
        self.signals = signals
        
    def start(self):
        self.is_running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _listen_loop(self):
        while self.is_running:
            try:
                with sr.Microphone() as source:
                    logger.info("Listening for command...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized command: {command}")
                self.handle_command(command)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Could not request results: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in voice recognition: {e}")

    def handle_command(self, command):
        try:
            if command.startswith("go to "):
                url = command.split("go to ")[1].strip()
                if not url.startswith(('http://', 'https://')):
                    url = f"https://{url}"
                self.signals.navigate.emit(url)
            
            elif command.startswith("search for "):
                search_query = command.split("search for ")[1].strip()
                self.signals.search.emit(search_query)
            
            elif command.startswith("type "):
                text = command.split("type ")[1].strip()
                self.signals.type_text.emit(text)
            
            elif command == "click":
                self.signals.click.emit()
            
            elif command.startswith("click on the first ") and "link" in command:
                try:
                    number = int(''.join(filter(str.isdigit, command)))
                    self.signals.click_nth_link.emit(number)
                except ValueError:
                    self.signals.click_nth_link.emit(1)
            
            elif command == "click the first video":
                self.signals.click_nth_video.emit(1)
            
            elif "scroll" in command:
                if "up" in command:
                    self.signals.scroll_up.emit()
                elif "down" in command:
                    self.signals.scroll_down.emit()
        
        except Exception as e:
            logger.error(f"Error handling command: {e}")

class Browser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = Signals()
        self.init_ui()
        self.setup_signals()
        
        # Initialize Selenium WebDriver for advanced interactions
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in background
        self.driver = webdriver.Chrome(options=options)
        
    def init_ui(self):
        self.setWindowTitle("Eye & Voice Controlled Browser")
        self.setGeometry(100, 100, 1280, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Toolbar
        toolbar = QHBoxLayout()
        
        self.back_btn = QPushButton("←")
        self.forward_btn = QPushButton("→")
        self.back_btn.clicked.connect(self.go_back)
        self.forward_btn.clicked.connect(self.go_forward)
        
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        
        self.reload_btn = QPushButton("⟳")
        self.reload_btn.clicked.connect(self.reload_page)

        toolbar.addWidget(self.back_btn)
        toolbar.addWidget(self.forward_btn)
        toolbar.addWidget(self.url_bar)
        toolbar.addWidget(self.reload_btn)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://www.google.com"))
        self.browser.urlChanged.connect(self.update_url)

        layout.addLayout(toolbar)
        layout.addWidget(self.browser)

        self.statusBar().showMessage("Eye tracking: Active")

    def setup_signals(self):
        self.signals.navigate.connect(self.navigate_to_url)
        self.signals.click.connect(lambda: pyautogui.click())
        self.signals.scroll_up.connect(lambda: pyautogui.scroll(100))
        self.signals.scroll_down.connect(lambda: pyautogui.scroll(-100))
        self.signals.type_text.connect(self.type_text)
        self.signals.search.connect(self.perform_search)
        self.signals.click_nth_link.connect(self.click_nth_link)
        self.signals.click_nth_video.connect(self.click_nth_video)

    def type_text(self, text):
        pyautogui.write(text)

    def perform_search(self, query):
        self.url_bar.setText(f"https://www.google.com/search?q={query}")
        self.navigate_to_url()

    def click_nth_link(self, n):
        try:
            # Update Selenium with current page
            self.driver.get(self.browser.url().toString())
            # Wait for and click nth search result
            links = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g h3"))
            )
            if len(links) >= n:
                link_location = links[n-1].location
                pyautogui.click(link_location['x'], link_location['y'])
        except Exception as e:
            logger.error(f"Error clicking nth link: {e}")

    def click_nth_video(self, n):
        try:
            current_url = self.browser.url().toString()
            if "youtube.com" in current_url:
                self.driver.get(current_url)
                videos = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-video-renderer"))
                )
                if len(videos) >= n:
                    video_location = videos[n-1].location
                    pyautogui.click(video_location['x'], video_location['y'])
        except Exception as e:
            logger.error(f"Error clicking video: {e}")

    def navigate_to_url(self, url=None):
        if isinstance(url, bool):  # Handle returnPressed signal
            url = None
        if url is None:
            url = self.url_bar.text()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.browser.setUrl(QUrl(url))

    def update_url(self, url):
        self.url_bar.setText(url.toString())

    def go_back(self):
        self.browser.back()

    def go_forward(self):
        self.browser.forward()

    def reload_page(self):
        self.browser.reload()

def main():
    app = QApplication(sys.argv)
    
    browser_window = Browser()
    browser_window.show()

    eye_tracker = EyeTracker()
    voice_controller = VoiceController(browser_window.signals)

    voice_controller.start()

    def eye_tracking_loop():
        try:
            eye_tracker.start_camera()
            
            while eye_tracker.is_running:
                ret, frame = eye_tracker.camera.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = eye_tracker.process_frame(frame)
                
                cv2.imshow("Eye Tracking Preview", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Error in eye tracking loop: {e}")
        finally:
            eye_tracker.stop_camera()
            cv2.destroyAllWindows()

    tracking_thread = threading.Thread(target=eye_tracking_loop, daemon=True)
    tracking_thread.start()

    sys.exit(app.exec_())
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        sys.exit(1)
        '''
import sys
import time
import threading
import logging
from typing import Tuple, Optional
from queue import Queue

import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLineEdit, QVBoxLayout, 
                           QWidget, QPushButton, QHBoxLayout, QLabel)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QObject, QRect
from PyQt5.QtGui import QIcon, QPainter, QPen, QBrush, QColor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DWELL_TIME = 1.5
BLINK_TIME = 2.0
GAZE_STABILITY_THRESHOLD = 15
SCREEN_SCALE_X = 1.5
SCREEN_SCALE_Y = 1.5

class Signals(QObject):
    navigate = pyqtSignal(str)
    click = pyqtSignal()
    scroll_up = pyqtSignal()
    scroll_down = pyqtSignal()
    type_text = pyqtSignal(str)
    search = pyqtSignal(str)
    show_grid = pyqtSignal()
    zoom_grid = pyqtSignal(int)
    select_grid = pyqtSignal(int)
    close_grid = pyqtSignal()
    toggle_eye_tracking = pyqtSignal(bool)
    click_nth_link = pyqtSignal(int)
    click_nth_video = pyqtSignal(int)

class GridOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.parent = parent
        self.zoomed_grid = None
        self.zoomed_number = None
        self.grid_visible = False

    def show_grid(self):
        if self.parent:
            self.setGeometry(self.parent.geometry())
            self.grid_visible = True
            self.zoomed_grid = None
            self.zoomed_number = None
            self.show()
            self.update()

    def zoom_grid(self, number):
        if 1 <= number <= 9:
            self.zoomed_number = number
            self.zoomed_grid = self.calculate_grid_rect(number)
            self.update()

    def calculate_grid_rect(self, number):
        width = self.width() // 3
        height = self.height() // 3
        row = (number - 1) // 3
        col = (number - 1) % 3
        return QRect(col * width, row * height, width, height)

    def paintEvent(self, event):
        if not self.grid_visible:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.zoomed_grid:
            self.draw_zoomed_grid(painter)
        else:
            self.draw_main_grid(painter)

    def draw_main_grid(self, painter):
        width = self.width() // 3
        height = self.height() // 3

        for i in range(9):
            row = i // 3
            col = i % 3
            rect = QRect(col * width, row * height, width, height)
            self.draw_grid_cell(painter, rect, i + 1)

    def draw_zoomed_grid(self, painter):
        rect = self.zoomed_grid
        width = rect.width() // 3
        height = rect.height() // 3

        for i in range(9):
            row = i // 3
            col = i % 3
            sub_rect = QRect(rect.x() + col * width, rect.y() + row * height, width, height)
            self.draw_grid_cell(painter, sub_rect, i + 1)

    def draw_grid_cell(self, painter, rect, number):
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(rect, Qt.AlignCenter, str(number))

    def close_grid(self):
        self.grid_visible = False
        self.hide()

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
        
        left_eye = landmarks[468]
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

class VoiceController:
    def __init__(self, signals):
        self.recognizer = sr.Recognizer()
        self.is_running = False
        self.signals = signals
        self.click_commands = ['click', 'press', 'c l i c k', 'p r e s s']
        
    def start(self):
        self.is_running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _listen_loop(self):
        while self.is_running:
            try:
                with sr.Microphone() as source:
                    logger.info("Listening for command...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Recognized command: {command}")
                self.handle_command(command)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                logger.debug("Could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Could not request results: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in voice recognition: {e}")

    def handle_command(self, command):
        try:
            for click_cmd in self.click_commands:
                if click_cmd in command:
                    self.signals.click.emit()
                    return

            if command.startswith(('show me grid', 'show me g r i d')):
                self.signals.show_grid.emit()
                return

            if command.startswith('zoom on grid '):
                try:
                    number = int(command.split('zoom on grid ')[1])
                    self.signals.zoom_grid.emit(number)
                    return
                except ValueError:
                    pass

            if command.startswith('select grid '):
                try:
                    number = int(command.split('select grid ')[1])
                    self.signals.select_grid.emit(number)
                    return
                except ValueError:
                    pass

            if command == 'close grid':
                self.signals.close_grid.emit()
                return

            if command in ['enable eye tracking', 'disable eye tracking']:
                self.signals.toggle_eye_tracking.emit(command == 'enable eye tracking')
                return

            if command.startswith("go to "):
                url = command.split("go to ")[1].strip()
                if not url.startswith(('http://', 'https://')):
                    url = f"https://{url}"
                self.signals.navigate.emit(url)
            
            elif command.startswith("search for "):
                search_query = command.split("search for ")[1].strip()
                self.signals.search.emit(search_query)
            
            elif command.startswith("type "):
                text = command.split("type ")[1].strip()
                self.signals.type_text.emit(text)
            
            elif "scroll" in command:
                if "up" in command:
                    self.signals.scroll_up.emit()
                elif "down" in command:
                    self.signals.scroll_down.emit()

        except Exception as e:
            logger.error(f"Error handling command: {e}")

class Browser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = Signals()
        self.init_ui()
        self.setup_signals()
        self.grid_overlay = GridOverlay(self)
        self.eye_tracking_enabled = True
        self.eye_tracker = None
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)

    def init_ui(self):
        self.setWindowTitle("Eye & Voice Controlled Browser")
        self.setGeometry(100, 100, 1280, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        toolbar = QHBoxLayout()
        
        self.back_btn = QPushButton("←")
        self.forward_btn = QPushButton("→")
        self.back_btn.clicked.connect(self.go_back)
        self.forward_btn.clicked.connect(self.go_forward)
        
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        
        self.reload_btn = QPushButton("⟳")
        self.reload_btn.clicked.connect(self.reload_page)

        toolbar.addWidget(self.back_btn)
        toolbar.addWidget(self.forward_btn)
        toolbar.addWidget(self.url_bar)
        toolbar.addWidget(self.reload_btn)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://www.google.com"))
        self.browser.urlChanged.connect(self.update_url)

        layout.addLayout(toolbar)
        layout.addWidget(self.browser)

        self.statusBar().showMessage("Eye tracking: Active")

    def setup_signals(self):
        self.signals.navigate.connect(self.navigate_to_url)
        self.signals.click.connect(lambda: pyautogui.click())
        self.signals.scroll_up.connect(lambda: pyautogui.scroll(100))
        self.signals.scroll_down.connect(lambda: pyautogui.scroll(-100))
        self.signals.type_text.connect(self.type_text)
        self.signals.search.connect(self.perform_search)
        self.signals.click_nth_link.connect(self.click_nth_link)
        self.signals.click_nth_video.connect(self.click_nth_video)

    def type_text(self, text):
        pyautogui.write(text)

    def perform_search(self, query):
        self.url_bar.setText(f"https://www.google.com/search?q={query}")
        self.navigate_to_url()

    def click_nth_link(self, n):
        try:
            # Update Selenium with current page
            self.driver.get(self.browser.url().toString())
            # Wait for and click nth search result
            links = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g h3"))
            )
            if len(links) >= n:
                link_location = links[n-1].location
                pyautogui.click(link_location['x'], link_location['y'])
        except Exception as e:
            logger.error(f"Error clicking nth link: {e}")

    def click_nth_video(self, n):
        try:
            current_url = self.browser.url().toString()
            if "youtube.com" in current_url:
                self.driver.get(current_url)
                videos = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-video-renderer"))
                )
                if len(videos) >= n:
                    video_location = videos[n-1].location
                    pyautogui.click(video_location['x'], video_location['y'])
        except Exception as e:
            logger.error(f"Error clicking video: {e}")

    def navigate_to_url(self, url=None):
        if isinstance(url, bool):  # Handle returnPressed signal
            url = None
        if url is None:
            url = self.url_bar.text()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        self.browser.setUrl(QUrl(url))

    def update_url(self, url):
        self.url_bar.setText(url.toString())

    def go_back(self):
        self.browser.back()

    def go_forward(self):
        self.browser.forward()

    def reload_page(self):
        self.browser.reload()

def main():
    app = QApplication(sys.argv)
    
    browser_window = Browser()
    browser_window.show()

    eye_tracker = EyeTracker()
    voice_controller = VoiceController(browser_window.signals)

    voice_controller.start()

    def eye_tracking_loop():
        try:
            eye_tracker.start_camera()
            
            while eye_tracker.is_running:
                ret, frame = eye_tracker.camera.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = eye_tracker.process_frame(frame)
                
                cv2.imshow("Eye Tracking Preview", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Error in eye tracking loop: {e}")
        finally:
            eye_tracker.stop_camera()
            cv2.destroyAllWindows()

    tracking_thread = threading.Thread(target=eye_tracking_loop, daemon=True)
    tracking_thread.start()

    sys.exit(app.exec_())
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        sys.exit(1)