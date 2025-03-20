import sys
import logging
from PyQt5.QtWidgets import QApplication
from browser import Browser
from eyetracker import EyeTracker
from voicecontroller import VoiceController
from signals import Signals
import threading
logger = logging.getLogger(__name__)
import cv2

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
    
        while True:
            order = takecommand().lower()
            
            if 'open project' in order:
                openProjectLink()
                
            elif 'open youtube' in order:
                openYouTubeSearch('test')  # You can change 'test' to a dynamic query if needed
                
            elif 'search youtube' in order:
                speak("What do you want to search on YouTube?")
                query = takecommand()
                openYouTubeAndSelectVideo(query)
                
            elif 'open browser' in order:
                openBrowser()
                
            elif 'open' in order:  # To open any specified app like Chrome, Notepad, etc.
                app_name = order.replace('open', '').strip()
                openApp(app_name)
            
            elif 'exit' in order:
                speak("Goodbye!")
                break

    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        sys.exit(1)
