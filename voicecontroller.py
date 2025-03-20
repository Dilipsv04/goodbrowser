import speech_recognition as sr
import threading
from signals import Signals
import logging

logger = logging.getLogger(__name__)

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
