from PyQt5.QtCore import pyqtSignal, QObject

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
