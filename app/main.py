import sys
import os
from PySide6.QtWidgets import QApplication

# --- FIX: Add the parent directory to sys.path ---
# This ensures Python can find the 'app' package regardless of where you run this script from.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# -------------------------------------------------

# 1. Force Backend Configuration FIRST
from app.config import setup_qt_backend
setup_qt_backend()

# 2. Import Main Window
from app.ui.main_window import MedicalApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalApp()
    window.show()
    sys.exit(app.exec())