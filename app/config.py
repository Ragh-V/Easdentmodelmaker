import sys
import os

def setup_qt_backend():
    # --- FORCE PYSIDE6 BACKEND ---
    sys.modules["PyQt5"] = None
    sys.modules["PyQt5.QtCore"] = None
    sys.modules["PyQt5.QtGui"] = None
    sys.modules["PyQt5.QtWidgets"] = None

    os.environ["QT_API"] = "pyside6"
    os.environ["PYVISTA_QT_BACKEND"] = "pyside6"

    try:
        import PySide6
        from PySide6 import QtWidgets, QtCore, QtGui
        
        class MockModule: pass
        pyqt5 = MockModule()
        pyqt5.QtWidgets = QtWidgets
        pyqt5.QtCore = QtCore
        pyqt5.QtGui = QtGui
        
        sys.modules["PyQt5"] = pyqt5
        sys.modules["PyQt5.QtWidgets"] = QtWidgets
        sys.modules["PyQt5.QtCore"] = QtCore
        sys.modules["PyQt5.QtGui"] = QtGui
    except ImportError as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)