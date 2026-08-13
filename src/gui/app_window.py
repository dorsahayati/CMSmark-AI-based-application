import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cmsmark
import os
import subprocess
from PyQt5 import QtCore
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Worker(QtCore.QObject):
    progress_text = QtCore.pyqtSignal(str)
    progress_value = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, input_files, threshold, needs_normalization):
        super().__init__()
        self.input_files = input_files
        self.threshold = threshold
        self.needs_normalization = needs_normalization # Store the value
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if not self.input_files:
                self.error.emit("No input files were provided to the worker.")
                return

            total_files = len(self.input_files)
            final_results_path = ""

            for i, csv_path in enumerate(self.input_files):
                if not self.is_running:
                    break
                
                self.progress_text.emit(f"Processing file {i+1}/{total_files}: {Path(csv_path).name}")
                self.progress_value.emit(int((i / total_files) * 100))

                # This is the long-running function from cmsmark.py
                # It's now correctly called inside the worker's run method.
                # perform_inference() saves results under the app's own
                # "results" folder and returns that path directly, so the
                # UI never has to guess/recompute it.
                final_results_path = cmsmark.perform_inference(csv_path, self.threshold, self.needs_normalization)
                self.progress_text.emit(f"Finished processing {Path(csv_path).name}. Results saved.")

            if self.is_running:
                self.progress_value.emit(100)
                self.finished.emit(final_results_path)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

class AppWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CMSMARK") # Changed window title
        
        # Use the get_icon_path method instead of resource_path
        icon_path = self.get_icon_path()
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        
        self.setGeometry(100, 100, 700, 550)
        self.set_stylesheet()
        self.initUI()
        self.results_path = None

    def set_stylesheet(self):
        style = """
            QWidget {
                background-color: #e8ebf7; /* Main background color */
                color: #1c1c1c; /* Main text color - dark for contrast */
                font-family: Segoe UI;
                font-size: 10pt;
            }
            QMainWindow {
                border: 1px solid #2e4a9e; /* Slightly darker border */
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #2e4a9e;
                border-radius: 5px;
                margin-top: 10px; /* Create space for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                left: 10px;
                /* This positions the title inside the margin space, not on the border */
            }
            QPushButton {
                background-color: #3f5fc2; /* Primary button color (lighter ribbon blue) */
                color: #ffffff;
                border: 1px solid #2e4a9e;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2e4a9e; /* Darker on hover */
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #223679; /* Even darker press */
            }
            QPushButton:disabled {
                background-color: #b0b9db;
                color: #888;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #ffffff; /* White background for inputs */
                border: 1px solid #2e4a9e;
                padding: 4px;
                border-radius: 4px;
            }
            QTextEdit, QLineEdit {
                background-color: #ffffff; /* White background for log */
                border: 1px solid #2e4a9e;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #1c1c1c; /* Dark text */
            }
            QProgressBar {
                border: 1px solid #2e4a9e;
                border-radius: 4px;
                text-align: center;
                background-color: #ffffff;
                color: #1c1c1c;
            }
            QProgressBar::chunk {
                background-color: #2e4a9e; /* A modern blue for progress */
                border-radius: 3px;
            }
        """
        self.setStyleSheet(style)

    def initUI(self):
        # Main container and layout
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        main_layout = QtWidgets.QVBoxLayout(container)

        # --- NEW: Input File Group ---
        input_group = QtWidgets.QGroupBox("Input file(s)")
        input_layout = QtWidgets.QVBoxLayout()
        input_group.setLayout(input_layout)
        self.uploadButton = QtWidgets.QPushButton("Upload CSV File(s)")
        input_layout.addWidget(self.uploadButton)

        # --- Parameters Group (modified) ---
        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QGridLayout()
        params_group.setLayout(params_layout)

        self.threshold_label = QtWidgets.QLabel("Threshold:")
        self.thresholdSpin = QtWidgets.QDoubleSpinBox()
        self.thresholdSpin.setRange(0.0, 1.0); self.thresholdSpin.setSingleStep(0.01); self.thresholdSpin.setValue(0.5)
        
        # The app only ever uses the "Excellent" feature set — this is shown
        # as a fixed, non-editable value rather than a selectable list.
        self.model_group_label = QtWidgets.QLabel("Feature Set:")
        self.groupValueLabel = QtWidgets.QLabel(cmsmark.FEATURE_GROUP)
        self.groupValueLabel.setStyleSheet("font-weight: bold;")

        self.normalize_checkbox = QtWidgets.QCheckBox("Apply Log2CPM+1 transformation")
        self.normalize_checkbox.setChecked(True)

        params_layout.addWidget(self.threshold_label, 0, 0)
        params_layout.addWidget(self.thresholdSpin, 0, 1)
        params_layout.addWidget(self.model_group_label, 1, 0)
        params_layout.addWidget(self.groupValueLabel, 1, 1)
        params_layout.addWidget(self.normalize_checkbox, 2, 0, 1, 2)

        # --- Run and Results Group (unchanged) ---
        run_group = QtWidgets.QGroupBox("Run")
        run_layout = QtWidgets.QHBoxLayout()
        run_group.setLayout(run_layout)
        self.runButton = QtWidgets.QPushButton("Run CMSMARK")
        self.openResultsButton = QtWidgets.QPushButton("Open Results Folder")
        self.openResultsButton.hide()
        run_layout.addWidget(self.runButton)
        run_layout.addWidget(self.openResultsButton)

        # --- Log and Progress Group (unchanged) ---
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout()
        log_group.setLayout(log_layout)
        self.statusLabel = QtWidgets.QLabel("Ready")
        self.progressBar = QtWidgets.QProgressBar()
        self.logView = QtWidgets.QTextEdit()
        self.logView.setReadOnly(True)
        log_layout.addWidget(self.statusLabel)
        log_layout.addWidget(self.progressBar)
        log_layout.addWidget(self.logView)

        # Add groups to main layout
        main_layout.addWidget(input_group) # Add new input group first
        main_layout.addWidget(params_group)
        main_layout.addWidget(run_group)
        main_layout.addWidget(log_group)

        # Connections
        self.uploadButton.clicked.connect(self.select_files)
        self.runButton.clicked.connect(self.run_inference)
        self.openResultsButton.clicked.connect(self.open_results_folder)

        self.input_files = []
        self._worker_thread = None

    def open_results_folder(self):
        if self.results_path and os.path.exists(self.results_path):
            # Use os.startfile for a cross-platform way to open a folder
            subprocess.run(['explorer', os.path.normpath(self.results_path)])
        else:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Results path not found or not set.")

    def select_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select input CSV file(s)", "", "CSV Files (*.csv)")
        if files:
            self.input_files = files
            self.statusLabel.setText(f"{len(files)} file(s) selected")
            self.logView.append(f"Selected {len(files)} file(s): {', '.join(Path(f).name for f in files)}")

    def append_log(self, text):
        self.logView.append(text)
        self.statusLabel.setText(text)

    def run_inference(self):
        if not self.input_files:
            QtWidgets.QMessageBox.warning(self, "No Input", "Please upload one or more CSV files first.")
            return

        self.runButton.setEnabled(False)
        self.uploadButton.setEnabled(False)
        self.openResultsButton.hide()
        self.logView.clear()
        self.progressBar.setValue(0)

        threshold = self.thresholdSpin.value()
        normalize_data = self.normalize_checkbox.isChecked() # Get the value from the checkbox

        # Create and run the worker thread, passing the new value
        self.thread = QtCore.QThread()
        self.worker = Worker(self.input_files, threshold, normalize_data)
        self.worker.moveToThread(self.thread)

        # Connect signals from worker to UI slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.progress_text.connect(self.append_log)
        self.worker.progress_value.connect(self.progressBar.setValue)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)

        self.thread.start()

    def _on_worker_finished(self, results_path):
        self.results_path = results_path
        self.progressBar.setValue(100)
        self.append_log("CMSMARK completed successfully.")
        self.runButton.setEnabled(True)
        self.uploadButton.setEnabled(True)
        if results_path:
            self.openResultsButton.show()

    def _on_worker_error(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during inference:\n{message}")
        self.append_log(f"ERROR: {message}")
        self.runButton.setEnabled(True)
        self.uploadButton.setEnabled(True)
        self.progressBar.setValue(0)

    def get_icon_path(self):
        """Get the correct path to the icon file whether running as script or frozen exe"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        return os.path.join(base_path, 'icon', 'app_icon.ico')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
