#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QProgressBar, QMenuBar, QMenu,
    QMessageBox, QHBoxLayout, QStatusBar, QTabWidget, QComboBox,
    QSpinBox, QGroupBox, QStyle, QFrame, QToolButton
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QIcon, QFont, QKeySequence, QShortcut
import soundfile as sf
from pathlib import Path
from whisperSSTis.transcribe import load_model, transcribe_long_audio, create_srt
from whisperSSTis.audio import get_file_info, get_audio_devices, record_audio

class RecordingWorker(QThread):
    finished = Signal(object)  # Emits the recorded audio data
    error = Signal(str)

    def __init__(self, duration, device_id=None):
        super().__init__()
        self.duration = duration
        self.device_id = device_id

    def run(self):
        try:
            audio_data = record_audio(self.duration, self.device_id)
            self.finished.emit(audio_data)
        except Exception as e:
            self.error.emit(str(e))

class TranscriptionWorker(QThread):
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, audio_data, duration, model, processor):
        super().__init__()
        self.audio_data = audio_data
        self.duration = duration
        self.model = model
        self.processor = processor

    def run(self):
        try:
            transcriptions = transcribe_long_audio(
                self.audio_data,
                self.model,
                self.processor,
                self.duration
            )
            self.finished.emit(transcriptions)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperSST")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        self.current_audio = None
        self.current_duration = None
        
        # Initialize model in a separate thread
        self.statusBar().showMessage("Loading model...")
        self.model_thread = QThread()
        self.model_thread.run = self.load_model_thread
        self.model_thread.finished.connect(self.model_loaded)
        self.model_thread.start()
        
        # Disable UI until model is loaded
        self.setEnabled(False)

    def setup_ui(self):
        # Set application style
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                padding: 10px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                padding: 6px 12px;
                min-width: 80px;
            }
            QTextEdit {
                font-family: monospace;
                padding: 5px;
            }
            QLabel {
                padding: 5px;
            }
        """)
        # Menu Bar with icons
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = file_menu.addAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open Audio File")
        open_action.triggered.connect(self.load_audio)
        open_action.setShortcut("Ctrl+O")
        
        save_action = file_menu.addAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save Transcription")
        save_action.triggered.connect(self.save_transcription)
        save_action.setShortcut("Ctrl+S")
        
        file_menu.addSeparator()
        exit_action = file_menu.addAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "Exit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")

        # Central Widget with Tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # File Input Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(10)
        
        # Audio File Section
        file_group = QGroupBox("Audio File")
        file_section = QVBoxLayout()  # Changed to VBox for better organization
        
        # Button section
        button_section = QHBoxLayout()
        self.load_button = QPushButton("Load Audio")
        self.load_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.load_button.setToolTip("Click to select an audio file")
        self.load_button.clicked.connect(self.load_audio)
        button_section.addWidget(self.load_button)
        button_section.addStretch()
        file_section.addLayout(button_section)
        
        # Info section
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        info_layout = QVBoxLayout(info_frame)
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setTextFormat(Qt.PlainText)
        info_layout.addWidget(self.file_info_label)
        file_section.addWidget(info_frame)
        
        file_group.setLayout(file_section)
        file_layout.addWidget(file_group)

        # Record Audio Tab
        record_tab = QWidget()
        record_layout = QVBoxLayout(record_tab)
        record_layout.setContentsMargins(10, 10, 10, 10)
        record_layout.setSpacing(10)

        # Device Selection
        device_group = QGroupBox("Recording Device")
        device_layout = QVBoxLayout()
        refresh_layout = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select your recording device")
        refresh_button = QPushButton()
        refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_button.setToolTip("Refresh device list")
        refresh_button.clicked.connect(self.refresh_devices)
        refresh_layout.addWidget(self.device_combo)
        refresh_layout.addWidget(refresh_button)
        device_layout.addLayout(refresh_layout)
        device_group.setLayout(device_layout)
        record_layout.addWidget(device_group)

        # Recording Controls
        record_controls = QGroupBox("Recording Controls")
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        duration_layout = QVBoxLayout()
        duration_label = QLabel("Duration:")
        duration_label.setAlignment(Qt.AlignCenter)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" seconds")
        self.duration_spin.setToolTip("Set recording duration")
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        controls_layout.addLayout(duration_layout)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))  # Using a checkmark icon instead
        self.record_button.setToolTip("Click to start/stop recording")
        self.record_button.clicked.connect(self.toggle_recording)

        controls_layout.addWidget(self.record_button)
        
        record_controls.setLayout(controls_layout)
        record_layout.addWidget(record_controls)

        # Add tabs
        tab_widget.addTab(file_tab, "File Input")
        tab_widget.addTab(record_tab, "Record Audio")

        # Common elements (shared between tabs)
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Transcription Display
        display_group = QGroupBox("Transcription")
        display_layout = QVBoxLayout()
        
        # Add toolbar for transcription actions
        toolbar_layout = QHBoxLayout()
        
        clear_button = QToolButton()
        clear_button.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        clear_button.setToolTip("Clear transcription (Ctrl+R)")
        clear_button.clicked.connect(self.clear_transcription)
        toolbar_layout.addWidget(clear_button)
        
        # Add clear shortcut
        QShortcut(QKeySequence("Ctrl+R"), self, self.clear_transcription)
        
        toolbar_layout.addStretch()
        display_layout.addLayout(toolbar_layout)
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Monospace", 10))
        self.text_display.setPlaceholderText("Transcription will appear here...")
        display_layout.addWidget(self.text_display)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)

        # Transcribe Button
        transcribe_layout = QHBoxLayout()
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        self.transcribe_button.setToolTip("Click to start transcription (Ctrl+T)")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        transcribe_layout.addWidget(self.transcribe_button)
        
        # Add transcribe shortcut
        QShortcut(QKeySequence("Ctrl+T"), self, self.start_transcription)
        
        main_layout.addLayout(transcribe_layout)

        # Status Bar
        self.statusBar().showMessage("Ready")

        # Recording timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)
        self.recording_time = 0

    def load_audio(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*.*)"
        )
        
        if file_path:
            try:
                audio_data, sr = sf.read(file_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                self.current_audio = audio_data
                self.current_duration = len(audio_data) / sr
                
                # Display file info
                file_info = get_file_info(audio_data, sr)
                info_text = "\n".join([f"{k}: {v}" for k, v in file_info.items()])
                self.file_info_label.setText(info_text)
                
                self.transcribe_button.setEnabled(True)
                self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")

    def refresh_devices(self):
        self.device_combo.clear()
        devices = get_audio_devices()
        for name, device_id in devices.items():
            self.device_combo.addItem(name, device_id)

    def toggle_recording(self):
        if self.record_button.text() == "Start Recording":
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        duration = self.duration_spin.value()
        device_id = self.device_combo.currentData()
        
        self.record_button.setText("Recording...")
        self.record_button.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.duration_spin.setEnabled(False)
        
        self.recording_worker = RecordingWorker(duration, device_id)
        self.recording_worker.finished.connect(self.recording_complete)
        self.recording_worker.error.connect(self.recording_error)
        self.recording_worker.start()
        
        self.recording_time = 0
        self.recording_timer.start(1000)  # Update every second
        self.statusBar().showMessage("Recording in progress...")

    def stop_recording(self):
        if hasattr(self, 'recording_worker'):
            self.recording_worker.terminate()
            self.recording_timer.stop()
            self.reset_recording_ui()

    def update_recording_time(self):
        self.recording_time += 1
        remaining = self.duration_spin.value() - self.recording_time
        if remaining >= 0:
            self.statusBar().showMessage(f"Recording... {remaining} seconds remaining")

    def recording_complete(self, audio_data):
        self.current_audio = audio_data
        self.current_duration = len(audio_data) / 16000  # Assuming 16kHz sample rate
        
        file_info = get_file_info(audio_data, 16000)
        info_text = "\n".join([f"{k}: {v}" for k, v in file_info.items()])
        self.file_info_label.setText(info_text)
        
        self.transcribe_button.setEnabled(True)
        self.reset_recording_ui()
        self.statusBar().showMessage("Recording complete")

    def recording_error(self, error_message):
        self.reset_recording_ui()
        QMessageBox.critical(self, "Error", f"Recording failed: {error_message}")
        self.statusBar().showMessage("Recording failed")

    def reset_recording_ui(self):
        self.record_button.setText("Start Recording")
        self.record_button.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.recording_timer.stop()

    def load_model_thread(self):
        try:
            self.model, self.processor = load_model()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

    def model_loaded(self):
        self.setEnabled(True)
        self.statusBar().showMessage("Model loaded successfully")

    def clear_transcription(self):
        if self.text_display.toPlainText():
            reply = QMessageBox.question(
                self,
                "Clear Transcription",
                "Are you sure you want to clear the transcription?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.text_display.clear()

    def start_transcription(self):
        if self.current_audio is None:
            return
        
        self.transcribe_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)  # Set range for determinate progress
        self.progress_bar.setValue(0)
        self.text_display.clear()
        self.statusBar().showMessage("Transcribing...")

        self.worker = TranscriptionWorker(
            self.current_audio,
            self.current_duration,
            self.model,
            self.processor
        )
        self.worker.finished.connect(self.transcription_complete)
        self.worker.error.connect(self.transcription_error)
        self.worker.start()

    def transcription_complete(self, transcriptions):
        self.text_display.setPlainText("\n".join(transcriptions))
        self.progress_bar.setValue(100)
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
        self.transcribe_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.statusBar().showMessage("Transcription complete")

    def transcription_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.load_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Transcription failed: {error_message}")
        self.statusBar().showMessage("Transcription failed")

    def save_transcription(self):
        if not self.text_display.toPlainText():
            QMessageBox.warning(self, "Warning", "No transcription to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            "",
            "Text Files (*.txt);;SRT Files (*.srt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                content = self.text_display.toPlainText()
                if file_path.lower().endswith('.srt'):
                    # Convert to SRT format if needed
                    transcriptions = content.split('\n')
                    content = create_srt(transcriptions)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.statusBar().showMessage(f"Saved to: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
