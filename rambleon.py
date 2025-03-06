import sys
import threading
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGraphicsDropShadowEffect,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QTabWidget,
)
from PyQt6.QtGui import QPainter, QColor, QFont, QCursor
from PyQt6.QtCore import Qt, QPoint, QTimer
from pynput import keyboard
import pyaudio
import pyautogui
import wave
from faster_whisper import WhisperModel
import time
import pyperclip
from PIL import ImageGrab
import openai
import base64
from omegaconf import OmegaConf
import logging
import tempfile
import os.path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MARGIN_RIGHT = 10 # pixels
MARGIN_TOP = 70 # pixels
UI_WIDTH = 23 # Number of characters to display in the GUI
CLIPBOARD_TIMEOUT = 15  # seconds
DEFAULT_LANGUAGE = "en" # Default language for language detection
LANGUAGE_THRESHOLD = 0.5 # Minimum probability for language detection
POLLING_INTERVAL = 4  # seconds
CWD = os.path.abspath(os.path.dirname(__file__))  # For running the script directly

# Load the configuration file
try:
    config = OmegaConf.load(os.path.join(CWD, "config.yaml"))
except Exception:
    logging.error("Error loading config.yaml:", exc_info=True)
    sys.exit(1)

# Global variables (for Multithreading Madness)
recording_flag = False  # True while audio is being recorded.
cancelled = False  # Set to True if the Esc key is pressed.
recording_thread = None  # Thread that records audio.
transcription_thread = None  # Thread that transcribes audio.
temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
temp_image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
hotkey = keyboard.Key.shift_r

# Store clipboard content and timestamp
clipboard_text = pyperclip.paste()
clipboard_time = time.time()

trigger_words = {
    "llm": config.settings.trigger_word_llm,
    "vllm": config.settings.trigger_word_vllm,
}

# Load settings from config
settings = config.settings

client_api_key = settings.client_api_key
llm_model = settings.llm_model
vision_model = settings.vision_model
client_base_url = settings.client_base_url
user_prompt_llm = settings.user_prompt_llm
user_prompt_vllm = settings.user_prompt_vllm
system_prompt_llm = settings.system_prompt_llm
system_prompt_vllm = settings.system_prompt_vllm
whisper_model_name = settings.whisper_model_name
extra_language = settings.extra_language
extra_llm_model = settings.extra_llm_model
user_prompt_extra_llm = settings.user_prompt_extra_llm
system_prompt_extra_llm = settings.system_prompt_extra_llm

# Clipboard lock
lock = threading.Lock()

# Initialize the OpenAI client
client = openai.OpenAI(
    base_url=client_base_url,
    api_key=client_api_key,
)

# Initialize the whisper model
whisper_model = WhisperModel(
    model_size_or_path=config.settings.whisper_model_name,
    compute_type="int8",
)


class AssistantGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.statusText = ""  # Status text shown in the window.
        self.statusBool = False  # True if the status text is being updated.
        self.old_pos = None  # For draggable window support.
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(200, 100)

        # Move window to the top right corner of the screen
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        x = screen_geometry.width() - self.width() - MARGIN_RIGHT
        y = MARGIN_TOP
        self.move(x, y)

        # Add settings button
        self.settings_button = QPushButton("", self)
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_button.move(165, 15)
        self.settings_button.setStyleSheet(
            "background-color: blue; border: none; width: 20px; height: 10px; border-radius: 5px;"
        )

        # Add vision symbol
        self.vision_indicator = QPushButton("", self)
        self.vision_indicator.move(135, 15)
        self.vision_indicator.setStyleSheet(
            "border: 1px solid white; width: 19px; height: 9px; border-radius: 5px;"
        )
        self.vision_indicator.hide()

        # Add agent symbol
        self.agent_indicator = QPushButton("", self)
        self.agent_indicator.move(135, 15)
        self.agent_indicator.setStyleSheet(
            "border: 1px solid gold; width: 19px; height: 9px; border-radius: 5px;"
        )
        self.agent_indicator.hide()

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

    def open_settings(self):
        global cancelled
        self.settings_window = SettingsWindow()
        self.settings_window.show()
        cancelled = True

    def show_window(self):
        QTimer.singleShot(0, self.show)

    def show(self):
        cursor_pos = QCursor.pos()
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.move(
            min(
                cursor_pos.x() - self.width() // 2 + 75,
                screen_geometry.width() - self.width(),
            ),
            min(
                cursor_pos.y() - self.height() - 20,
                screen_geometry.height() - self.height(),
            ),
        )
        super().show()

    def hide_window(self):
        QTimer.singleShot(0, self.hide)

    def paintEvent(self, event):
        """Draw a rounded, semi-transparent background,
        a red circle if recording, and the current status text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

        # Draw white border around the rect
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(58, 61, 82))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)

        # Draw red circle if recording
        if recording_flag:
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPoint(20, 20), 8, 8)

        # Draw status text in the center
        if self.statusBool:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(QColor(128, 128, 128))
        font = QFont().defaultFamily()
        painter.setFont(QFont(font, 16))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.statusText)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def updateText(self, text, status=True):
        """Update the displayed status text."""
        self.statusText = text
        self.statusBool = status
        self.update()


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.config = OmegaConf.load(os.path.join(CWD, "config.yaml"))
        self.inputs = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Settings")

        main_layout = QVBoxLayout()

        header_layout = QVBoxLayout()
        header_layout.addWidget(
            self.create_label("RambleOn", True),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        header_layout.addWidget(
            self.create_label("Theo Depr - 1/03/2025 - v1.0"),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        main_layout.addLayout(header_layout)

        tabs = QTabWidget()
        tabs.addTab(self.create_whisper_model_tab(), "Whisper Model")
        tabs.addTab(self.create_trigger_words_tab(), "Trigger Words")
        tabs.addTab(self.create_models_tab(), "Models")
        tabs.addTab(self.create_client_settings_tab(), "Provider settings")
        tabs.addTab(self.create_prompts_tab(), "Prompts")

        main_layout.addWidget(tabs)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 6px; border-radius: 5px;"
        )
        save_button.clicked.connect(self.save_settings)

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(
            "background-color: #d9534f; color: white; padding: 6px; border-radius: 5px;"
        )
        cancel_button.clicked.connect(self.hide)

        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_label(self, text, bold=False):
        """Helper function to create styled QLabel"""
        label = QLabel(text)
        if bold:
            label.setStyleSheet("font-weight: bold; font-size: 14px;")
        else:
            label.setStyleSheet("font-size: 12px; color: gray;")
        return label

    def create_section(self, title, fields):
        """Creates a form-like section with a title"""
        section = QFrame()
        section_layout = QVBoxLayout()
        section_layout.addWidget(self.create_label(title, True))

        form_layout = QFormLayout()

        for label_text, default_value in fields:
            input_field = QLineEdit()
            input_field.setText(default_value)
            input_field.setMinimumWidth(200)
            self.inputs[label_text] = input_field
            form_layout.addRow(QLabel(label_text), input_field)

        section_layout.addLayout(form_layout)
        section.setLayout(section_layout)
        section.setStyleSheet("")
        return section

    def create_whisper_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(
            self.create_section(
                "Whisper Model",
                [
                    ("Model Name:", self.config.settings.whisper_model_name),
                ],
            )
        )
        layout.addWidget(
            self.create_label(
                "Model options: tiny, base, small, medium, large, turbo \nThe model size determines the speed and accuracy of the model."
            )
        )
        tab.setLayout(layout)
        return tab

    def create_trigger_words_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(
            self.create_section(
                "Trigger Words",
                [
                    ("LLM:", self.config.settings.trigger_word_llm),
                    ("VLLM:", self.config.settings.trigger_word_vllm),
                ],
            )
        )
        layout.addWidget(
            self.create_label(
                "Trigger words are used to activate the language and vision models."
            )
        )
        tab.setLayout(layout)
        return tab

    def create_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(
            self.create_section(
                "Models",
                [
                    ("LLM Model:", self.config.settings.llm_model),
                    ("Vision Model:", self.config.settings.vision_model),
                    ("Extra Language:", self.config.settings.extra_language),
                    ("Extra LLM Model:", self.config.settings.extra_llm_model),
                ],
            )
        )
        layout.addWidget(
            self.create_label(
                "Define the language and vision models to use for processing the text. \n Additionally, you can define an extra language with its own model."   
            )
        )
        tab.setLayout(layout)
        return tab

    def create_client_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(
            self.create_section(
                "Client Settings",
                [
                    ("Client Base URL:", self.config.settings.client_base_url),
                    ("Client API Key:", self.config.settings.client_api_key),
                ],
            )
        )
        layout.addWidget(
            self.create_label(
                "Define the provider API URL and API key."
            )
        )
        tab.setLayout(layout)
        return tab

    def create_prompts_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(
            self.create_section(
                "Prompts",
                [
                    ("User Prompt for LLM:", self.config.settings.user_prompt_llm),
                    (
                        "User Prompt for Extra LLM:",
                        self.config.settings.user_prompt_extra_llm,
                    ),
                    ("User Prompt for VLLM:", self.config.settings.user_prompt_vllm),
                    ("System Prompt LLM:", self.config.settings.system_prompt_llm),
                    ("System Prompt VLLM:", self.config.settings.system_prompt_vllm),
                    (
                        "System Prompt for Extra LLM:",
                        self.config.settings.system_prompt_extra_llm,
                    ),
                ],
            )
        )
        layout.addWidget(
            self.create_label(
                "Define the prompts for the language and vision models. \nUse {text} to reference the user input and {clipboard} to reference the clipboard content."
            )
        )
        tab.setLayout(layout)
        return tab

    def save_settings(self):
        """Save settings logic here"""
        global \
            whisper_model, \
            client, \
            trigger_words, \
            llm_model, \
            vision_model, \
            user_prompt_llm, \
            user_prompt_vllm, \
            system_prompt_llm, \
            system_prompt_vllm, \
            extra_language, \
            extra_llm_model, \
            user_prompt_extra_llm, \
            system_prompt_extra_llm
        self.config.settings.trigger_word_llm = self.inputs["LLM:"].text()
        self.config.settings.trigger_word_vllm = self.inputs["VLLM:"].text()
        self.config.settings.llm_model = self.inputs["LLM Model:"].text()
        self.config.settings.vision_model = self.inputs["Vision Model:"].text()
        self.config.settings.client_base_url = self.inputs["Client Base URL:"].text()
        self.config.settings.client_api_key = self.inputs["Client API Key:"].text()
        self.config.settings.whisper_model_name = self.inputs["Model Name:"].text()
        self.config.settings.user_prompt_llm = self.inputs[
            "User Prompt for LLM:"
        ].text()
        self.config.settings.user_prompt_vllm = self.inputs[
            "User Prompt for VLLM:"
        ].text()
        self.config.settings.system_prompt_llm = self.inputs[
            "System Prompt LLM:"
        ].text()
        self.config.settings.system_prompt_vllm = self.inputs[
            "System Prompt VLLM:"
        ].text()
        self.config.settings.extra_language = self.inputs["Extra Language:"].text()
        self.config.settings.extra_llm_model = self.inputs["Extra LLM Model:"].text()
        self.config.settings.user_prompt_extra_llm = self.inputs[
            "User Prompt for Extra LLM:"
        ].text()
        self.config.settings.system_prompt_extra_llm = self.inputs[
            "System Prompt for Extra LLM:"
        ].text()

        trigger_words = {
            "llm": self.config.settings.trigger_word_llm,
            "vllm": self.config.settings.trigger_word_vllm,
        }

        llm_model = self.config.settings.llm_model
        vision_model = self.config.settings.vision_model

        user_prompt_llm = self.config.settings.user_prompt_llm
        user_prompt_vllm = self.config.settings.user_prompt_vllm
        system_prompt_llm = self.config.settings.system_prompt_llm
        system_prompt_vllm = self.config.settings.system_prompt_vllm
        user_prompt_extra_llm = self.config.settings.user_prompt_extra_llm
        system_prompt_extra_llm = self.config.settings.system_prompt_extra_llm

        extra_language = self.config.settings.extra_language
        extra_llm_model = self.config.settings.extra_llm_model

        try:
            client = openai.OpenAI(
                base_url=self.config.settings.client_base_url,
                api_key=self.config.settings.client_api_key,
            )
        except Exception:
            logging.error("Error loading models:", exc_info=True)
            return

        try:
            whisper_model = WhisperModel(
                model_size_or_path=self.config.settings.whisper_model_name,
                compute_type="int8",
            )
        except Exception:
            logging.error("Error loading models:", exc_info=True)
            return

        OmegaConf.save(self.config, os.path.join(CWD, "config.yaml"))
        self.hide()


def record_audio():
    """Record audio and save it to a temporary file."""
    global recording_flag, cancelled, temp_audio_file
    logging.info("Audio: Starting recording...")
    frames = []
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )
    except Exception:
        logging.error("Audio: Error opening stream:", exc_info=True)
        return
    while recording_flag:
        try:
            data = stream.read(1024)
        except Exception:
            logging.error("Audio: Error reading stream:", exc_info=True)
            break
        frames.append(data)
        if cancelled:
            logging.info("Audio: Recording cancelled.")
            break
    stream.stop_stream()
    stream.close()
    audio.terminate()
    logging.info("Audio: Recording stopped, saving to file...")
    try:
        wf = wave.open(temp_audio_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
        wf.close()
        logging.info("Audio: Saved audio to %s", temp_audio_file)
    except Exception:
        logging.error("Audio: Error saving file:", exc_info=True)


def text_to_gui(text):
    """Update the GUI with the given text."""
    global gui, cancelled

    if len(text) <= UI_WIDTH:
        gui.updateText(text)
        if cancelled:
            gui.hide_window()
            return
    else:
        # Scroll the text in the GUI if it's too long
        for i in range(len(text) - UI_WIDTH):
            gui.updateText(text[i : i + UI_WIDTH])
            if cancelled:
                gui.hide_window()
                return
            time.sleep(0.05)


def transcribe_and_send():
    """Transcribe the audio and forward the text"""
    global cancelled, gui, whisper_model
    if cancelled:
        gui.hide_window()
        return

    text = ""
    transcibe_language = DEFAULT_LANGUAGE
    text, info = transcribe_audio()
    transcibe_language = (
        info.language
        if info.language_probability > LANGUAGE_THRESHOLD
        else DEFAULT_LANGUAGE
    )
    text_to_gui(text)

    if cancelled:
        gui.hide_window()
        return

    # Process resulting text
    text_to_output(text, transcibe_language)
    gui.hide_window()


def transcribe_audio():
    """Transcribe the audio file using the whisper model."""
    try:
        logging.info("Transcription: Transcribing audio...")
        segments, info = whisper_model.transcribe(temp_audio_file)
        return "".join(segment.text for segment in segments), info
    except Exception:
        logging.error("Transcription: Error during transcription:", exc_info=True)
        return "", None


def text_to_output(text, transcribe_language):
    """Process the transcribed text and send it to the appropriate model."""
    global gui, cancelled
    if cancelled:
        gui.hide_window()
        return

    # Check if the text is empty
    if text.strip() == "":
        gui.updateText("Please speak")
        return

    # Check if the text is a trigger word
    first_word = text.lower().split()[0]
    if len(text.split()) < 2:
        text_to_keyboard(text)
        return

    text_no_first_word = " ".join(text.split()[1:])

    if cancelled:
        gui.hide_window()
        return

    if trigger_words["llm"] in first_word:
        gui.agent_indicator.show()
        process_agent_command(text_no_first_word, transcribe_language)

    elif trigger_words["vllm"] in first_word:
        gui.vision_indicator.show()
        process_vision_command(text_no_first_word)

    else:
        for i in range(0, len(text), 8):
            text_to_keyboard(text[i : i + 8])
            # Check if the process was cancelled
            if cancelled:
                gui.hide_window()
                return


def clipboard_watcher():
    """ Continuously monitor clipboard for changes in a separate thread. """
    global clipboard_text, clipboard_time
    while True:
        current_content = pyperclip.paste()
        with lock:
            if current_content != clipboard_text:
                clipboard_time = time.time()
                clipboard_text = current_content
        time.sleep(POLLING_INTERVAL)


def get_recent_clipboard():
    """ Get clipboard content if updated within the last CLIPBOARD_TIMEOUT seconds. """
    global clipboard_text, clipboard_time
    with lock:
        if time.time() - clipboard_time <= CLIPBOARD_TIMEOUT:
            return clipboard_text
    return ""


def process_agent_command(text, transcribe_language):
    """Process the text using the language model."""
    global \
        gui, \
        llm_model, \
        extra_language, \
        extra_llm_model \

    # Change the model if the language is different
    if transcribe_language == extra_language:
        model = extra_llm_model
        logging.info("LLM: Using extra language model")
    else:
        model = llm_model
        logging.info("LLM: Using default language model")

    for chunk in llm_call(text=text, clipboard_text=get_recent_clipboard(), model=model):
        if chunk is not None:
            text_to_keyboard(chunk)
        if cancelled:
            gui.agent_indicator.hide()
            gui.hide_window()
            return

    logging.info("LLM responded")
    gui.agent_indicator.hide()


def process_vision_command(text):
    """Process the text using the vision model."""
    global gui, cancelled, vision_model
    take_screenshot()
    logging.info("Vision: Sending to VLLM...")

    for chunk in vision_call(
        text=text,
        image_path=temp_image_file,
        model=vision_model,
    ):
        if chunk is not None:
            text_to_keyboard(chunk)
        if cancelled:
            gui.vision_indicator.hide()
            gui.hide_window()
            return

    logging.info("Vision: VLLM responded")
    gui.vision_indicator.hide()


def llm_call(text: str, clipboard_text: str, model: str):
    """Call the language model with the given text."""
    global client, user_prompt_llm, system_prompt_llm
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt_llm.format(
                        text=text, clipboard=clipboard_text
                    ),
                },
                {"role": "system", "content": system_prompt_llm},
            ],
            stream=True,
        )
    except Exception:
        logging.error("LLM: bad response", exc_info=True)
        return
    for chunk in response:
        if chunk.choices[0].delta is not None:
            yield chunk.choices[0].delta.content


def encode_image(image_path):
    """Encode the image as base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def vision_call(text: str, image_path: str, model: str):
    """Call the vision model with the given text and image
    and yield the response."""
    global client, user_prompt_vllm, system_prompt_vllm
    image_base64 = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt_vllm.format(
                                text=text, clipboard=clipboard_text
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
                {"role": "system", "content": system_prompt_vllm},
            ],
            stream=True,
        )
    except Exception:
        logging.error("Vision: bad response", exc_info=True)
        return

    for chunk in response:
        if chunk.choices[0].delta is not None:
            yield chunk.choices[0].delta.content


def text_to_keyboard(text):
    """Send the text to the keyboard."""
    try:
        if "\n" not in text:
            pyautogui.write(text, interval=0.001)
        else:
            # Handle newlines by sending shift+enter
            lines = text.split("\n")
            for i, line in enumerate(lines):
                pyautogui.write(line, interval=0.001)
                if i < len(lines) - 1:
                    pyautogui.hotkey("shift", "enter")

    except Exception:
        logging.error("Keystroke: Error sending keystrokes:", exc_info=True)


def take_screenshot():
    """Tries to take a screenshot from the clipboard, then the screen."""
    global cancelled, gui, temp_image_file
    try:
        logging.info("Vision: Taking clipboard screenshot")
        screenshot = ImageGrab.grabclipboard()
        screenshot.save(temp_image_file)
        return
    except Exception:
        logging.info("Vision: Taking full screenshot")

    if cancelled:
        gui.hide_window()
        return

    try:
        logging.info("Vision: Taking full screenshot")
        screenshot = ImageGrab.grab(
            all_screens=True,
        )
        screenshot.save(temp_image_file)
        return
    except Exception:
        logging.error("Vision: Error taking screenshot:", exc_info=True)


def on_press(key):
    """Handle key presses."""
    global \
        recording_flag, \
        cancelled, \
        recording_thread, \
        gui, \
        transcription_thread, \
        hotkey

    if key == hotkey and not recording_flag:
        logging.info("Recording started")
        if transcription_thread:
            cancelled = True
            transcription_thread.join()
            transcription_thread = None

        cancelled, recording_flag = False, True
        recording_thread = threading.Thread(target=record_audio, daemon=True)
        recording_thread.start()
        gui.show_window()
        gui.updateText("Listening...")

    if key == keyboard.Key.esc:
        cancelled = True


def on_release(key):
    """Handle key releases."""
    global recording_flag, recording_thread, gui, transcription_thread, hotkey
    if key == hotkey and recording_flag:
        logging.info("Recording stopped")
        recording_flag = False
        gui.updateText("Transcribing...", False)

        if transcription_thread is not None:
            transcription_thread.join()
            transcription_thread = None

        if recording_thread is not None:
            recording_thread.join()

        transcription_thread = threading.Thread(target=transcribe_and_send, daemon=True)
        transcription_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv) # Create the application
    gui = AssistantGUI() # Create the GUI

    threading.Thread(target=clipboard_watcher, daemon=True).start() # Start clipboard watcher
    logging.info("Assistant ready.")
    
    # Start the global keyboard listener on a separate thread.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    sys.exit(app.exec())