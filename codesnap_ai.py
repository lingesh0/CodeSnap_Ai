# Minimal PyQt5 test (uncomment to test basic window)
# from PyQt5.QtWidgets import QApplication, QWidget
# app = QApplication([])
# w = QWidget()
# w.show()
# app.exec_()

import sys
import os
import json
import requests
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QSystemTrayIcon, QMenu, QAction, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QClipboard
import keyboard
import pytesseract
from PIL import ImageGrab
import ctypes
import pystray
from PIL import Image
import argparse
import pyperclip
import pyautogui
import time
import textwrap
import re
import ast
import random

# Add this function at the top level
def human_type(text: str, stop_event=None):
    for line in text.splitlines():
        if stop_event and stop_event.is_set():
            print("[⏹️] Typing stopped by user.")
            return
        pyautogui.typewrite(line)
        pyautogui.press('enter')
        time.sleep(random.uniform(0.05, 0.15))  # Slight pause between lines

# Add this function at the top level
def clean_and_format_python_code(raw_code: str) -> str:
    """
    Cleans, dedents, and formats Python code using black for PEP8 compliance.
    Removes triple quotes, stray markdown, and fixes indentation.
    Returns ready-to-compile code.
    """
    # Remove markdown code fences and triple quotes
    code = re.sub(r"```(?:python)?", "", raw_code)
    code = code.replace("```", "")
    code = code.replace('"""', "")
    code = code.replace("'''", "")
    code = code.strip()

    # Dedent and strip each line
    code = textwrap.dedent(code)
    code = "\n".join(line.rstrip() for line in code.splitlines())

    # Optionally, remove leading/trailing single/double quotes if the whole code is quoted
    if (code.startswith("'") and code.endswith("'")) or (code.startswith('"') and code.endswith('"')):
        code = code[1:-1].strip()

    # Format with black if available
    try:
        import black
        code = black.format_str(code, mode=black.Mode())
    except Exception:
        pass  # If black is not installed or code is not valid, skip formatting

    return code.strip()

WDA_EXCLUDEFROMCAPTURE = 0x11

def set_window_display_affinity(hwnd):
    try:
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        print("SetWindowDisplayAffinity applied (ctypes).")
    except Exception as e:
        print("SetWindowDisplayAffinity (ctypes) error:", e)

print("Starting GhostPrompt...")

HOTKEY = 'ctrl+alt+g'      # Main window (restore)
OVERLAY_HOTKEY = 'ctrl+alt+o'  # Overlay
SCREEN_OCR_HOTKEY = 'ctrl+alt+s'  # Full screen OCR
HIDE_WINDOW_HOTKEY = 'ctrl+alt+h'  # Move window off-screen
COPY_HOTKEY = 'ctrl+alt+c'  # Copy chat text
OLLAMA_API_URL = 'http://localhost:11434/api/chat'

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Try to import pywin32 for SetWindowDisplayAffinity
import platform
WIN32_AVAILABLE = platform.system() == 'Windows'
import plyer.platforms.win.notification
from plyer import notification
import pyttsx3

pyautogui.FAILSAFE = False

class GhostOverlay(QWidget):
    def __init__(self, message=""):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Click-through
        self.label = QLabel(message, self)
        self.label.setStyleSheet("color: rgba(0, 255, 0, 220); font-size: 16pt; background: transparent;")
        self.label.move(50, 50)
        self.resize(800, 100)
        self.hide()

    def update_text(self, text):
        self.label.setText(text)
        self.adjustSize()

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

class OllamaWorker(QThread):
    update_text = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, prompt, model="llama3"):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.error = None

    def run(self):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.prompt}
            ],
            "stream": True
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                OLLAMA_API_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    content = data.get("message", {}).get("content", "")
                    buffer += content
                    self.update_text.emit(buffer)
            self.finished.emit(buffer)
        except Exception as e:
            self.error = str(e)
            self.update_text.emit(f"[ERROR] Ollama failed: {e}")
            self.finished.emit("")

class GhostPrompt(QWidget):
    def __init__(self):
        super().__init__()
        self.overlay = GhostOverlay()  # Moved before init_ui to fix AttributeError
        self.init_ui()
        self.is_visible = False
        self.tray_icon = None
        self.ollama_worker = None
        self.latest_response_text = ""  # Store latest AI response
        self.last_code_snippet = None  # Store last code to paste
        self.last_generated_code = ""  # Store last valid code for typing
        self.stop_typing_event = threading.Event()  # Event to stop typing
        self.setup_tray()
        self.hotkey_thread = threading.Thread(target=self.register_hotkeys, daemon=True)
        self.hotkey_thread.start()

    def handle_ai_response(self, response: str):
        """
        Cleans and stores the AI-generated response as code,
        and optionally shows a system notification or feedback.
        """
        cleaned_code = clean_and_format_python_code(response)
        if cleaned_code:
            self.last_generated_code = cleaned_code
            print("[✅] AI response stored for auto-typing.")
        else:
            print("[⚠️] Received empty AI response.")

    def paste_code_like_typing(self):
        self.stop_typing_event.clear()  # Reset before starting
        if self.last_generated_code:
            human_type(self.last_generated_code, stop_event=self.stop_typing_event)
        else:
            print("No valid code stored.")

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.93)
        self.setStyleSheet('background: #23272e; color: #eee; border-radius: 12px;')
        self.setFixedSize(420, 260)

        # Close (X) button
        self.close_btn = QPushButton('✕', self)
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet('''
            QPushButton {
                background: #2c313a;
                color: #eee;
                border: none;
                border-radius: 14px;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #e74c3c;
                color: #fff;
            }
        ''')
        self.close_btn.clicked.connect(self.close)

        # Overlay toggle button
        self.overlay_btn = QPushButton('Show/Hide Overlay', self)
        self.overlay_btn.setStyleSheet('background: #222; color: #0f0; border: 1px solid #0f0; padding: 6px; border-radius: 6px;')
        self.overlay_btn.clicked.connect(self.overlay.toggle_visibility)

        # Top bar layout for close button
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        top_bar.addWidget(self.close_btn)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(0)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText('Ask anything...')
        self.input_box.setStyleSheet('background: #2c313a; color: #eee; border-radius: 6px; padding: 8px;')
        self.input_box.returnPressed.connect(self.ask_ai)

        self.ask_btn = QPushButton('Ask AI', self)
        self.ask_btn.setStyleSheet('background: #4A90E2; color: white; border: none; padding: 10px; border-radius: 6px;')
        self.ask_btn.clicked.connect(self.ask_ai)

        self.response_area = QTextEdit(self)
        self.response_area.setReadOnly(True)
        self.response_area.setStyleSheet('background: #23272e; color: #eee; border-radius: 6px;')

        self.copy_btn = QPushButton('Copy', self)
        self.copy_btn.setStyleSheet('background: #888; color: white; border: none; padding: 8px; border-radius: 6px;')
        self.copy_btn.clicked.connect(self.copy_response)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.input_box)
        layout.addWidget(self.ask_btn)
        layout.addWidget(self.overlay_btn)
        layout.addWidget(self.response_area)
        layout.addWidget(self.copy_btn)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

    def setup_tray(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.setWindowIcon(icon)
        self.tray_icon.setToolTip('GhostPrompt')
        menu = QMenu()
        show_action = QAction('Show', self)
        show_action.triggered.connect(self.show_window)
        hide_action = QAction('Hide', self)
        hide_action.triggered.connect(self.hide_window)
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(show_action)
        menu.addAction(hide_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()
        print("Tray icon set up.")

    def register_hotkeys(self):
        print(f"Registering hotkey: {HOTKEY}")
        keyboard.add_hotkey(HOTKEY, lambda: QTimer.singleShot(0, self.restore_window_on_screen))
        print(f"Registering overlay hotkey: {OVERLAY_HOTKEY}")
        keyboard.add_hotkey(OVERLAY_HOTKEY, lambda: QTimer.singleShot(0, self.overlay.toggle_visibility))
        print(f"Registering screen OCR hotkey: {SCREEN_OCR_HOTKEY}")
        keyboard.add_hotkey(SCREEN_OCR_HOTKEY, lambda: QTimer.singleShot(0, self.screen_ocr_hotkey))
        print(f"Registering hide window hotkey: {HIDE_WINDOW_HOTKEY}")
        keyboard.add_hotkey(HIDE_WINDOW_HOTKEY, lambda: QTimer.singleShot(0, self.move_window_off_screen))
        print(f"Registering copy hotkey: {COPY_HOTKEY}")
        keyboard.add_hotkey(COPY_HOTKEY, lambda: QTimer.singleShot(0, self.copy_response))
        # Register new hotkeys for code-only copy and auto-paste
        print("Registering code-only copy hotkey: ctrl+alt+c")
        keyboard.add_hotkey('ctrl+alt+c', lambda: QTimer.singleShot(0, self.copy_code_only))
        print("Registering code auto-paste hotkey: ctrl+alt+p")
        keyboard.add_hotkey('ctrl+alt+p', lambda: QTimer.singleShot(0, self.paste_code_like_typing), suppress=True)
        print("Registering stop typing hotkey: q")
        keyboard.add_hotkey('q', lambda: self.stop_typing_event.set(), suppress=True)
        keyboard.wait()

    def move_window_off_screen(self):
        print("Moving window off-screen")
        self.move(-2000, -2000)
        self.hide()

    def restore_window_on_screen(self):
        print("Restoring window to screen")
        # Move to a visible position (e.g., top-left corner)
        self.move(100, 100)
        self.show_window()
        # Set display affinity if available
        if WIN32_AVAILABLE:
            try:
                hwnd = int(self.winId())
                set_window_display_affinity(hwnd)
                print("SetWindowDisplayAffinity applied.")
            except Exception as e:
                print("SetWindowDisplayAffinity error:", e)
        else:
            print("pywin32 not available, cannot set display affinity.")

    def setup_pystray(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        image = Image.open(icon_path) if os.path.exists(icon_path) else Image.new('RGB', (64,64))
        menu = pystray.Menu(
            pystray.MenuItem('Show', self.show_window),
            pystray.MenuItem('Hide', self.hide_window),
            pystray.MenuItem('Exit', self.quit_app)
        )
        self.tray = pystray.Icon("GhostPrompt", image, menu=menu)
        threading.Thread(target=self.tray.run, daemon=True).start()

    def show_window(self):
        self.show()
        self.activateWindow()

    def hide_window(self):
        self.move(-10000, -10000)
        set_window_display_affinity(self.winId())

    def quit_app(self):
        self.tray.stop()
        QApplication.quit()
        self.hide()  # Start hidden
        self.is_visible = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide_window()

    def show_notification(self, message):
        notification.notify(
            title='GhostPrompt',
            message=message,
            app_icon=os.path.join(os.path.dirname(__file__), "icon.ico")
        )

    def speak(self, text):
        def tts_thread():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        threading.Thread(target=tts_thread, daemon=True).start()

    def ask_ai(self):
        prompt = self.input_box.text().strip()
        if not prompt:
            self.response_area.setPlainText('Please enter a question.')
            return
        # Modify prompt to request only code output
        code_prompt = (
            f"{prompt}\n\n"
            "Please output only the code (no explanation, no markdown formatting). "
            "Infer the input and output format from the question if possible. "
            "Return ready-to-compile, PEP8-compliant Python code."
        )
        self.response_area.setPlainText('Thinking... (Llama 3, then Gemini 1.5 Flash fallback)')
        self.overlay.update_text('Thinking... (Llama 3, then Gemini 1.5 Flash fallback)')
        # Try Ollama first, then Gemini if error
        def on_llama_finished(content):
            if content.startswith("[ERROR] Ollama failed:") or not content.strip():
                self.ask_gemini(prompt)
            else:
                self.update_answer_box(f"[Llama 3 - Local]\n\n{content}")
        self.ollama_worker = OllamaWorker(code_prompt)
        self.ollama_worker.update_text.connect(lambda text: self.response_area.setPlainText(f"[Llama 3 - Local]\n\n{text}"))
        self.ollama_worker.finished.connect(on_llama_finished)
        self.ollama_worker.start()

    def update_answer_box(self, content):
        self.latest_response_text = content  # Store latest response for code copy
        self.handle_ai_response(content)
        if content.startswith("[ERROR] Ollama failed:"):
            self.response_area.setPlainText(content)
            self.overlay.update_text(content)
        else:
            self.response_area.setPlainText(f"[Llama 3 - Local]\n\n{content}")
            self.overlay.update_text(content)
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )

    def on_answer_complete(self, content):
        self.handle_ai_response(content)
        # If Ollama failed, fallback to Gemini
        if content.startswith("[ERROR] Ollama failed:"):
            self.ask_gemini(self.input_box.text().strip())

    def ask_gemini(self, prompt):
        # Modify prompt to request only code output
        code_prompt = (
            f"{prompt}\n\n"
            "Please output only the code (no explanation, no markdown formatting). "
            "Infer the input and output format from the question if possible. "
            "Return ready-to-compile, PEP8-compliant Python code."
        )
        self.response_area.setPlainText('Thinking... (Gemini 1.5 Flash)')
        self.overlay.update_text('Thinking... (Gemini 1.5 Flash)')
        def gemini_thread():
            try:
                import google.generativeai as genai
                from dotenv import load_dotenv
                load_dotenv()
                GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
                if not GEMINI_API_KEY:
                    GEMINI_API_KEY = "AIzaSyBIp7DNNfyAFxUQKiycBdRadMfvDZ06wmU"
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(code_prompt)
                try:
                    answer = response.text.strip()
                except AttributeError:
                    answer = response.candidates[0].content.parts[0].text.strip()
                QTimer.singleShot(0, lambda: self.update_answer_box(f"[Gemini 1.5 Flash - Cloud]\n\n{answer}"))
            except Exception as e:
                answer = f'Error: {e}\nNeither Ollama nor Gemini is available.'
                QTimer.singleShot(0, lambda: self.update_answer_box(answer))
        threading.Thread(target=gemini_thread, daemon=True).start()

    def screen_ocr_hotkey(self):
        # Hide windows for clean capture
        self.hide()
        self.overlay.hide()
        QTimer.singleShot(300, self.capture_and_ocr_fullscreen)

    def capture_and_ocr_fullscreen(self):
        # Capture full screen
        img = ImageGrab.grab()
        # Run OCR
        text = pytesseract.image_to_string(img)
        print("OCR extracted text:", text)
        if text.strip():
            self.input_box.setText(text.strip())
            self.ask_ai()
        else:
            self.response_area.setPlainText('OCR did not extract any text.')
            self.overlay.update_text('OCR did not extract any text.')
        # Optionally, show the window again
        self.show()

    def copy_response(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.response_area.toPlainText())

    def copy_code_only(self):
        response_text = self.response_area.toPlainText()
        code_start = response_text.find("```")
        code_end = response_text.rfind("```")
        if code_start != -1 and code_end != -1 and code_start != code_end:
            code = response_text[code_start+3:code_end].strip()
        else:
            code = response_text.strip()
        pyperclip.copy(code)
        self.last_code_snippet = self.extract_code_from_response(response_text)  # Save for paste
        print("✅ Code copied to clipboard.")

    def extract_code_from_response(self, response):
        # Extract code blocks between ```...``` OR take the whole content
        matches = re.findall(r"```(?:python)?\n([\s\S]+?)```", response)
        if matches:
            raw_code = matches[0]
        else:
            raw_code = response.strip()
        # Remove Python-style triple quotes if present
        raw_code = re.sub(r'"""', '', raw_code)
        # Remove single or double quotes that wrap the entire code
        raw_code = raw_code.strip()
        if (raw_code.startswith("'") and raw_code.endswith("'")) or (raw_code.startswith('"') and raw_code.endswith('"')):
            raw_code = raw_code[1:-1]
        # Detect and warn about code-as-string logic (e.g., return 'return ...')
        code_as_string_pattern = r'return\s+["\"][\s\S]*return[\s\S]*["\"]'
        if re.search(code_as_string_pattern, raw_code):
            print("[CodeSnapAI WARNING] Detected a return statement returning a string of code. Please review the logic: do not return code as a string.")
            # Remove such lines
            raw_code = re.sub(r'return\s+["\"][\s\S]*return[\s\S]*["\"]', '', raw_code)
        # Normalize indentation and remove any markdown artifacts
        cleaned_code = textwrap.dedent(raw_code).strip()
        # Strip leading/trailing whitespace from each line
        cleaned_code = "\n".join(line.rstrip() for line in cleaned_code.splitlines())
        # Replace tabs with 4 spaces for consistent indentation
        cleaned_code = cleaned_code.replace('\t', '    ')
        # Strip illegal characters (except common code symbols)
        cleaned_code = re.sub(r'[^a-zA-Z0-9_ ()\[\]{}:\n=+\-*/.,\'\"#<>%!?\\]', '', cleaned_code)
        # Optionally, try to reformat Python code using ast and black (if available)
        try:
            import black
            tree = ast.parse(cleaned_code)
            cleaned_code = black.format_str(cleaned_code, mode=black.Mode())
        except Exception:
            pass  # If not valid Python or black not installed, skip
        # Simple code review: warn if common built-in names are used as variables
        builtins = {'str', 'list', 'dict', 'set', 'int', 'float', 'bool', 'input', 'print', 'type', 'id', 'sum', 'min', 'max', 'any', 'all', 'map', 'filter', 'zip', 'open', 'file', 'object'}
        for builtin in builtins:
            pattern = rf'\b{builtin}\b'
            if re.search(pattern, cleaned_code):
                print(f"[CodeSnapAI WARNING] Variable name '{builtin}' detected. Consider renaming to avoid shadowing built-ins.")
        return cleaned_code

# Add this function at the top level
def clean_code_output(raw_text):
    lines = raw_text.strip().splitlines()
    cleaned = "\n".join(line for line in lines if not line.strip().startswith("#"))
    if "```" in cleaned:
        cleaned = cleaned.replace("```python", "").replace("```", "")
    # Remove leading/trailing quotes
    cleaned = cleaned.strip()
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1]
    return cleaned.strip()

def is_valid_python_code(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

# Restore missing functions/classes here
CONFIG_DEFAULT = {
    "mode": "gui",  # gui, cli, tray
    "show_popups": True,
    "use_notification": True,
    "use_tts": True
}

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**CONFIG_DEFAULT, **config}
        except Exception as e:
            print("Failed to load config.json:", e)
    return CONFIG_DEFAULT.copy()

# --- Notification/TTS helpers (reusable) ---
def show_notification(message, config=None):
    if config is None or config.get("use_notification", True):
        try:
            notification.notify(
                title='GhostPrompt',
                message=message,
                app_icon=os.path.join(os.path.dirname(__file__), "icon.ico")
            )
        except Exception as e:
            print("Notification error:", e)

def speak(text, config=None):
    if config is None or config.get("use_tts", True):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

class GhostPromptCLI:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.register_hotkeys()
        print("[GhostPrompt CLI] Ready. Press the hotkey or type 'exit' to quit.")
        self.input_loop()

    def register_hotkeys(self):
        print(f"Registering CLI hotkey: {HOTKEY}")
        keyboard.add_hotkey(HOTKEY, self.ask_ai_hotkey)

    def ask_ai_hotkey(self):
        print("\n[Hotkey Triggered]")
        self.ask_ai()

    def ask_ai(self):
        prompt = input("Ask anything: ").strip()
        if not prompt:
            print("Please enter a question.")
            return
        print("Thinking...")
        # Modify prompt to request only code output
        code_prompt = (
            f"{prompt}\n\n"
            "Please output only the code (no explanation, no markdown formatting). "
            "Infer the input and output format from the question if possible."
        )
        answer = self.ask_ollama(code_prompt)
        if not answer or answer.startswith("[ERROR]"):
            answer = self.ask_gemini(code_prompt)
        print("\n[Answer]\n" + answer)
        if self.config.get("use_notification", True):
            show_notification(answer, self.config)
        if self.config.get("use_tts", True):
            speak(answer, self.config)

    def ask_ollama(self, prompt):
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                OLLAMA_API_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            return f"[ERROR] Ollama failed: {e}"

    def ask_gemini(self, prompt):
        # Modify prompt to request only code output
        code_prompt = (
            f"{prompt}\n\n"
            "Please output only the code (no explanation, no markdown formatting). "
            "Infer the input and output format from the question if possible."
        )
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            if not GEMINI_API_KEY:
                GEMINI_API_KEY = "AIzaSyBIp7DNNfyAFxUQKiycBdRadMfvDZ06wmU"
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(code_prompt)
            try:
                answer = response.text.strip()
            except AttributeError:
                answer = response.candidates[0].content.parts[0].text.strip()
            return answer
        except Exception as e:
            return f'Error: {e}\nNeither Ollama nor Gemini is available.'

    def input_loop(self):
        while self.running:
            cmd = input("Type 'ask' to query, or 'exit' to quit: ").strip().lower()
            if cmd == 'exit':
                self.running = False
            elif cmd == 'ask':
                self.ask_ai()
            else:
                print("Unknown command. Type 'ask' or 'exit'.")

def run_pystray_mode(config):
    def on_show(icon, item):
        print("[Tray] Show selected (no GUI in tray mode)")
    def on_ask(icon, item):
        prompt = input("[Tray] Ask anything: ").strip()
        if not prompt:
            print("Please enter a question.")
            return
        print("Thinking...")
        answer = GhostPromptCLI(config).ask_ollama(prompt)
        if not answer or answer.startswith("[ERROR]"):
            answer = GhostPromptCLI(config).ask_gemini(prompt)
        print("\n[Answer]\n" + answer)
        if config.get("use_notification", True):
            show_notification(answer, config)
        if config.get("use_tts", True):
            speak(answer, config)
    def on_exit(icon, item):
        print("[Tray] Exiting...")
        icon.stop()
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    image = Image.open(icon_path) if os.path.exists(icon_path) else Image.new('RGB', (64,64))
    menu = pystray.Menu(
        pystray.MenuItem('Ask', on_ask),
        pystray.MenuItem('Exit', on_exit)
    )
    tray = pystray.Icon("GhostPrompt", image, menu=menu)
    tray.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GhostPrompt Invisible Assistant')
    parser.add_argument('--mode', choices=['gui', 'cli', 'tray'], help='Run mode: gui, cli, or tray')
    args = parser.parse_args()
    config = load_config()
    if args.mode:
        config['mode'] = args.mode
    mode = config.get('mode', 'gui')
    if mode == 'cli':
        GhostPromptCLI(config)
    elif mode == 'tray':
        run_pystray_mode(config)
    else:
        print("Launching QApplication...")
        app = QApplication(sys.argv)
        window = GhostPrompt()
        window.hide()  # Start hidden
        print("App running. Waiting for hotkey or tray interaction...")
        sys.exit(app.exec_())