# GhostPrompt (CodeSnapAI)

GhostPrompt (CodeSnapAI) is an AI-powered Python assistant for developers. It generates, cleans, and auto-types ready-to-compile Python code directly into your editor or IDE, simulating human typing. It supports both local (Llama/Ollama) and cloud (Gemini 1.5 Flash) LLMs, and offers advanced features like OCR, system tray integration, and customizable hotkeys.

## Features
- **AI Code Generation:** Get Python code suggestions from Llama (Ollama) or Gemini 1.5 Flash.
- **Auto-formatting:** All code is cleaned and formatted to PEP8 using `black`.
- **Auto-typing:** Simulates human typing of code into any focused window.
- **Hotkey Support:** Trigger actions with customizable hotkeys (e.g., Ctrl+Alt+P to auto-type, 'q' to stop typing).
- **OCR Integration:** Extract text/code from screenshots and use as prompt.
- **System Tray:** Minimize to tray, quick access, and background operation.
- **Notifications & TTS:** Optional desktop notifications and text-to-speech.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd code_snapAI
   ```
2. **Install Python 3.8+** (recommended: 3.10+)
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Required packages: PyQt5, pyautogui, keyboard, pytesseract, pillow, requests, pyperclip, black, plyer, pyttsx3, python-dotenv, google-generativeai
4. **Install Tesseract OCR:**
   - Download from: https://github.com/tesseract-ocr/tesseract
   - Set the path in `codesnap_ai.py` if needed.
5. **(Optional) Install Ollama for Llama 3:**
   - See: https://ollama.com/
6. **(Optional) Set up Gemini API key:**
   - Add your key to a `.env` file as `GEMINI_API_KEY=your_key_here`

## Usage
- **Run the app:**
  ```bash
  python codesnap_ai.py
  ```
- **Hotkeys:**
  - `Ctrl+Alt+G`: Show main window
  - `Ctrl+Alt+O`: Toggle overlay
  - `Ctrl+Alt+S`: OCR screen and use as prompt
  - `Ctrl+Alt+C`: Copy code only
  - `Ctrl+Alt+P`: Auto-type last generated code
  - `q`: Stop auto-typing
- **Ask for code:**
  - Type your question in the input box and press Enter or click "Ask AI".
  - The app will try Llama (Ollama) first, then Gemini 1.5 Flash if needed.
- **Auto-type code:**
  - Focus your code editor/IDE, then press `Ctrl+Alt+P` to auto-type the last valid code.
  - Press `q` to stop typing at any time.

## Configuration
- Edit `config.json` for mode and notification preferences.
- Set Tesseract path in `codesnap_ai.py` if not default.
- Place an `icon.png` and `icon.ico` in the project directory for tray/notifications.

## Troubleshooting
- **No code is typed:** Make sure you have a valid code response and the target window is focused.
- **Hotkeys not working:** Run as administrator or check for conflicts with other software.
- **Tesseract errors:** Ensure Tesseract is installed and the path is correct.
- **Ollama/Gemini errors:** Check API/server status and keys.
- **Notifications fail:** Place a valid `icon.ico` in the project directory.

## License
