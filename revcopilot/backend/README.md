# RevCopilot Backend

## Features
- Upload a binary and get AI-powered reverse engineering analysis
- Tutor mode: dynamic hints from binary strings/heuristics or AI
- AI mode: Dartmouth LLM integration for insights
- Auto-solve mode: Uses angr to find solutions for CTF-style crackmes

## Requirements
- Python 3.9+
- See `requirements.txt`

## Quickstart

1. **Install dependencies:**

    ```bash
    cd revcopilot/backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run the backend server:**

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

3. **Open the web UI:**

    Go to [http://localhost:8000](http://localhost:8000) in your browser.

4. **Upload a binary:**

    - Click "Upload Binary" and select your file.
    - Choose a mode (Auto, AI, Tutor).
    - (Optional) Enter your Dartmouth API key for AI/Tutor modes.
    - Click "Start Analysis".

5. **View results:**

    - Solution (if found) will be shown.
    - Hints and AI insights are displayed in their respective tabs.

## Notes
- For best results, use CTF-style crackmes or binaries with simple input checks.
- angr may not solve all binaries automatically; see logs for errors.
- For AI/Tutor, a Dartmouth API key is required.

## Troubleshooting
- If file upload or analysis fails, check the browser console and backend logs for errors.
- For angr errors, ensure your Python version is compatible (3.9+ recommended).

---

For more details, see the code in `main.py`.
