# 📣 RambleOn

**"And now's the time, the time is now..."** – Led Zeppelin

Introducing another random AI project, this time an AI python 🐍 tool to ramble to and still be productive. **RambleOn** is a dictation and AI assistance tool, leveraging OpenAI's **Whisper** 🗣️ for speech-to-text and **LLMs** to clean up your brain-dump 🧠 (or at least make it readable) but also to answer questions. Every AI provider is building AI tools inside their application, but why not have **one AI tool across all applications**.

<img width="332" alt="Screenshot 2025-02-17 at 22 59 50" src="https://github.com/user-attachments/assets/280fdf81-b9aa-4951-a957-d41a331a05b7" />

*Oui, Oui a Fancy GUI of course (only 2 weeks of struggling)*

- ⚡ **Write everywhere** – Sends **keystrokes** to wherever the cursor is.
- 🕵️ Saying **Agent**, ... – Sends the voice command to an LLM
- 👀 Saying **Vision**, ... – Sends the voice command and an image to a vLLM
- 📋 **Clipboard Context** – Clipboard text/image are also send to the LLMs

## 🎥 Demo

I guess I keep on rambling, so here watch this demo.

https://github.com/user-attachments/assets/6b7cb5ee-24c8-4248-82f6-26aa8f48caed


## 👌 Usage

Press and hold the **right Shift key** ⬆️ to start recording ⏺, and release it when you’re done speaking 🗣️. Whisper will transcribe your speech into text ✍️, which is then sent as keyboard strokes and will start typing to ⌨️ wherever the cursor is placed.

If the first word is **Trigger word** like **“agent”** 🕵️ or **“vision”** 👀 an **LLM/vLLM** call is made. Now, the AI response will be inserted at the cursor.
To provide more context, you can copy text beforehand if done **within 10 seconds** of the call, the **clipboard text** will also be included. The vision model will attempt to use a **screenshot** from the clipboard or capture a full-screen image if needed.

## Installation

Simply paste these in a terminal at a preferred location.

```bash
brew install uv
git clone https://github.com/TheoDepr/RambleOn.git
cd RambleOn
uv run rambleon.py
```

#### More Detailed

- Install UV, a python package manager.

```bash
brew install uv
```

- Clone this repo.

```bash
git clone https://github.com/TheoDepr/lisper.git
```

- Run using uv and allow the required permissions

```bash
uv run lisper.py
```

#### Potential issues

If on **linux** "portaudio.h not found" run:

```bash
sudo apt install portaudio19-dev
```

If on **Mac** "portaudio.h not found" run:

```bash
brew install portaudio
```

### ✍️ Add your AI Provider

- Press the **right Shift button** ⬆️ to open the GUI.
- Click on the **Blue circle** 🔵 to open the settings window.
- Go to Provider settings,  and set the **API URL and API KEY**.
- Go to Models, and set the **model name**.

- Personally, I use LMStudio togheter with Tailscale

#### All settings include

- Type of Whisper model
- Trigger words
- LLM/vLLM models
- Provider settings
- Prompts
