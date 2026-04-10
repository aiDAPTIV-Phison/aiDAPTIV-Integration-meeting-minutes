<div align="center" style="border-bottom: none">
    <h1>
        <br>
        aiDAPTIV Meetily
        <br>
        Privacy-First AI Meeting Assistant
    </h1>
    <a href="https://github.com/aiDAPTIV-Phison/aiDAPTIV-Integration-meeting-minutes/releases"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"></a>
    <a href="https://github.com/aiDAPTIV-Phison/aiDAPTIV-Integration-meeting-minutes/releases"><img src="https://img.shields.io/badge/Supported_OS-macOS,_Windows-white" alt="Supported OS"></a>
    <br>
    <blockquote>
    <p>This project is a fork of <a href="https://github.com/Zackriya-Solutions/meeting-minutes"><b>Meetily</b></a> by <a href="https://zackriya.com">Zackriya Solutions</a>, used under the <a href="LICENSE.md">MIT License</a>. Modifications and enhancements by aiDAPTIV.</p>
    </blockquote>
</div>

## About

aiDAPTIV Meetily is a privacy-first AI meeting assistant that runs entirely on your local machine. It captures your meetings, transcribes them in real-time, and generates summaries — all without sending any data to the cloud.

## Features

- **Local Processing** — All transcription and summarization runs on your machine. No data leaves your computer.
- **Real-time Transcription** — Live transcript via **Whisper** or **Parakeet** models.
- **AI-Powered Summaries** — Supports **Ollama** (local), Claude, Groq, OpenRouter, and OpenAI.
- **GPU Acceleration** — Metal + CoreML (macOS), CUDA (NVIDIA), Vulkan (AMD/Intel).
- **Multi-Platform** — macOS, Windows, Linux.
- **Open Source** — MIT License, free to use and modify.

## Installation

### Windows

1. Download the latest `x64-setup.exe` from [Releases](https://github.com/aiDAPTIV-Phison/aiDAPTIV-Integration-meeting-minutes/releases/latest)
2. Right-click the downloaded file → **Properties** → Check **Unblock** → Click **OK**
3. Run the installer (if Windows shows a security warning: Click **More info** → **Run anyway**)

### Build from Source (Linux / macOS / Windows)

```bash
git clone https://github.com/aiDAPTIV-Phison/aiDAPTIV-Integration-meeting-minutes
cd aiDAPTIV-Integration-meeting-minutes/frontend
pnpm install
pnpm run tauri:build
```

For detailed build instructions, see the [Building from Source guide](docs/BUILDING.md).

## Architecture

aiDAPTIV Meetily is built with [Tauri](https://tauri.app/) (Rust + Next.js). For details, see the [Architecture documentation](docs/architecture.md).

## License

MIT License — see [LICENSE.md](LICENSE.md).

## Acknowledgments

- Based on [Meetily](https://github.com/Zackriya-Solutions/meeting-minutes) by [Zackriya Solutions](https://zackriya.com) (MIT License).
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp), [Screenpipe](https://github.com/mediar-ai/screenpipe), [transcribe-rs](https://crates.io/crates/transcribe-rs).
- **NVIDIA** Parakeet model and [istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) for ONNX conversion.
