# Jarvis: A Hybrid Desktop AI Agent

**A sophisticated, conversational AI desktop assistant designed to streamline your digital life. It combines the power of local on-device tools with remote Google Workspace services, all managed through a clean, intuitive graphical interface.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

*(Note: A placeholder image is used. Please replace this with a real screenshot of your app, ideally showing the new progress panel in action!)*

## Core Features

*   **🚀 Parallel Task Orchestration:** When asked to research multiple US federal dockets, Jarvis bypasses the standard agent loop and activates a dedicated orchestrator. It runs parallel analysis jobs for each docket, displaying live progress in the UI, before synthesizing the results into a single, cohesive briefing.
*   **🧠 Seamless Multi-Step Execution:** Jarvis can handle complex, chained commands (e.g., "research, then create a meeting, then send an email") by completing each stage of the task autonomously without needing to be reminded.
*   **✅ Unified Workspace Management:** Interact with your Google Calendar, Gmail, and Google Drive using natural language.
*   **🔬 ArXiv Research Integration:** Search and download academic papers from ArXiv directly through the agent.
*   **🛠️ Local Tool Integration:** The agent can access local tools to:
    *   Perform real-time web searches using DuckDuckGo.
    *   Analyze US federal dockets via the Regulations.gov API, creating summary PDFs.
    *   Create and manage files in a secure, sandboxed local directory (`~/local_jarvis_sandbox`).
*   **🔒 Secure Credential Management:** All API keys (OpenAI, Google, Regulations.gov, Gemini) are stored securely in your native OS keychain using the `keyring` library. No plaintext secrets are ever stored on disk.
*   **🖥️ Rich, Interactive Interface:** A modern UI built with PyQt6 that renders agent responses in formatted Markdown and includes a dynamic progress panel for parallel tasks.
*   **🔌 Extensible MCP Architecture:** Connects to a `workspace-mcp` server, which manages all Google authentications and tool executions, making the system modular and easy to extend.

## How It Works

Jarvis employs a powerful hybrid architecture that intelligently routes user requests:

1.  **UI & Command Interceptor (PyQt6):** The graphical front-end that you interact with. It does more than just display conversation; it first analyzes your prompt. If it detects a request for a complex, parallel task (like summarizing multiple dockets), it bypasses the agent and directly activates the Task Orchestrator.
2.  **Task Orchestrator:** For multi-docket requests, this dedicated module takes control. It runs multiple `summarize_docket` jobs in parallel using `asyncio`, reports progress back to the UI in real-time, and then uses a final "synthesis" agent to combine all results into a single, cohesive response.
3.  **AI Agent (`openai-agents` SDK):** For all other requests, the "brain" of the operation takes over. It interprets your commands, formulates plans, and decides which tools to use in sequence to achieve your goal.
4.  **Local Tools (Python):** A set of Python functions that the agent and orchestrator can call. These handle tasks on your local machine.
5.  **Remote Tools (MCP Servers):** The agent communicates with two separate MCP servers: one for Google Workspace (`workspace-mcp`) and one for academic papers (`arxiv-mcp-server`). These servers handle all external API interactions, including complex OAuth 2.0 flows, making the system modular and easy to extend.

This architecture ensures that simple tasks are handled efficiently by the agent, while complex, parallelizable tasks are executed by a specialized orchestrator for maximum performance and user feedback, preventing common agent issues like hallucination or getting stuck.

## Getting Started

### Prerequisites

*   Python 3.11+
*   An active virtual environment is highly recommended.
*   `pip` for package management.

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project and its dependencies
# This command also installs the 'arxiv-mcp-server' from the URL specified in pyproject.toml
pip install -e .
```

### 2. Initial Configuration (First Run)

The first time you run the application, it will guide you through setup.

Run the application using the configured entry point:
```bash
local-jarvis
```

On the first run:
- A `config.yaml` file with default settings for the orchestrator will be created automatically in your project directory.
- A series of dialog boxes will appear asking for any missing API keys. You will need:
    *   **OpenAI API Key:** For powering the agent's language model.
    *   **Google Email Address:** The email you will use for Google authentication.
    *   **Regulations.gov API Key:** For the docket analysis tool.
    *   **Google Gemini API Key:** Required by the docket analysis library for summarization.
- Once entered, these keys will be stored securely in your OS keychain.

### 3. Google Authentication (First Tool Use)

The very first time you ask the agent to perform a Google-related task (e.g., "What's on my calendar?"), a dialog box will appear in the UI with a Google authentication link. Follow the standard Google sign-in and consent flow to grant access.

## Usage

1.  **Launch the Application:**
    ```bash
    local-jarvis
    ```
2.  **Start the Server:** In the UI, click the "Start Server" button. Wait for the status to change to "Server: Running".
3.  **Chat with Jarvis:** Begin interacting with the agent in the input field.

### Example Prompts

#### Standard Single-Step & Multi-Step Requests
*   `What's on my calendar for the rest of the week?`
*   `Research the location of Little Sky Bakery.`
*   `Draft an email to my manager with the subject 'Project Update' and a body of 'Here is the latest status report.'`

#### Parallel Orchestration & Multi-Step Chaining
*   **`Jarvis, please prepare a briefing on dockets DOE-HQ-2024-0007, OMB-2024-0004, and BIS-2024-0047. Once that's done, set up a 1-hour meeting with manager@example.com for tomorrow at 2 PM to discuss the findings and send him an email with the summary.`**

When you run the orchestration prompt, you will see:
1.  The progress panel appears, tracking each docket analysis in real time.
2.  Once complete, a synthesized briefing appears in the chat.
3.  Jarvis will **automatically continue** to the next steps, creating the calendar event and sending the email without any further prompts.
4.  You will find the individual PDFs for each docket in your `~/local_jarvis_sandbox` folder.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.