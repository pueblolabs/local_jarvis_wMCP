# src/main.py
import sys
import os
import logging
import yaml  # <-- ADD THIS IMPORT
from dotenv import load_dotenv

from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog, QLineEdit

import asyncio
from qasync import QEventLoop

from src.core.multi_agent import create_agent
from src.ui.main_window import MainWindow
from src.utils import secure_store

# Configure logging for better debugging. This will create a log file
# in the same directory where you run the command.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_debug.log", mode='w')
    ]
)

DEFAULT_CONFIG_YAML = """
# config.yaml
# Configuration for Jarvis agent features.
# This file is created automatically if it doesn't exist.

# Settings for the parallel docket orchestrator
docket_orchestrator:
  synthesis_prompt: |
    You are a helpful assistant. You have received several executive summaries from analyses of different US federal dockets.
    Your task is to combine these individual summaries into a single, well-structured, and cohesive briefing for a senior executive.
    Use Markdown for formatting, including headers for each docket ID and bullet points for clarity.
    
    IMPORTANT: Be sure to include the full file path for each generated PDF brief in your final response, so the user knows where to find them.
    
    Do not mention that you are synthesizing multiple reports; simply present the final combined briefing as if it were a single document.
"""

def show_error_and_exit(title: str, message: str):
    """Displays a critical error message box and exits the application."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setText(f"<b>{title}</b>")
    msg_box.setInformativeText(message)
    msg_box.setWindowTitle("Application Error")
    msg_box.exec()
    sys.exit(1)

def load_and_validate_config():
    """
    Loads configuration, interacts with the user for missing values,
    and uses the secure keychain for storage.
    """
    load_dotenv()
    
    # --- ADD THIS BLOCK TO CREATE config.yaml if it doesn't exist ---
    if not os.path.exists('config.yaml'):
        logging.info("config.yaml not found. Creating a default one.")
        with open('config.yaml', 'w') as f:
            f.write(DEFAULT_CONFIG_YAML)
    # --- END OF BLOCK ---
            
    config = {}

    # 1. OpenAI API Key
    api_key = secure_store.get_secret("OPENAI_API_KEY")
    if not api_key:
        key, ok = QInputDialog.getText(
            None, "OpenAI API Key Required",
            "Please enter your OpenAI API Key:", QLineEdit.EchoMode.Password
        )
        if ok and key:
            secure_store.set_secret("OPENAI_API_KEY", key)
            api_key = key
        else:
            # Raising SystemExit is a clean way to abort from a config function
            raise SystemExit("Setup cancelled: An OpenAI API Key is required.")
    config["OPENAI_API_KEY"] = api_key
    os.environ['OPENAI_API_KEY'] = api_key # Required by the agents SDK

    # 2. User Google Email
    # Although the agent doesn't use this directly, our UI needs it for context.
    user_email = secure_store.get_secret("USER_GOOGLE_EMAIL")
    if not user_email:
        email, ok = QInputDialog.getText(
            None, "Google Email Required",
            "Please enter the Google email address you will use for authentication:"
        )
        if ok and email:
            secure_store.set_secret("USER_GOOGLE_EMAIL", email)
            user_email = email
        else:
            raise SystemExit("Setup cancelled: A Google email address is required.")
    config["USER_GOOGLE_EMAIL"] = user_email


    # 4. RegNavigator and Gemini API Keys
    regs_api_key = secure_store.get_secret("REGS_API_KEY")
    if not regs_api_key:
        key, ok = QInputDialog.getText(
            None, "RegNavigator API Key",
            "Please enter your Regulations.gov API Key:", QLineEdit.EchoMode.Password
        )
        if ok and key:
            secure_store.set_secret("REGS_API_KEY", key)
            regs_api_key = key
        else:
            raise SystemExit("Setup cancelled: A RegNavigator API Key is required for the summarize_docket tool.")
    config["REGS_API_KEY"] = regs_api_key
    os.environ['REGS_API_KEY'] = regs_api_key

    gemini_api_key = secure_store.get_secret("GEMINI_API_KEY")
    if not gemini_api_key:
        key, ok = QInputDialog.getText(
            None, "Gemini API Key",
            "Please enter your Google Gemini API Key (for docket analysis):", QLineEdit.EchoMode.Password
        )
        if ok and key:
            secure_store.set_secret("GEMINI_API_KEY", key)
            gemini_api_key = key
        else:
            raise SystemExit("Setup cancelled: A Gemini API Key is required for the summarize_docket tool.")
    config["GEMINI_API_KEY"] = gemini_api_key
    os.environ['GEMINI_API_KEY'] = gemini_api_key

    # 5. MCP Server Command Configuration
    raw_command = os.getenv("MCP_SERVER_COMMAND", "uvx,workspace-mcp")
    cmd_list = [part.strip() for part in raw_command.split(",")]
    if "--single-user" not in cmd_list:
        cmd_list.append("--single-user")
    config["MCP_SERVER_COMMAND"] = cmd_list
    config["MCP_SERVER_CWD"] = os.getenv("MCP_SERVER_CWD")

    # 6. ArXiv MCP Server Configuration
    arxiv_raw_command = os.getenv("ARXIV_MCP_SERVER_COMMAND", "arxiv-mcp-server")
    arxiv_cmd_list = [part.strip() for part in arxiv_raw_command.split(",")]
    config["ARXIV_MCP_SERVER_COMMAND"] = arxiv_cmd_list
    # The storage path will be passed as a command-line argument to the server
    config["ARXIV_MCP_SERVER_STORAGE_PATH"] = os.getenv("ARXIV_MCP_SERVER_STORAGE_PATH")

    logging.info("Configuration loaded and validated successfully.")
    return config

def main() -> None:
    """
    Initializes and runs the Google Workspace Agent desktop application with a
    single asyncio event loop shared between Qt and our async code.
    """
    # Create Qt application
    app = QApplication(sys.argv)

    # Bridge Qt and asyncio via qasync
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    async def _bootstrap() -> None:
        try:
            config = load_and_validate_config()
        except SystemExit as e:
            logging.info(e)
            app.quit()
            return
        except Exception as e:
            show_error_and_exit("Fatal Initialization Error", str(e))
            return

        # Instantiate and show the main window
        window = MainWindow(
            agent_factory=create_agent,
            workspace_mcp_command=config["MCP_SERVER_COMMAND"],
            workspace_mcp_cwd=config["MCP_SERVER_CWD"],
            arxiv_mcp_command=config["ARXIV_MCP_SERVER_COMMAND"],
            arxiv_mcp_storage_path=config["ARXIV_MCP_SERVER_STORAGE_PATH"],
        )
        window.show()

    # Run the bootstrap coroutine, then enter the Qt event loop forever
    with loop:
        loop.run_until_complete(_bootstrap())
        loop.run_forever()


if __name__ == "__main__":
    main()