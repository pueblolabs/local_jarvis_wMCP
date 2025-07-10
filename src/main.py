# src/main.py
import sys
import os
import logging
from dotenv import load_dotenv

from PyQt6.QtWidgets import QApplication, QMessageBox, QInputDialog, QLineEdit

import asyncio
from qasync import QEventLoop

from src.ui.main_window import MainWindow
# The main window now manages the MCPServer, so we don't need to import it here.
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

    # 3. MCP Server Command Configuration
    raw_command = os.getenv("MCP_SERVER_COMMAND", "uvx,workspace-mcp")
    cmd_list = [part.strip() for part in raw_command.split(",")]
    if "--single-user" not in cmd_list:
        cmd_list.append("--single-user")
    config["MCP_SERVER_COMMAND"] = cmd_list
    config["MCP_SERVER_CWD"] = os.getenv("MCP_SERVER_CWD")

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
            mcp_command=config["MCP_SERVER_COMMAND"],
            mcp_cwd=config["MCP_SERVER_CWD"],
        )
        window.show()

    # Run the bootstrap coroutine, then enter the Qt event loop forever
    with loop:
        loop.run_until_complete(_bootstrap())
        loop.run_forever()


if __name__ == "__main__":
    main()