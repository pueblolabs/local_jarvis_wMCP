# src/ui/main_window.py
"""
Main application window for the Jarvis Desktop Agent.

This module defines the main graphical user interface (GUI) using PyQt6 and qasync.
It provides a chat-like interface for interacting with an AI agent,
controls for managing the backend server, and a log view for diagnostics.
"""

import sys
import asyncio
import re
import html
import logging
import platform
import os
import warnings
from enum import Enum, auto
from typing import Optional, List, Dict, Any

# Suppress harmless Pydantic warnings that can clutter the console.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --- Qt Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser, QLineEdit,
    QPushButton, QStatusBar, QHBoxLayout, QTextEdit, QLabel, QFrame,
    QMessageBox, QStyle
)
from PyQt6.QtCore import QUrl, pyqtSlot
from PyQt6.QtGui import QIcon, QPalette, QColor, QDesktopServices
import pyperclip
from qasync import asyncSlot

# --- Linkification Helpers -------------------------------------------------
URL_RE = re.compile(r'(https?://[^\s<>"]+)')
MD_LINK_RE = re.compile(r'\[([^\]]+)]\((https?://[^\s)]+)\)')

def linkify(text: str, shorten: bool = True) -> str:
    """Convert bare URLs and Markdown-style links into HTML anchor tags.

    Bare links are shortened to keep the UI tidy; markdown links preserve
    their provided title.
    """
    # Markdown links first: [title](url)
    def _md(match):
        title, url = match.groups()
        return f'<a href="{html.escape(url)}">{html.escape(title)}</a>'
    text = MD_LINK_RE.sub(_md, text)

    # Bare URLs
    def _url(match):
        url = match.group(0)
        display = url
        if shorten and len(url) > 45:
            display = url[:25] + "…" + url[-12:]
        return f'<a href="{html.escape(url)}">{html.escape(display)}</a>'
    return URL_RE.sub(_url, text)

# --- Application-specific Imports ---
from agents import Runner, RunResult
from agents.mcp import MCPServerStdio
from src.core.workspace_agent import create_workspace_agent

logger = logging.getLogger(__name__)

# --- UI Constants & Theming ---

class UIColors:
    """Centralized color palette for the UI."""
    BACKGROUND = "#2B2B2B"
    BACKGROUND_LIGHT = "#3C3F41"
    FOREGROUND = "#A9B7C6"
    PRIMARY = "#4A90E2"
    PRIMARY_LIGHT = "#64A4E8"
    SUCCESS = "#5FAD56"
    ERROR = "#D9534F"
    WARNING = "#F0AD4E"
    USER_MSG = "#87CEEB"  # Sky Blue
    AGENT_MSG = "#E0E0E0"

STYLESHEET = f"""
    QMainWindow, QWidget {{
        background-color: {UIColors.BACKGROUND};
        color: {UIColors.FOREGROUND};
        font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif;
        font-size: 14px;
    }}
    QFrame {{
        background-color: {UIColors.BACKGROUND_LIGHT};
        border-radius: 8px;
    }}
    QLabel {{
        background-color: transparent;
    }}
    QLabel#TitleLabel {{
        font-size: 18px;
        font-weight: bold;
        padding: 5px;
    }}
    QPushButton {{
        background-color: {UIColors.PRIMARY};
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {UIColors.PRIMARY_LIGHT};
    }}
    QPushButton:disabled {{
        background-color: #555555;
        color: #888888;
    }}
    QLineEdit, QTextEdit, QTextBrowser {{
        background-color: {UIColors.BACKGROUND};
        border: 1px solid {UIColors.BACKGROUND_LIGHT};
        border-radius: 5px;
        padding: 8px;
        color: {UIColors.FOREGROUND};
    }}
    QLineEdit:focus, QTextEdit:focus {{
        border: 1px solid {UIColors.PRIMARY};
    }}
    QStatusBar {{
        font-size: 12px;
    }}
    QScrollBar:vertical {{
        border: none;
        background: {UIColors.BACKGROUND};
        width: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {UIColors.PRIMARY};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
"""

class ServerState(Enum):
    """Represents the possible states of the MCP server."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()

# --- Main Application Window ---

class MainWindow(QMainWindow):
    """The main window of the Jarvis Desktop Agent application."""

    def __init__(self, mcp_command: list, mcp_cwd: Optional[str]):
        super().__init__()
        self.mcp_command = mcp_command
        self.mcp_cwd = mcp_cwd

        self._setup_agent()
        self._setup_ui()
        self._connect_signals()

        self.update_server_status(ServerState.STOPPED)
        self.append_conversation(
            f"<b style='color: {UIColors.PRIMARY};'>Jarvis</b>: "
            "Hello! Please start the server to begin."
        )

    def _setup_agent(self):
        """Initializes the agent and MCP server connection."""
        self.mcp_server = MCPServerStdio(
            params={
                "command": self.mcp_command[0],
                "args": self.mcp_command[1:],
                "cwd": self.mcp_cwd,
                "env": {**os.environ},
            }
        )
        self.agent = create_workspace_agent(mcp_server=self.mcp_server)
        self.conversation_history: List[Dict[str, Any]] = []

    def _setup_ui(self):
        """Sets up all UI elements, layouts, and styles."""
        self.setWindowTitle("Jarvis Desktop Agent")
        self.setMinimumSize(800, 600)
        self._set_initial_size()
        self.setStyleSheet(STYLESHEET)

        # Create Widgets
        self._create_widgets()

        # Create Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_panel, right_panel = self._create_panels()
        main_layout.addWidget(left_panel, 3)  # Give conversation more space
        main_layout.addWidget(right_panel, 1)

    def _create_widgets(self):
        """Instantiates all widgets used in the UI."""
        # Common Icons
        style = self.style()
        self.start_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.stop_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        self.send_icon = style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward)

        # Conversation Panel
        self.conversation_view = QTextBrowser()
        self.conversation_view.setOpenExternalLinks(True)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message to the agent...")
        self.send_button = QPushButton()
        self.send_button.setIcon(self.send_icon)

        # Server Control Panel
        self.start_button = QPushButton("Start Server")
        self.start_button.setIcon(self.start_icon)
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.setIcon(self.stop_icon)

        self.server_log_view = QTextEdit()
        self.server_log_view.setReadOnly(True)
        font_family = "Menlo" if platform.system() == "Darwin" else "Consolas" if platform.system() == "Windows" else "Monospace"
        self.server_log_view.setFontFamily(font_family)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.server_status_label = QLabel()
        self.status_bar.addPermanentWidget(self.server_status_label)

    def _create_panels(self) -> (QFrame, QFrame):
        """Creates and layouts the left and right panels."""
        # --- Left (Conversation) Panel ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        
        title_left = QLabel("Conversation")
        title_left.setObjectName("TitleLabel")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        left_layout.addWidget(title_left)
        left_layout.addWidget(self.conversation_view, 1)
        left_layout.addLayout(input_layout)

        # --- Right (Control) Panel ---
        right_panel = QFrame()
        right_panel.setFixedWidth(350)
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)

        title_right = QLabel("Server Control")
        title_right.setObjectName("TitleLabel")

        server_buttons = QHBoxLayout()
        server_buttons.addWidget(self.start_button)
        server_buttons.addWidget(self.stop_button)

        right_layout.addWidget(title_right)
        right_layout.addLayout(server_buttons)
        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel("Server Log:"))
        right_layout.addWidget(self.server_log_view, 1)
        
        return left_panel, right_panel

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self.send_button.clicked.connect(self.on_send_command)
        self.input_field.returnPressed.connect(self.on_send_command)
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.stop_server)

    def _set_initial_size(self):
        """Sets the initial size of the window based on screen dimensions."""
        screen = self.screen().availableGeometry()
        self.resize(int(screen.width() * 0.7), int(screen.height() * 0.75))

    # --- Asynchronous Slots & Core Logic ---

    @asyncSlot()
    async def start_server(self):
        """Asynchronously starts the MCP server."""
        self.update_server_status(ServerState.STARTING)
        self.append_server_log("Attempting to connect to MCP server...")
        try:
            await self.mcp_server.connect()
            self.update_server_status(ServerState.RUNNING)
            self.append_server_log("INFO: MCP Server connection successful.")
            self.append_conversation(
                f"<b style='color: {UIColors.PRIMARY};'>Jarvis</b>: "
                "Server connected. What can I do for you today?"
            )
        except Exception as e:
            self.update_server_status(ServerState.ERROR)
            self.append_server_log(f"ERROR: {e}")
            logger.error("Failed to start MCP server.", exc_info=True)

    @asyncSlot()
    async def stop_server(self):
        """Asynchronously stops the MCP server."""
        if self.mcp_server.is_running():
            self.update_server_status(ServerState.STOPPING)
            self.append_server_log("Attempting to clean up MCP server connection...")
            try:
                await self.mcp_server.cleanup()
            except Exception as e:
                logger.error("Error during MCP server cleanup.", exc_info=True)
                self.append_server_log(f"WARN: Error during cleanup: {e}")
            finally:
                self.update_server_status(ServerState.STOPPED)
                self.append_server_log("INFO: MCP Server cleanup complete.")
        else:
            self.update_server_status(ServerState.STOPPED)


    @pyqtSlot()
    def on_send_command(self):
        """Handles the send button click or enter press in the input field."""
        user_text = self.input_field.text().strip()
        if not user_text:
            return
        # Create a task to run the async handler without blocking the GUI
        asyncio.create_task(self._handle_send_command(user_text))

    async def _handle_send_command(self, user_text: str):
        """The core asynchronous logic for processing a user command."""
        self.append_conversation(f"<p style='color: {UIColors.USER_MSG};'><b>You</b>: {user_text}</p>")
        self._set_input_enabled(False)

        try:
            input_list = self.conversation_history + [{"role": "user", "content": user_text}]
            result: RunResult = await Runner.run(self.agent, input_list)
            self.conversation_history = result.to_input_list()
            final_output = str(result.final_output)

            auth_url_match = re.search(r'(https?://accounts.google.com/o/oauth2/auth\S+)', final_output)
            if auth_url_match:
                auth_url = auth_url_match.group(1).strip()
                self._handle_auth_url(auth_url)
            else:
                final_output_html = linkify(final_output)
                self.append_conversation(
                    f"<p style='color: {UIColors.AGENT_MSG};'><b>Jarvis</b>: {final_output_html}</p>"
                )

        except Exception as e:
            friendly_error = self._format_error(str(e))
            self.append_conversation(
                f"<p style='background-color: {UIColors.ERROR}20; padding: 10px; border-radius: 5px;'>"
                f"<b style='color: {UIColors.ERROR};'>⚠️ Agent Error</b>: {friendly_error}</p>"
            )
            logger.error("Agent execution failed.", exc_info=True)
        finally:
            self._set_input_enabled(True)

    # --- UI State and Helpers ---

    def update_server_status(self, state: ServerState):
        """Updates the UI based on the server's state."""
        status_map = {
            ServerState.STOPPED: ("Server: Stopped", UIColors.FOREGROUND, False),
            ServerState.STARTING: ("Server: Starting...", UIColors.WARNING, False),
            ServerState.RUNNING: ("Server: Running", UIColors.SUCCESS, True),
            ServerState.STOPPING: ("Server: Stopping...", UIColors.WARNING, False),
            ServerState.ERROR: ("Server: Error", UIColors.ERROR, False),
        }
        
        text, color_hex, is_running = status_map[state]
        self.server_status_label.setText(text)
        
        palette = self.server_status_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor(color_hex))
        self.server_status_label.setPalette(palette)

        self.start_button.setEnabled(state in [ServerState.STOPPED, ServerState.ERROR])
        self.stop_button.setEnabled(state == ServerState.RUNNING)
        self._set_input_enabled(is_running)
        if is_running:
            self.input_field.setFocus()

    def _set_input_enabled(self, enabled: bool):
        """Enables or disables the user input field and send button."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        if enabled:
            self.input_field.clear()
            self.input_field.setFocus()

    def _append_and_scroll(self, widget: QTextBrowser | QTextEdit, text: str):
        """Appends text to a widget and scrolls to the bottom."""
        widget.append(text)
        scrollbar = widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def append_conversation(self, html: str):
        """Appends a message to the conversation view."""
        self._append_and_scroll(self.conversation_view, html)
        
    def append_server_log(self, text: str):
        """Appends a message to the server log view."""
        self._append_and_scroll(self.server_log_view, text)

    def _format_error(self, error_str: str) -> str:
        """Converts a technical error string into a user-friendly message."""
        error_lower = error_str.lower()
        if "timed out" in error_lower:
            return "The request to the server timed out. It might be busy or unresponsive."
        if "connectionerror" in error_lower:
            return "Could not connect to the tool server. Please ensure it's running correctly."
        if "badrequesterror" in error_lower:
            return "There was an issue communicating with the AI model. Check the debug log."
        return "An unexpected error occurred. Check the `agent_debug.log` file for details."

    def _handle_auth_url(self, url: str):
        """Displays a dialog box for handling Google Authentication URLs."""
        self.append_conversation(
            f"<p style='background-color: {UIColors.WARNING}20; padding: 10px; border-radius: 5px;'>"
            f"<b style='color: {UIColors.WARNING};'>Authentication Required</b>: I need you to authenticate with Google. "
            "Please follow the instructions in the pop‑up window or your browser, then try your command again.</p>"
        )
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Google Authentication Required")
        msg_box.setText(
            "To continue, please sign in with Google.\n\n"
            "A browser window can be opened for you. After logging in, return here and retry your command."
        )
        msg_box.setDetailedText(url)
        open_button = msg_box.addButton("Open Browser", QMessageBox.ButtonRole.ActionRole)
        copy_button = msg_box.addButton("Copy URL", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Cancel)

        open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
        copy_button.clicked.connect(lambda: pyperclip.copy(url))
        msg_box.exec()

    def closeEvent(self, event):
        """Ensures the server is stopped when the window is closed."""
        self.append_server_log("INFO: Main window closing, shutting down server...")
        asyncio.create_task(self.stop_server())
        event.accept()