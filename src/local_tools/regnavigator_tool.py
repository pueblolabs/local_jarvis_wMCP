# src/local_tools/regnavigator_tool.py

import os
import keyring
from agents import function_tool
from . import register_tool
import asyncio

SANDBOX_DIR = os.path.expanduser("~/local_jarvis_sandbox")
SERVICE_NAME = "com.yourcompany.google-workspace-agent"

def _ensure_api_keys_in_env():
    """
    Loads API keys from the keychain into environment variables if they aren't already set.
    """
    keys_to_check = {
        "REGS_API_KEY": "REGS_API_KEY",
        "GEMINI_API_KEY": "GEMINI_API_KEY",
    }
    for env_var, key_name in keys_to_check.items():
        if not os.getenv(env_var):
            value = keyring.get_password(SERVICE_NAME, key_name)
            if value:
                os.environ[env_var] = value

async def _summarize_docket_logic(docket_id: str, max_comments: int = 5) -> str:
    """
    Core logic to summarize a US federal docket. This raw async function is called
    directly by the orchestrator for performance.
    """
    _ensure_api_keys_in_env()
    from src.local_tools.lib.regnavigator_lite.core import analyse

    try:
        if not os.getenv("REGS_API_KEY") or not os.getenv("GEMINI_API_KEY"):
            return (
                "Error: Could not find RegNavigator or Gemini API keys. "
                "Please ensure they were entered correctly on startup."
            )

        max_comments = max(1, min(max_comments, 250))
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        summary_bullets, pdf_data = await analyse(docket_id, target=max_comments)

        pdf_filename = f"{docket_id.replace('/', '_')}_briefing.pdf"
        pdf_path = os.path.join(SANDBOX_DIR, pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)

        return (
            f"**Docket {docket_id} Summary:**\n"
            f"{summary_bullets}\n\n"
            f"Full PDF brief saved to: `{pdf_path}`"
        )
    except Exception as e:
        error_message = str(e)
        if "api_key" in error_message.lower():
            return f"An authentication error occurred with the API: {e}. Please verify your API keys."
        return f"An error occurred while running RegNavigator for docket {docket_id}: {e}."

# Create the tool for the agent by decorating the logic function
summarize_docket = function_tool(_summarize_docket_logic)
summarize_docket.description = "Summarize a US federal docket via RegNavigator. Creates a local PDF. Use this for single docket requests."

# Register the decorated tool object for the agent to discover
register_tool(summarize_docket)