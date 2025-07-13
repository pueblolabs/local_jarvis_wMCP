# src/local_tools/regnavigator_tool.py

import os
import keyring
from agents import function_tool
from . import register_tool
import asyncio

SANDBOX_DIR = os.path.expanduser("~/local_jarvis_sandbox")
# Use the same service name as the rest of the application
SERVICE_NAME = "com.yourcompany.google-workspace-agent"

def _ensure_api_keys_in_env():
    """
    Loads API keys from the keychain into environment variables if they aren't already set.
    This ensures the tool works even if it's imported before the main config loads.
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

@register_tool
@function_tool
async def summarize_docket(docket_id: str, max_comments: int = 5) -> str:
    """Summarize a US federal docket via RegNavigator.

    Args:
        docket_id: e.g. "IRS-2022-0029".
        max_comments: Sample size for comment analysis (1-250). Defaults to 5.

    Returns:
        Executive-summary bullets plus path of the generated PDF brief.
    """
    # Ensure the environment is ready right before we use the library
    _ensure_api_keys_in_env()

    # Import the library here to ensure it configures itself with the correct environment
    from .lib.regnavigator_lite.core import analyse

    try:
        # Check one last time if the keys were actually loaded.
        if not os.getenv("REGS_API_KEY") or not os.getenv("GEMINI_API_KEY"):
            return (
                "Error: Could not find RegNavigator or Gemini API keys in the keychain. "
                "Please ensure they were entered correctly on startup."
            )

        max_comments = max(1, min(max_comments, 250))

        # ensure sandbox folder exists
        os.makedirs(SANDBOX_DIR, exist_ok=True)

        summary_bullets, pdf_data = await analyse(docket_id, target=max_comments)

        pdf_filename = f"{docket_id.replace('/', '_')}_briefing.pdf"
        pdf_path = os.path.join(SANDBOX_DIR, pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)

        return (
            f"Successfully analyzed docket {docket_id} using {max_comments} comments.\n\n"
            f"**Executive Summary:**\n{summary_bullets}\n\n"
            f"PDF brief saved to: {pdf_path}"
        )

    except Exception as e:
        # Provide a more specific error to the user
        error_message = str(e)
        if "api_key" in error_message.lower():
            return f"An authentication error occurred with the API: {e}. Please verify your API keys."
        return (
            f"An error occurred while running RegNavigator for docket {docket_id}: {e}."
        )

@register_tool
@function_tool
async def summarize_dockets_concurrently(docket_ids: list[str], max_comments: int = 5) -> str:
    """Summarize multiple US federal dockets concurrently via RegNavigator.

    Args:
        docket_ids: A list of docket IDs, e.g. ["IRS-2022-0029", "FTC-2023-0066"].
        max_comments: Sample size for comment analysis (1-250) for each docket. Defaults to 5.

    Returns:
        A summary of the results for each docket, including the paths of the generated PDF briefs.
    """
    tasks = [summarize_docket(docket_id, max_comments) for docket_id in docket_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    response = "Finished processing all dockets.\n\n"
    for docket_id, result in zip(docket_ids, results):
        if isinstance(result, Exception):
            response += f"**Docket {docket_id}:**\n- Error: {result}\n\n"
        else:
            response += f"**Docket {docket_id}:**\n{result}\n\n"

    return response
