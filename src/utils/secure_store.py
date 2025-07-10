# src/utils/secure_store.py
import keyring
import keyring.errors
import logging

logger = logging.getLogger(__name__)

SERVICE_NAME = "com.yourcompany.google-workspace-agent"

def set_secret(key: str, value: str) -> bool:
    try:
        keyring.set_password(SERVICE_NAME, key, value)
        logger.info(f"Successfully stored '{key}' in keychain for service '{SERVICE_NAME}'.")
        return True
    except keyring.errors.NoKeyringError:
        logger.warning("No OS keyring found. Secret will not be stored securely.")
        return False
    except Exception as e:
        logger.error(f"Failed to set secret for '{key}': {e}", exc_info=True)
        return False

def get_secret(key: str) -> str | None:
    try:
        return keyring.get_password(SERVICE_NAME, key)
    except keyring.errors.NoKeyringError:
        logger.warning("No OS keyring found. Cannot retrieve secret.")
        return None
    except Exception as e:
        logger.error(f"Failed to get secret for '{key}': {e}", exc_info=True)
        return None

def delete_secret(key: str):
    try:
        keyring.delete_password(SERVICE_NAME, key)
        logger.info(f"Successfully deleted '{key}' from keychain.")
    except keyring.errors.PasswordDeleteError:
        logger.debug(f"Secret '{key}' not found in keychain for deletion.")
    except keyring.errors.NoKeyringError:
        pass
    except Exception as e:
        logger.error(f"Failed to delete secret for '{key}': {e}", exc_info=True)