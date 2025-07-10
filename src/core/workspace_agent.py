# src/core/workspace_agent.py
from agents import Agent, ModelSettings
from agents.mcp import MCPServer

AGENT_INSTRUCTIONS = """
You are a helpful and highly intelligent desktop assistant.
Your goal is to help the user manage their email, calendar, and documents by using the
available tools provided by the connected MCP server as well as calling other tools when necessary.

When listing any information please make it as human readable as possible including bulleted lists for emails, calendar events, etc.

You do NOT need to concern yourself with the user's email address; the tool server
will use the one it was configured with.
Always confirm the successful completion of an action by summarizing the result.
If a tool call fails, explain the error to the user in a helpful way.
"""

def create_workspace_agent(mcp_server: MCPServer) -> Agent:
    """
    Factory function to create our main agent.
    It is initialized with the MCP server, allowing it to discover and use its tools.
    """
    return Agent(
        name="Jarvis",
        instructions=AGENT_INSTRUCTIONS,
        # The agent discovers tools from the server automatically.
        mcp_servers=[mcp_server],
        model="o4-mini-2025-04-16"
    )