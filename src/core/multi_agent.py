# src/core/multi_agent.py
from agents import Agent
from agents.mcp import MCPServer
from src.core.workspace_agent import create_workspace_agent
from typing import List

def create_agent(mcp_servers: List[MCPServer]) -> Agent:
    """
    Factory function to create our main agent.
    """
    return create_workspace_agent(mcp_servers)