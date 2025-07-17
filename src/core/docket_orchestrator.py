# src/core/docket_orchestrator.py

import asyncio
import yaml
from typing import List, Callable, Any, Dict

from agents import Agent, Runner
# --- CORRECTED IMPORT ---
# We import the raw, callable logic function, not the decorated tool object.
from src.local_tools.regnavigator_tool import _summarize_docket_logic

def create_synthesis_agent() -> Agent:
    """Creates a temporary agent for synthesizing results."""
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    synthesis_prompt = config['docket_orchestrator']['synthesis_prompt']

    return Agent(
        name="SynthesisAgent",
        instructions=synthesis_prompt,
        model="o4-mini-2025-04-16"
    )

class DocketOrchestrator:
    """
    Orchestrates the parallel analysis of multiple dockets,
    reporting progress and synthesizing the final results.
    """

    def __init__(self, progress_callback: Callable[[str, str], None]):
        """
        Initializes the orchestrator.
        Args:
            progress_callback: A function to call with progress updates.
                               It will be called with (docket_id, status_string).
        """
        self.progress_callback = progress_callback
        self.synthesis_agent = create_synthesis_agent()

    async def _run_single_docket_summary(self, docket_id: str, max_comments: int) -> Dict[str, Any]:
        """Runs the analysis for a single docket and reports progress."""
        try:
            self.progress_callback(docket_id, "PROCESSING...")
            
            # --- CORRECTED CALL ---
            # Now we are calling the actual async function, which is callable.
            result = await _summarize_docket_logic(docket_id=docket_id, max_comments=max_comments)

            self.progress_callback(docket_id, "COMPLETED")
            return {"docket_id": docket_id, "status": "success", "summary": result}
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.progress_callback(docket_id, error_message)
            return {"docket_id": docket_id, "status": "error", "summary": error_message}

    async def orchestrate(self, docket_ids: List[str], max_comments: int) -> str:
        """
        Main orchestration method. Runs summaries in parallel and synthesizes results.
        """
        for docket_id in docket_ids:
            self.progress_callback(docket_id, "QUEUED")

        tasks = [
            self._run_single_docket_summary(docket_id, max_comments)
            for docket_id in docket_ids
        ]
        
        results = await asyncio.gather(*tasks)

        self.progress_callback("synthesis", "SYNTHESIZING...")
        
        successful_summaries = [r['summary'] for r in results if r['status'] == 'success']
        if not successful_summaries:
            self.progress_callback("synthesis", "FAILED: No successful summaries.")
            return "Could not generate a briefing as all docket analyses failed."

        # Format the collected summaries for the synthesis agent
        combined_text = "\n\n---\n\n".join(successful_summaries)
        
        final_result = await Runner.run(self.synthesis_agent, combined_text)
        
        self.progress_callback("synthesis", "COMPLETED")
        return final_result.final_output