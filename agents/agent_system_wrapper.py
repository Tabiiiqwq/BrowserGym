import logging
from typing import Dict, Any, Optional, Protocol
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from benchmark_extractor.unified_benchmark_integration import UnifiedBenchmarkInterface

logger = logging.getLogger(__name__)


class AgentSystemWrapper(Protocol):
    """Protocol defining the interface for agent system wrappers"""
    
    def setup(self) -> None:
        """Setup browser and context for the agent"""
        ...
    
    def run_task(self, task_data: Dict[str, Any]) -> str:
        """
        Run the agent on a task and return the answer
        
        Args:
            task_data: Task information from UnifiedBenchmarkInterface.get_task()
            
        Returns:
            Agent's answer as a string
        """
        ...
    
    def get_page(self) -> Optional[Page]:
        """Get the current page object for evaluation"""
        ...
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        ...