import json
import tempfile
import importlib.resources
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import playwright.sync_api

logger = logging.getLogger(__name__)


@dataclass
class WebArenaTaskInfo:
    """Container for extracted WebArena task information"""
    task_id: int
    intent: str  # Task description/goal
    start_url: str  # Starting URL(s)
    sites: List[str]  # Required sites for authentication
    geolocation: Dict[str, float]  # Latitude/longitude
    config: Dict[str, Any]  # Full task configuration
    evaluator_config_file: str  # Temporary config file for evaluation


class WebArenaTaskExtractor:
    """Extract and manage WebArena tasks for external agents"""
    
    def __init__(self):
        """Initialize the extractor with WebArena instance"""
        from browsergym.webarena.instance import WebArenaInstance
        self.webarena_instance = WebArenaInstance()
        self._load_task_configs()
    
    def _load_task_configs(self):
        """Load all WebArena task configurations"""
        try:
            import webarena
            all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
            
            # Substitute URLs with actual instance URLs
            for pattern, url_key in {
                "__GITLAB__": "gitlab",
                "__REDDIT__": "reddit", 
                "__SHOPPING__": "shopping",
                "__SHOPPING_ADMIN__": "shopping_admin",
                "__WIKIPEDIA__": "wikipedia",
                "__MAP__": "map",
            }.items():
                all_configs_str = all_configs_str.replace(pattern, self.webarena_instance.urls[url_key])
            
            self.all_configs = json.loads(all_configs_str)
            logger.info(f"Loaded {len(self.all_configs)} WebArena task configurations")
            
        except Exception as e:
            logger.error(f"Failed to load WebArena configs: {e}")
            self.all_configs = []
    
    def get_task_info(self, task_id: int) -> Optional[WebArenaTaskInfo]:
        """
        Extract task information for a specific task ID
        
        Args:
            task_id: The WebArena task ID
            
        Returns:
            WebArenaTaskInfo object with task details, or None if not found
        """
        # Find task config
        task_configs = [conf for conf in self.all_configs if conf["task_id"] == task_id]
        if not task_configs:
            logger.error(f"Task ID {task_id} not found")
            return None
        
        config = task_configs[0]  # Take first match
        
        # Create temporary config file for evaluation
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            f.flush()
            config_file = f.name
        
        return WebArenaTaskInfo(
            task_id=task_id,
            intent=config["intent"],
            start_url=config.get("start_url", ""),
            sites=config.get("sites", []),
            geolocation=config.get("geolocation", {"latitude": 0, "longitude": 0}),
            config=config,
            evaluator_config_file=config_file
        )
    
    def get_all_task_ids(self) -> List[int]:
        """Get list of all available task IDs"""
        return [config["task_id"] for config in self.all_configs]
    
    def prepare_task_for_agent(self, task_id: int, 
                             with_homepage_hint: bool = False,
                             with_na_hint: bool = False) -> Optional[Dict[str, Any]]:
        """
        Prepare task information in a format suitable for external agents
        
        Args:
            task_id: WebArena task ID
            with_homepage_hint: Include homepage navigation hint
            with_na_hint: Include "N/A" answer hint
            
        Returns:
            Dictionary with task information for agent consumption
        """
        task_info = self.get_task_info(task_id)
        if not task_info:
            return None
        
        # Build goal description
        goal = task_info.intent
        
        if with_homepage_hint:
            goal += f"""

(Note: if you want to visit other websites, check out the homepage at {self.webarena_instance.home_url}. It has a list of websites you can visit. {self.webarena_instance.home_url}/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)
"""
        
        if with_na_hint:
            goal += """\

If you believe the task is impossible to complete, provide the answer "N/A".
"""
        
        # Parse start URLs
        start_urls = []
        if task_info.start_url:
            start_urls = task_info.start_url.split(" |AND| ")
        
        return {
            "task_id": task_id,
            "goal": goal,
            "start_urls": start_urls,
            "primary_url": start_urls[0] if start_urls else "",
            "sites": task_info.sites,
            "geolocation": task_info.geolocation,
            "viewport": {"width": 1280, "height": 720},
            "timeout": 10000,  # ms
            "credentials": self.webarena_instance.credentials,
            "instance_urls": self.webarena_instance.urls,
            "home_url": self.webarena_instance.home_url,
            "_evaluation_config": task_info.evaluator_config_file  # For internal evaluation use
        }


class WebArenaEvaluator:
    """Handle evaluation of completed tasks using BrowserGym's evaluators"""
    
    def __init__(self):
        self.extractor = WebArenaTaskExtractor()
    
    def evaluate_task_result(self, task_id: int, 
                           agent_answer: str, 
                           page_state: playwright.sync_api.Page) -> Tuple[float, bool, str, Dict]:
        """
        Evaluate task completion using WebArena's evaluation system
        
        Args:
            task_id: WebArena task ID
            agent_answer: The agent's final answer/response
            page_state: Current page state
            
        Returns:
            Tuple of (score, is_done, message, info_dict)
        """
        task_info = self.extractor.get_task_info(task_id)
        if not task_info:
            return 0.0, True, "Task not found", {"error": "Invalid task ID"}
        
        try:
            # Import evaluator
            from webarena.evaluation_harness.evaluators import evaluator_router
            from webarena.browser_env.actions import ActionTypes
            
            # Build evaluator
            evaluator = evaluator_router(task_info.evaluator_config_file)
            
            # Create fake trajectory for evaluation (only answer is used)
            last_action = {"action_type": ActionTypes.STOP, "answer": agent_answer}
            trajectory = [{}, last_action]  # StateInfo, Action
            
            # Call evaluator
            score = evaluator(
                trajectory=trajectory,
                config_file=task_info.evaluator_config_file,
                page=page_state,
                client=None
            )
            
            success = score > 0 or last_action["action_type"] == ActionTypes.STOP
            return score, success, "", {}
            
        except Exception as e:
            logger.error(f"Evaluation failed for task {task_id}: {e}")
            return 0.0, True, f"Evaluation error: {e}", {"error": str(e)}