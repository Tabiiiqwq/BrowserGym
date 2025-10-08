import logging
import os
import tempfile
import gymnasium as gym
from typing import Dict, Any, Optional
from playwright.sync_api import Page
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo_agent.agent import DemoAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs
from browsergym.core.env import BrowserEnv

from agents.agent_system_wrapper import AgentSystemWrapper

logger = logging.getLogger(__name__)


class BrowserGymAgentWrapper(AgentSystemWrapper):
    def __init__(self, 
                 agent_args: DemoAgentArgs,
                 max_steps: int = 50,
                 headless: bool = False,
                 viewport: Optional[Dict[str, int]] = None):
        self.agent_args = agent_args
        self.max_steps = max_steps
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        
        self.env: Optional[BrowserEnv] = None
        self.agent = None
        self._temp_dir = None
    
    def setup(self) -> None:
        pass
    
    def run_task(self, task_data: Dict[str, Any]) -> str:
        try:
            self.agent = self.agent_args.make_agent()         
            self.env = self._create_env_for_task(task_data)
            answer = self._run_agent_episode()
            return answer
            
        except Exception as e:
            logger.error(f"Error running task: {e}")
            raise
    
    def _create_env_for_task(self, task_data: Dict[str, Any]) -> BrowserEnv:

        
        # TODO: Try openended setting for all tasks
        if task_data['benchmark'] == 'webarena':
            task_name = f"browsergym/webarena.{task_data['task_id']}"
            task_kwargs = {}
            import browsergym.webarena
            import browsergym.webarenalite
            
        else:  # assistantbench
            task_name = f"browsergym/assistantbench.{task_data['task_id']}"
            task_kwargs = {
                # "start_url": task_data['start_url'],
                # "goal": task_data['description']
            }
            import browsergym.assistantbench
        
        env = gym.make(
            task_name,
            headless=self.headless,
            max_episode_steps=self.max_steps,
            viewport=self.viewport,
            action_mapping=self.agent.action_set.to_python_code,
            task_kwargs=task_kwargs
        )
        
        return env
    
    def _run_agent_episode(self) -> str:
        """
        Run a single episode with the agent and extract the answer
        """
        obs, info = self.env.reset()
        done = False
        step_count = 0
        last_action = ""
        
        while not done and step_count < self.max_steps:
            try:
                # Get action from agent
                action, action_info = self.agent.get_action(self.agent.obs_preprocessor(obs))
                print(f"Action at step {step_count}: {action}")
                
                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                last_action = action
                step_count += 1
                
                logger.debug(f"Step {step_count}: Action={action}, Reward={reward}, Done={done}")
                
            except Exception as e:
                logger.error(f"Error in agent step {step_count}: {e}")
                break

        answer = self._extract_answer_from_actions(last_action, obs, info)
        
        return answer
    
    def _extract_answer_from_actions(self, last_action: str, obs: Dict[str, Any], info: Dict[str, Any]) -> str:
        """
        Extract the answer from agent actions or observations
        
        This is a heuristic approach and might need refinement based on
        how different agents structure their final answers.
        """
        # Try to extract from last action if it contains send_msg_to_user
        if "send_msg_to_user" in last_action:
            import re
            match = re.search(r'send_msg_to_user\("([^"]+)"\)', last_action)
            if match:
                return match.group(1)
        
        # Try to extract from chat messages
        if obs.get("chat_messages"):
            for msg in reversed(obs["chat_messages"]):
                if msg.get("role") == "assistant" and msg.get("message"):
                    return msg["message"]
        
        return last_action.strip()
    
    def get_page(self) -> Optional[Page]:
        """Get the current page object for evaluation"""
        if self.env and hasattr(self.env, 'page'):
            return self.env.page
        elif self.env and hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'page'):
            return self.env.unwrapped.page
        return None
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.env:
                self.env.close()
                self.env = None
            
            # # Clean up temporary directory
            # if self._temp_dir and os.path.exists(self._temp_dir):
            #     import shutil
            #     shutil.rmtree(self._temp_dir, ignore_errors=True)
            #     logger.info(f"Cleaned up temp directory: {self._temp_dir}")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")