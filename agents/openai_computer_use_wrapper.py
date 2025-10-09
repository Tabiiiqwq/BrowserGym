"""
OpenAI Computer Use Agent Wrapper

This module implements an agent wrapper that uses OpenAI's Computer Use API
to interact with web pages through a browser interface.
"""

import logging
import os
import sys
import tempfile
import time
import base64
from typing import Dict, Any, Optional, List, NamedTuple
from playwright.sync_api import Page
from dataclasses import dataclass

# Add external dependencies
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "external", "openai-cua-sample-app"))

try:
    from computers import Computer
    from computers.default import LocalPlaywrightBrowser
    from utils import create_response
    from agent.agent import Agent
except ImportError as e:
    logging.error(f"Failed to import OpenAI CUA dependencies: {e}")
    logging.error("Make sure the openai-cua-sample-app submodule is properly initialized")
    raise

from agents.agent_system_wrapper import AgentSystemWrapper


@dataclass
class CostInfo:
    """Cost information for OpenAI API usage"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from a single API call"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1
        self._calculate_cost()

    def _calculate_cost(self):
        """Calculate cost based on current token usage"""
        # Computer Use Preview pricing: $3/M input tokens, $12/M output tokens
        self.input_cost = (self.input_tokens / 1_000_000) * 3.0
        self.output_cost = (self.output_tokens / 1_000_000) * 12.0
        self.total_cost = self.input_cost + self.output_cost


class OpenAIComputerUseAgentWrapper(AgentSystemWrapper):
    """
    Agent wrapper that uses OpenAI's Computer Use API to perform web tasks
    """

    def __init__(
        self,
        headless: bool = False,
        max_steps: int = 50,
    ):
        self.headless = headless
        self.max_steps = max_steps

        self.computer: Optional[Computer] = None

        # Cost tracking
        self.cost_info = CostInfo()
        self.agent = None

        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def setup(self) -> None:
        """Setup browser and computer environment"""
        try:
            # Create temporary directory

            # Initialize computer with local Playwright browser
            self.computer = LocalPlaywrightBrowser(headless=self.headless)
            self.computer.__enter__()  # Start the computer context

            extra_tools = [
                {
                    "type": "function",
                    "name": "back",
                    "description": "Go back to the previous page.",
                    "parameters": {},
                },
                {
                    "type": "function",
                    "name": "goto",
                    "description": "Go to a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Fully qualified URL to navigate to.",
                            },
                        },
                        "additionalProperties": False,
                        "required": ["url"],
                    },
                },
            ]

            self.agent = Agent(computer=self.computer, tools=extra_tools, cost_info=self.cost_info)

            print("OpenAI Computer Use agent setup completed")

        except Exception as e:
            print(f"Failed to setup OpenAI Computer Use agent: {e}")
            raise

    def run_task(self, task_data: Dict[str, Any], task_cfg: Dict[str, Any]) -> str:
        """
        Run the agent on a task and return the answer

        Args:
            task_data: Task information from UnifiedBenchmarkInterface.get_task()

        Returns:
            Agent's answer as a string
        """
        try:
            if not self.computer:
                raise RuntimeError("Computer not initialized. Call setup() first.")

            # Reset cost tracking for this task
            self.cost_info = CostInfo()

            
            # Prepare the task prompt
            task_prompt = self._create_task_prompt(task_data, task_cfg)
            
            if not task_cfg['real_a2a']:
                # Navigate to start URL if provided
                start_url = self._get_start_url(task_data)
                if start_url:
                    print(f"Navigating to start URL: {start_url}")
                    self.computer.goto(start_url)
                    time.sleep(2)  # Wait for page to load

                # Login for webarena
                if task_data['benchmark'] == 'webarena' and task_data.get('credentials'):
                    # Perform login if credentials are provided
                    for site in task_data['sites']:
                        print(f"Logging into site: {site}")
                        self.ui_login(site, start_url, self.get_page(), task_data['credentials'])
                        time.sleep(2)  # Wait after login

            # Run the agent loop
            answer = self._run_agent_loop(task_prompt)

            return answer

        except Exception as e:
            print(f"Error running OpenAI Computer Use agent: {e}")
            # Still print cost summary even if there was an error
            if hasattr(self, "cost_info"):
                self.print_cost_summary()
            raise

    def _create_task_prompt(self, task_data: Dict[str, Any], task_cfg) -> str:
        """Create a task prompt from task data"""
        if task_cfg['real_a2a']:
            if task_data["benchmark"] == "webarena":
                goal = task_data["goal"]
                start_url = task_data.get("primary_url", "")

                prompt = f"""Please help me complete this web task:

    GOAL: {goal}

    STARTING URL: {start_url}

    CREDENTIALS: {task_data.get('credentials', 'N/A')}

    Please navigate to the starting URL and complete the specified task. When you have found the answer or completed the task, provide a clear final answer.

    Important instructions:
    1. Navigate to the starting URL first
    2. Log in with the provided credentials if needed
    3. Complete the task step by step
    4. When you find the answer, state it clearly in your final message
    5. Be precise and thorough in your actions
    """

            else:  # assistantbench
                description = task_data["description"]
                start_url = task_data.get("start_url", "https://google.com")

                prompt = f"""Please help me answer this question by browsing the web:

    QUESTION: {description}

    STARTING URL: {start_url}

    Please use the browser to find the answer to this question. You can start from the provided URL or navigate to other sites as needed.

    Important instructions:
    1. Start from the given URL (or Google if no specific URL provided)
    2. Search for and find the information needed to answer the question
    3. When you find the answer, state it clearly and concisely in your final message
    4. Be precise and provide specific details when available
    """
        else: # will auto goto init url, and login
            if task_data["benchmark"] == "webarena":
                goal = task_data["goal"]
                start_url = task_data.get("primary_url", "")

                prompt = f"""Please help me complete this web task:

    GOAL: {goal}

    Please complete the specified task. When you have found the answer or completed the task, provide a clear final answer.

    Important instructions:
    1. Complete the task step by step
    2. When you find the answer, state it clearly in your final message
    3. Be precise and thorough in your actions
    """

            else:  # assistantbench
                description = task_data["description"]
                start_url = task_data.get("start_url", "https://google.com")

                prompt = f"""Please help me answer this question by browsing the web:

    QUESTION: {description}

    Please use the browser to find the answer to this question. You can start from the provided URL or navigate to other sites as needed.

    Important instructions:
    1. Search for and find the information needed to answer the question
    2. When you find the answer, state it clearly and concisely in your final message
    3. Be precise and provide specific details when available
    """
            

        return prompt

    def _get_start_url(self, task_data: Dict[str, Any]) -> str:
        """Extract start URL from task data"""
        if task_data["benchmark"] == "webarena":
            return task_data.get("primary_url", "")
        else:  # assistantbench
            return task_data.get("start_url", "https://google.com")

    def _add_user_message(self, content: str) -> None:
        """Add a user message to conversation history"""
        self.conversation_history.append({"role": "user", "content": content})

    def _run_agent_loop(self, message: str) -> str:
        input_items = [
            {
                "role": "developer",
                "content": "You are a helpful web agent, help user to complete task. Besides basic actions, you can use the additional back() and goto() functions to navigate the browser.",
            },
            {"role": "user", "content": message},
        ]
        output_items = self.agent.run_full_turn(input_items, show_images=False)
        return output_items[-1]["content"]['text']

    def ui_login(self, site: str, url: str, page: Page, credentials: Dict[str, Any]):
        """
        Should only be called once per site (expects user to be logged out).
        Borrowed and adapted from webarena/instance.py
        """


        # open a new page (tab) to perform the login
        page = page.context.new_page()

        match site:
            case "reddit":
                username = credentials[site]["username"]
                password = credentials[site]["password"]

                page.goto(f"{url}")
                page.get_by_role("link", name="Log in").click()
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Log in").click()

            case "gitlab":
                username = credentials[site]["username"]
                password = credentials[site]["password"]

                page.goto(f"{url}/users/sign_in")
                page.get_by_label("Username or email").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "shopping":
                username = credentials[site]["username"]
                password = credentials[site]["password"]

                page.goto(f"{url}/customer/account/login/")
                page.get_by_label("Email", exact=True).fill(username)
                page.get_by_label("Password", exact=True).fill(password)
                page.get_by_role("button", name="Sign In").click()

            case "shopping_admin":
                username = credentials[site]["username"]
                password = credentials[site]["password"]

                page.goto(url)
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "wikipedia":
                page.goto(url)

            case "map":
                page.goto(url)

            case _:
                raise ValueError

        # release login page
        page.close()

    def get_page(self) -> Optional[Page]:
        """Get the current page object for evaluation"""
        if self.computer and hasattr(self.computer, "_page"):
            return self.computer._page
        return None

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.computer:
                self.computer.__exit__(None, None, None)
                self.computer = None

        except Exception as e:
            print(f"Error during cleanup: {e}")
