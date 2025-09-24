# Enhanced MCP server for BrowserGym with complete agent interface
import argparse
import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image

from browsergym.core.action.highlevel import ACTION_SUBSETS, HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert image to base64 URL"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    import base64
    import io
    
    # Convert to JPEG format
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"


@dataclass
class BgymConfig:
    headless: bool = True
    timeout_ms: int = 10000
    record_video_dir: str | None = None
    demo_mode: HighLevelActionSet.DemoMode = "default"
    validate_actions: list[str] = field(default_factory=list)
    action_subset: str = "miniwob"
    task_name: str = "browsergym/openended"
    start_url: str = "about:blank"
    goal: str | None = None
    seed: int | None = None


@dataclass
class AppContext:
    gym: BrowserEnv
    config: BgymConfig
    current_task_id: str
    actions: HighLevelActionSet
    initialized: bool = False


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Enhanced BrowserGym MCP server",
        usage="python %(prog)s [options]",
        epilog="To run Dev UI: mcp dev %(prog)s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )
    parser.add_argument(
        "-r",
        "--record_video_dir",
        type=str,
        default=None,
        help="Directory to save recorded videos",
    )
    parser.add_argument(
        "--demo_mode",
        type=str,
        default="off",
        choices=["off", "default", "all_blue", "only_visible_elements"],
        help="Demo mode for action set",
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=10000,
        help="Timeout in milliseconds for each step",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="miniwob",
        choices=ACTION_SUBSETS.keys(),
        help="Subset of actions to use",
    )
    parser.add_argument(
        "--validate_actions",
        type=str,
        nargs="+",
        default=["click", "goto"],
        help="Names of actions for which validation should be performed",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="browsergym/openended",
        help="Task to initialize on startup (e.g., 'miniwob.click-test', 'webarena.task-123')",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="about:blank",
        help="Start URL for openended tasks",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal description for openended tasks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for task initialization",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the MCP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the MCP server",
    )
    args, _ = parser.parse_known_args()
    return args


args = get_cli_args()
config = BgymConfig(
    headless=args.headless,
    timeout_ms=args.timeout_ms,
    record_video_dir=args.record_video_dir,
    demo_mode=args.demo_mode,
    validate_actions=args.validate_actions,
    action_subset=args.subset,
    task_name=args.task,
    start_url=args.start_url,
    goal=args.goal,
    seed=args.seed,
)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize action set
    actions = HighLevelActionSet(demo_mode=config.demo_mode, subsets=config.action_subset)
    
    # Prepare task kwargs based on task type
    task_kwargs = {}
    if config.task_name == "browsergym/openended":
        task_kwargs = {
            "start_url": config.start_url,
            "goal": config.goal
        }
    
    # Create environment with specified task
    _gym: BrowserEnv = await asyncio.to_thread(
        gym.make,
        config.task_name,
        headless=config.headless,
        record_video_dir=config.record_video_dir,
        action_mapping=actions.to_python_code,
        timeout=config.timeout_ms,
        task_kwargs=task_kwargs,
    )  # type: ignore
    
    # Reset with seed if provided
    reset_kwargs = {"seed": config.seed} if config.seed is not None else {}
    await asyncio.to_thread(_gym.reset, **reset_kwargs)

    try:
        yield AppContext(
            gym=_gym, 
            config=config, 
            current_task_id=config.task_name, 
            actions=actions,
            initialized=True
        )
    finally:
        # Cleanup on shutdown
        await asyncio.to_thread(_gym.close)


mcp = FastMCP("BrowserGym Enhanced", lifespan=app_lifespan,
              host=args.host, port=args.port)


def format_func_call(func: Callable, args, kwargs) -> str:
    """Format function call for logging"""
    args_str = ", ".join(repr(arg) for arg in args)
    kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
    all_args_str = ", ".join(filter(None, [args_str, kwargs_str]))
    return f"{func.__name__}({all_args_str})"


def serialize_observation(obs: dict) -> dict:
    """Convert observation to JSON-serializable format"""
    serialized = {}
    
    for key, value in obs.items():
        if key == "screenshot":
            # Convert screenshot to base64
            if value is not None:
                serialized[key] = image_to_jpg_base64_url(value)
            else:
                serialized[key] = None
        elif key in ["dom_object", "axtree_object"]:
            # Convert DOM/AXTree to string representation
            if key == "dom_object":
                serialized[key] = flatten_dom_to_str(value) if value else ""
                serialized[key + "_pruned"] = prune_html(serialized[key])
            elif key == "axtree_object":
                serialized[key] = flatten_axtree_to_str(value) if value else ""
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            serialized[key] = value.tolist()
        elif isinstance(value, tuple):
            # Convert tuples to lists
            serialized[key] = list(value)
        else:
            # Keep other values as-is
            serialized[key] = value
            
    return serialized


# === Core MCP Tools ===

@mcp.add_tool
async def get_observation() -> dict:
    """
    Get the complete observation information of the current web page, including screenshot, URL, DOM structure, etc.

    Returns:
        dict: Observation information containing the following fields:
            - url: The current page URL
            - screenshot: Page screenshot (base64 encoded)
            - dom_object: DOM tree structure (string format)
            - axtree_object: Accessibility tree structure (string format)
            - open_pages_urls: List of URLs of all opened pages
            - open_pages_titles: List of titles of all opened pages
            - active_page_index: Index of the currently active page
            - last_action: The last executed action
            - last_action_error: Error message of the last action
            - elapsed_time: Task execution time
            - goal: Current task goal
            - chat_messages: History of chat messages
    """

    context: AppContext = mcp.get_context().request_context.lifespan_context  # type: ignore
    gym_env = context.gym
    
    obs = gym_env._get_obs()
    return serialize_observation(obs)


@mcp.add_tool
async def get_available_actions() -> dict:
    """
    Get the list of currently available actions and their detailed descriptions.

    Returns:
        dict: Contains the following fields:
            - actions: List of action names
            - descriptions: Detailed description of each action
            - examples: Usage examples of actions
            - subset_name: The name of the currently used action subset
    """

    context: AppContext = mcp.get_context().request_context.lifespan_context  # type: ignore
    
    action_set = context.actions
    
    # Get action descriptions
    descriptions = action_set.describe(with_long_description=True, with_examples=True)
    
    # Get action names
    action_names = [fn.__name__ for fn in ACTION_SUBSETS[config.action_subset]]
    
    return {
        "actions": action_names,
        "descriptions": descriptions,
        "subset_name": config.action_subset,
        "available_subsets": list(ACTION_SUBSETS.keys())
    }


# initialize_task is removed - tasks are initialized at server startup via command line arguments


@mcp.add_tool
async def get_task_info() -> dict:
    """
    Get detailed information about the current task.

    Returns:
        dict: Task information, including goal, status, etc.
    """

    context: AppContext = mcp.get_context().request_context.lifespan_context  # type: ignore
    gym_env = context.gym
    
    obs = gym_env._get_obs()
    
    return {
        "task_id": context.current_task_id,
        "goal": obs.get("goal", ""),
        "goal_object": obs.get("goal_object", []),
        "initialized": context.initialized,
        "current_url": obs.get("url", ""),
        "open_pages": list(zip(obs.get("open_pages_urls", []), obs.get("open_pages_titles", []))),
        "elapsed_time": obs.get("elapsed_time", [0])[0] if obs.get("elapsed_time") is not None else 0
    }


@mcp.add_tool
async def reset_environment() -> dict:
    """
    Reset the current task environment to its initial state.

    Returns:
        dict: Reset result containing task information and new observation
    """
    context: AppContext = mcp.get_context().request_context.lifespan_context  # type: ignore
    gym_env = context.gym
    
    # Reset the environment
    obs, info = await asyncio.to_thread(gym_env.reset)
    
    return {
        "reset_success": True,
        "task_info": info.get("task_info", {}),
        "observation": serialize_observation(obs)
    }


# === Action Execution Wrapper ===

def fn_wrapper(func: Callable, validate: bool = True):
    """Wrapper for action functions to execute in gym context"""
    async def decorator(*args, **kwargs):
        """
        Decorator to execute function from the action space in the context of the gym.
        """
        context: AppContext = mcp.get_context().request_context.lifespan_context  # type: ignore
        gym_env = context.gym
        
        # Load the parent module of the function to use as function context
        import browsergym.core.action.functions as fn_context

        fn = getattr(fn_context, func.__name__)

        gym_env.last_action = format_func_call(fn, args, kwargs)
        info, send_message_to_user, report_infeasible_instructions = await asyncio.to_thread(
            gym_env.pre_step
        )

        # Set up the module vars from the current state of the gym
        fn_context.send_message_to_user = send_message_to_user
        fn_context.report_infeasible_instructions = report_infeasible_instructions
        fn_context.page = gym_env.page
        fn_context.demo_mode = config.demo_mode

        try:
            fn(*args, **kwargs)
            gym_env.last_action_error = ""
        except Exception as e:
            gym_env.last_action_error = f"{type(e).__name__}: {e}"
            match = re.match("TimeoutError: Timeout ([0-9]+)ms exceeded.", gym_env.last_action_error)
            if match:
                info["action_exec_timeout"] = float(match.groups()[0]) / 1000

        obs, reward, terminated, truncated, step_info = await asyncio.to_thread(gym_env.post_step, info, validate)
        
        return {
            "action_executed": format_func_call(fn, args, kwargs),
            "success": not gym_env.last_action_error,
            "error": gym_env.last_action_error,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "observation": serialize_observation(obs),
            "info": step_info
        }

    decorator.__wrapped__ = func  # type: ignore
    decorator.__name__ = func.__name__
    decorator.__doc__ = func.__doc__
    return decorator


# Register all action functions as MCP tools
for fn in ACTION_SUBSETS[config.action_subset]:
    validate = fn.__name__ in config.validate_actions
    mcp.add_tool(fn_wrapper(fn, validate))


if __name__ == "__main__":
    print("=== Enhanced BrowserGym MCP Server ===")
    print(f"Task Configuration:")
    print(f"- Task: {config.task_name}")
    if config.task_name == "browsergym/openended":
        print(f"- Start URL: {config.start_url}")
        goal_display = config.goal.encode('ascii', 'replace').decode('ascii') if config.goal else 'None'
        print(f"- Goal: {goal_display}")
    print(f"- Seed: {config.seed or 'Random'}")
    print("")
    print(f"Server Configuration:")
    print(f"- Action subset: {config.action_subset}")
    print(f"- Available actions: {[fn.__name__ for fn in ACTION_SUBSETS[config.action_subset]]}")
    print(f"- Headless mode: {config.headless}")
    print(f"- Demo mode: {config.demo_mode}")
    print(f"- Timeout: {config.timeout_ms}ms")
    print("")
    print("Available MCP Tools:")
    print("1. get_observation() - Get current page observation")
    print("2. get_available_actions() - Query available actions") 
    print("3. get_task_info() - Get task information")
    print("4. reset_environment() - Reset to initial state")
    print(f"5. Action functions: {[fn.__name__ for fn in ACTION_SUBSETS[config.action_subset]]}")
    print("")
    print("Note: Task is fixed at startup. Use multiple server instances for different tasks.")
    print("Ready to serve MCP clients!")
    mcp.run(transport="stdio")