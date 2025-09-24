#!/usr/bin/env python3

"""
A simple example to demonstrate how an agent interacts with a MiniWoB task.
"""

import os
import gymnasium as gym
import browsergym.miniwob
import browsergym.core
from browsergym.core.action.highlevel import HighLevelActionSet

os.environ["MINIWOB_URL"] = "file:///E:/rensh/AgentBeats/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"


def main():
    print("=== BrowserGym MiniWoB Task Example ===")
    
    # Set MiniWoB URL (using local files)
    # In actual use, you need to set the MINIWOB_URL environment variable first
    if "MINIWOB_URL" not in os.environ:
        print("Warning: MINIWOB_URL is not set. Please install MiniWoB++ first.")
        return
    
    # Create a click-button task
    task_name = "browsergym/miniwob.click-button"
    print(f"Creating task: {task_name}")
    
    # Initialize environment
    env = gym.make(
        task_name,
        headless=False,  # Show browser window for observation
        wait_for_user_message=False
    )
    
    # Reset environment and start task
    print("\n=== Task Start ===")
    obs, info = env.reset(seed=42)
    
    # 1. Show task goal
    print(f"Task goal: {obs['goal']}")
    print(f"Target object: {obs['goal_object']}")
    
    # 2. Show page information
    print(f"Current URL: {obs['url']}")
    print(f"Number of open pages: {len(obs['open_pages_urls'])}")
    print(f"Active page index: {obs['active_page_index']}")

    print('obs keys:', obs.keys())
    for k, v in obs.items():
        if k not in ['axtree_object', 'screenshot', 'dom_object', 'extra_element_properties']:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: (big object, omitted)")
    print('info keys:', info.keys())
    print(info)
    
    # 3. Show DOM structure (accessibility tree)
    print(f"\n=== Page Accessibility Tree (first 500 characters) ===")
    from browsergym.utils.obs import flatten_axtree_to_str
    axtree_str = flatten_axtree_to_str(obs['axtree_object'])
    print(axtree_str[:500] + "..." if len(axtree_str) > 500 else axtree_str)
    
    # 4. Show screenshot information
    print(f"\n=== Screenshot Information ===")
    print(f"Screenshot shape: {obs['screenshot'].shape}")
    
    # 5. Create action set
    action_set = HighLevelActionSet(
        subsets=["bid", "coord"],  # Allow bid and coordinate click
        strict=False,
        multiaction=False
    )
    
    print(f"\n=== Available Actions ===")
    print(action_set.describe(with_examples=True))
    
    # 6. Simulate agent executing some actions
    print(f"\n=== Executing Actions ===")
    
    step_count = 0
    max_steps = 5
    
    while step_count < max_steps:
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        
        # Check if the task is already completed
        if info.get('task_info', {}).get('DONE_GLOBAL', False):
            print("Task completed!")
            break
        
        # This is just a demonstration. Normally, the agent would analyze the page and decide what to do.
        # For the click-button task, we need to locate the target button and click it.
        
        # Simple heuristic: search for a button on the page
        axtree_str = flatten_axtree_to_str(obs['axtree_object'])
        print(f"Page content: {axtree_str[:200]}...")
        
        # Look for button elements
        import re
        button_matches = re.findall(r'button "([^"]*)" <(\d+)>', axtree_str)
        if button_matches:
            button_text, bid = button_matches[0]
            action = f'click("{bid}")'
            print(f"Executing action: {action} (click button: {button_text})")
        else:
            # If no button is found, try other clickable elements
            clickable_matches = re.findall(r'<(\d+)>', axtree_str)
            if clickable_matches:
                bid = clickable_matches[0]
                action = f'click("{bid}")'
                print(f"Executing action: {action} (click element)")
            else:
                print("No clickable elements found, trying noop")
                action = 'noop()'
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Last action error: {obs['last_action_error']}")
        
        if terminated or truncated:
            print("Task ended")
            break
    
    # Show final result
    print(f"\n=== Task Result ===")
    print(f"Final reward: {reward}")
    print(f"Task info: {info.get('task_info', {})}")
    
    # Close environment
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    main()
