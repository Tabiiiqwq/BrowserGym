import asyncio
import logging
from typing import Dict, Any, Optional
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
import fire

from benchmark_extractor.unified_benchmark_integration import UnifiedBenchmarkInterface
from agents.agent_system_wrapper import AgentSystemWrapper
from agents.browsergym_agent_wrapper import BrowserGymAgentWrapper
from demo_agent.agent import DemoAgentArgs

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()


class HumanEval:
    
    def __init__(self):
        self.interface = UnifiedBenchmarkInterface()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    def setup_browser(self):
        playwright = sync_playwright().start()
        self.browser = playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = self.context.new_page()
        print("Browser launched")
    
    def cleanup_browser(self):
        if self.browser:
            self.browser.close()
            print("Browser closed")
    
    def display_task_info(self, task_data: Dict[str, Any]):
        print("\n" + "="*60)
        print("Task Information")
        print("="*60)
        print(f"Task ID: {task_data['task_id']}")
        print(f"Benchmark: {task_data['benchmark']}")
        
        if task_data['benchmark'] == 'webarena':
            print(f"Goal: {task_data['goal']}")
            print(f"Start URL: {task_data['primary_url']}")
            if task_data.get('sites'):
                print(f"Sites: {', '.join(task_data['sites'])}")
                print("\nCredentials:")
                for site, creds in task_data.get('credentials', {}).items():
                    if site in task_data.get('sites', []):
                        print(f"  {site}: User Name={creds.get('username', 'N/A')}, Password={creds.get('password', 'N/A')}")
        else:  # assistantbench
            print(f"Goal: {task_data['description']}")
            print(f"Start URL: {task_data['start_url']}")
        
        print("="*60)
    
    def navigate_to_start_url(self, task_data: Dict[str, Any]):
        if task_data['benchmark'] == 'webarena':
            url = task_data['primary_url']
        else:
            url = task_data['start_url']
        
        print(f"Go to: {url}")
        self.page.goto(url)
    
    def get_user_answer(self) -> str:
        print("\n" + "="*60)
        print("✏️  Complete the task in the browser, then enter your answer here")
        print("="*60)
        
        while True:
            answer = input("Your answer ('quit' to exit): ").strip()
            if answer.lower() == 'quit':
                return ""
            elif answer:
                return answer
            else:
                print("❌ Error can't be empty")
    
    def evaluate_answer(self, benchmark: str, task_id: str, answer: str, page: Page = None) -> Dict[str, Any]:
        try:
            if benchmark == "webarena":
                if not page:
                    return {
                        "score": 0.0,
                        "success": False,
                        "error": "Need page object for WebArena evaluation"
                    }
                
                result = self.interface.evaluate_result(benchmark, task_id, answer, page)
                
                return {
                    "score": result.score,
                    "success": result.success,
                    "error": result.error_message
                }
            
            else:  # assistantbench
                result = self.interface.evaluate_result(benchmark, task_id, answer)
                return {
                    "score": result.score,
                    "success": result.success,
                    "gold_answer": result.gold_answer,
                    "error": result.error_message
                }
                
        except Exception as e:
            logger.error(f"Eval error: {e}")
            return {
                "score": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def display_result(self, result: Dict[str, Any], answer: str):
        print("\n" + "="*60)
        print("Eval Result")
        print("="*60)
        print(f"Your answer: {answer}")
        print(f"Score: {result['score']:.2f}")
        print(f"Result: {'✅ Succ' if result['success'] else '❌ Fail'}")
        
        if result.get('gold_answer'):
            print(f"Gold Answer: {result['gold_answer']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        print("="*60)
    
    def run_single_task(self, benchmark: str, task_id: str):
        task_data = self.interface.get_task(benchmark, task_id)
        if not task_data:
            print(f"❌ Can' find task {task_id} in {benchmark}")
            return
        
        try:
            self.setup_browser()
            self.display_task_info(task_data)
            self.navigate_to_start_url(task_data)
            answer = self.get_user_answer()
            
            if not answer:
                print("❌ Cancleed by user")
                return
        
            print("🔄 Start to eval...")
            page_for_eval = self.page if benchmark == "webarena" else None
            result = self.evaluate_answer(benchmark, task_id, answer, page_for_eval)
            
            self.display_result(result, answer)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            self.cleanup_browser()

class AgentEval:
    """Simplified agent evaluation class"""
    
    def __init__(self):
        self.interface = UnifiedBenchmarkInterface()
    
    def run_single_task(self, 
                           agent_wrapper: AgentSystemWrapper,
                           benchmark: str, 
                           task_id: str,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single task using the provided agent wrapper
        
        Args:
            agent_wrapper: Agent system wrapper implementing AgentSystemWrapper protocol
            benchmark: Benchmark name ('webarena' or 'assistantbench')
            task_id: Task identifier
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get task data
        task_data = self.interface.get_task(benchmark, task_id)
        if not task_data:
            error_msg = f"❌ Cannot find task {task_id} in {benchmark}"
            if verbose:
                print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "score": 0.0
            }
        
        if verbose:
            self._display_task_info(task_data)
        
        try:
            # Setup agent browser
            agent_wrapper.setup()
            
            if verbose:
                print("🤖 Running agent...")
            
            # Run agent on task
            answer = agent_wrapper.run_task(task_data)
            
            if not answer:
                error_msg = "❌ Agent returned empty answer"
                if verbose:
                    print(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "score": 0.0
                }
            
            if verbose:
                print(f"🔄 Agent answer: {answer}")
                print("🔄 Evaluating...")
            
            # Evaluate result
            page_for_eval = agent_wrapper.get_page() if benchmark == "webarena" else None
            result = self.interface.evaluate_result(benchmark, task_id, answer, page_for_eval)
            
            eval_result = {
                "success": result.success,
                "score": result.score,
                "agent_answer": answer,
                "error": result.error_message,
                "benchmark": benchmark,
                "task_id": task_id
            }
            
            if result.gold_answer:
                eval_result["gold_answer"] = result.gold_answer
            
            if verbose:
                self._display_result(eval_result)
            
            return eval_result
            
        except Exception as e:
            error_msg = f"❌ Evaluation error: {e}"
            logger.error(error_msg, exc_info=True)
            if verbose:
                print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "score": 0.0,
                "benchmark": benchmark,
                "task_id": task_id
            }
        
        finally:
            # Cleanup
            try:
                agent_wrapper.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")
    
    def _display_task_info(self, task_data: Dict[str, Any]):
        """Display task information"""
        print("\n" + "="*60)
        print("Task Information")
        print("="*60)
        print(f"Task ID: {task_data['task_id']}")
        print(f"Benchmark: {task_data['benchmark']}")
        
        if task_data['benchmark'] == 'webarena':
            print(f"Goal: {task_data['goal']}")
            print(f"Start URL: {task_data['primary_url']}")
            if task_data.get('sites'):
                print(f"Sites: {', '.join(task_data['sites'])}")
        else:  # assistantbench
            print(f"Goal: {task_data['description']}")
            print(f"Start URL: {task_data['start_url']}")
        
        print("="*60)
    
    def _display_result(self, result: Dict[str, Any]):
        """Display evaluation results"""
        print("\n" + "="*60)
        print("Evaluation Result")
        print("="*60)
        print(f"Agent Answer: {result['agent_answer']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Result: {'✅ Success' if result['success'] else '❌ Failed'}")
        
        if result.get('gold_answer'):
            print(f"Gold Answer: {result['gold_answer']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        print("="*60)
    
    def run_multiple_tasks(self,
                              agent_wrapper: AgentSystemWrapper,
                              benchmark: str,
                              task_ids: list,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate multiple tasks
        
        Args:
            agent_wrapper: Agent system wrapper
            benchmark: Benchmark name
            task_ids: List of task IDs to evaluate
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing aggregated results
        """
        results = []
        successful_tasks = 0
        total_score = 0.0
        
        for i, task_id in enumerate(task_ids, 1):
            if verbose:
                print(f"\n🔄 Evaluating task {i}/{len(task_ids)}: {task_id}")
            
            result = self.run_single_task(agent_wrapper, benchmark, task_id, verbose)
            results.append(result)
            
            if result['success']:
                successful_tasks += 1
            total_score += result.get('score', 0.0)
        
        # Calculate aggregated metrics
        success_rate = successful_tasks / len(task_ids) if task_ids else 0.0
        average_score = total_score / len(task_ids) if task_ids else 0.0
        
        summary = {
            "benchmark": benchmark,
            "total_tasks": len(task_ids),
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_score": average_score,
            "individual_results": results
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Summary")
            print("="*60)
            print(f"Benchmark: {benchmark}")
            print(f"Total Tasks: {len(task_ids)}")
            print(f"Successful Tasks: {successful_tasks}")
            print(f"Success Rate: {success_rate:.2%}")
            print(f"Average Score: {average_score:.2f}")
            print("="*60)
        
        return summary

def main(action: str,
               benchmark: str,
               task_id: str):
    if action == "list_tasks":
        interface = UnifiedBenchmarkInterface()
        task_ids = interface.get_task_ids(benchmark)
        print(f"Available tasks in {benchmark}: {task_ids}")
    elif action == "human_eval":
        evaluator = HumanEval()
        if task_id:
            evaluator.run_single_task(benchmark, task_id)
    elif action == "agent_eval":
        evaluator = AgentEval()
        
        agent_args = DemoAgentArgs(agent_name="test_agent",
                                   model_name="gpt-4o",
                                   use_html=True,
                                   use_axtree=True,
                                   use_screenshot=False)

        agent_wrapper = BrowserGymAgentWrapper(agent_args=agent_args)
        if task_id:
            evaluator.run_single_task(agent_wrapper, benchmark, task_id)
        
        
    


if __name__ == "__main__":
    fire.Fire(main)
        