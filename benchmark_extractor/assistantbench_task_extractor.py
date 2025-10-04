import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class AssistantBenchTaskInfo:
    """Container for extracted AssistantBench task information"""
    task_id: str
    original_id: str  # Original AssistantBench ID
    task_description: str
    gold_answer: str
    start_url: str
    split: str  # 'validation', 'test', or 'imp'


class AssistantBenchTaskExtractor:
    """Extract and manage AssistantBench tasks for external agents and human evaluation"""
    
    def __init__(self):
        """Initialize the extractor with AssistantBench dataset"""
        self.start_url = "https://google.com"
        self._load_task_data()
    
    def _load_task_data(self):
        """Load all AssistantBench task data"""
        try:
            # Load the dataset
            DATA_DATASET = "AssistantBench/AssistantBench"
            all_tasks = load_dataset(DATA_DATASET)

            # print(all_tasks.keys()) # validation, test
            
            # Extract validation data
            self.validation_tasks = {}
            for i, row in enumerate(all_tasks["validation"]):
                task_id = f"validation.{i}"
                self.validation_tasks[task_id] = AssistantBenchTaskInfo(
                    task_id=task_id,
                    original_id=row["id"],
                    task_description=row["task"],
                    gold_answer=row["answer"] if row["answer"] is not None else "",
                    start_url=self.start_url,
                    split="validation"
                )
            
            # Extract test data
            self.test_tasks = {}
            for i, row in enumerate(all_tasks["test"]):
                task_id = f"test.{i}"
                self.test_tasks[task_id] = AssistantBenchTaskInfo(
                    task_id=task_id,
                    original_id=row["id"],
                    task_description=row["task"],
                    gold_answer=row["answer"] if row["answer"] is not None else "",
                    start_url=self.start_url,
                    split="test"
                )
            
            # self.implementation_tasks = {
            #     "imp.0": AssistantBenchTaskInfo(
            #         task_id="imp.0",
            #         original_id="test_imp_id_0",
            #         task_description="What is the weather in Paris yesterday in Celsius? Answer with the number only.",
            #         gold_answer="20",
            #         start_url=self.start_url,
            #         split="imp"
            #     )
            # }
            
            # Combine all tasks
            self.all_tasks = {
                **self.validation_tasks,
                **self.test_tasks,
                # **self.implementation_tasks
            }
            
            logger.info(f"Loaded AssistantBench tasks: {len(self.validation_tasks)} validation, "
                       f"{len(self.test_tasks)} test")
            
        except Exception as e:
            logger.error(f"Failed to load AssistantBench data: {e}")
            self.all_tasks = {}
            self.validation_tasks = {}
            self.test_tasks = {}
            # self.implementation_tasks = {}
    
    def get_task_info(self, task_id: str) -> Optional[AssistantBenchTaskInfo]:
        """
        Extract task information for a specific task ID
        
        Args:
            task_id: The AssistantBench task ID (e.g., 'validation.0', 'test.5', 'imp.0')
            
        Returns:
            AssistantBenchTaskInfo object with task details, or None if not found
        """
        return self.all_tasks.get(task_id)
    
    def get_all_task_ids(self, split: Optional[str] = None) -> List[str]:
        """
        Get list of all available task IDs
        
        Args:
            split: Optional filter by split ('validation', 'test', 'imp')
            
        Returns:
            List of task IDs
        """
        if split is None:
            return list(self.all_tasks.keys())
        
        if split == "validation":
            return list(self.validation_tasks.keys())
        elif split == "test":
            return list(self.test_tasks.keys())
        else:
            return []
    
    def prepare_task_for_agent(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Prepare task information in a format suitable for external agents
        
        Args:
            task_id: AssistantBench task ID
            
        Returns:
            Dictionary with task information for agent consumption
        """
        task_info = self.get_task_info(task_id)
        if not task_info:
            return None
        
        return {
            "task_id": task_id,
            "original_id": task_info.original_id,
            "description": task_info.task_description,
            "start_url": task_info.start_url,
            "split": task_info.split,
            "gold_answer": task_info.gold_answer,  # For evaluation only
            "_evaluation_data": {
                "gold_answer": task_info.gold_answer,
                "task_id": task_info.original_id
            }
        }
    
    def prepare_batch_tasks(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """Prepare multiple tasks for batch processing"""
        tasks = []
        for task_id in task_ids:
            task = self.prepare_task_for_agent(task_id)
            if task:
                tasks.append(task)
        return tasks


class AssistantBenchEvaluator:
    """Handle evaluation of completed tasks using AssistantBench's evaluation system"""
    
    def __init__(self):
        self.extractor = AssistantBenchTaskExtractor()
    
    def evaluate_task_result(self, task_id: str, agent_answer: str) -> Tuple[float, bool, str, Dict]:
        """
        Evaluate task completion using AssistantBench's evaluation system
        
        Args:
            task_id: AssistantBench task ID
            agent_answer: The agent's final answer/response
            
        Returns:
            Tuple of (accuracy_score, is_done, message, info_dict)
        """
        task_info = self.extractor.get_task_info(task_id)
        if not task_info:
            return 0.0, True, "Task not found", {"error": "Invalid task ID"}
        
        try:
            # Import the evaluator from AssistantBench
            from browsergym.assistantbench.evaluation.evaluator import question_scorer
            
            # Evaluate the answer
            accuracy, has_answer = question_scorer(agent_answer, task_info.gold_answer)
            
            return accuracy, True, "", {
                "has_answer": has_answer,
                "gold_answer": task_info.gold_answer,
                "prediction": agent_answer
            } # accuracy, done, msg, info
            
        except Exception as e:
            logger.error(f"Evaluation failed for task {task_id}: {e}")
            return 0.0, True, f"Evaluation error: {e}", {"error": str(e)}


if __name__ == '__main__':
    extractor = AssistantBenchTaskExtractor()