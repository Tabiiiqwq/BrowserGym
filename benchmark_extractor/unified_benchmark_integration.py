import sys
sys.path.append(".")
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass


from .assistantbench_task_extractor import AssistantBenchTaskExtractor, AssistantBenchEvaluator
from .webarena_task_extractor import WebArenaTaskExtractor, WebArenaEvaluator
import playwright.sync_api

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTaskResult:
    """Unified result structure for both WebArena and AssistantBench"""
    task_id: str
    benchmark: str  # 'webarena' or 'assistantbench'
    agent_answer: str
    score: float
    success: bool
    gold_answer: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class UnifiedBenchmarkInterface:
    """Unified interface for both WebArena and AssistantBench"""
    
    def __init__(self):
        """Initialize extractors and evaluators for both benchmarks"""
        self.webarena_extractor = None
        self.webarena_evaluator = None
        self.assistantbench_extractor = None
        self.assistantbench_evaluator = None
        
        # Try to initialize WebArena
        try:
            self.webarena_extractor = WebArenaTaskExtractor()
            self.webarena_evaluator = WebArenaEvaluator()
            self.webarena_available = True
            logger.info("WebArena integration loaded successfully")
        except Exception as e:
            self.webarena_available = False
            logger.warning(f"WebArena not available: {e}")
        
        # Try to initialize AssistantBench
        try:
            self.assistantbench_extractor = AssistantBenchTaskExtractor()
            self.assistantbench_evaluator = AssistantBenchEvaluator()
            self.assistantbench_available = True
            logger.info("AssistantBench integration loaded successfully")
        except Exception as e:
            self.assistantbench_available = False
            logger.warning(f"AssistantBench not available: {e}")
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks"""
        benchmarks = []
        if self.webarena_available:
            benchmarks.append("webarena")
        if self.assistantbench_available:
            benchmarks.append("assistantbench")
        return benchmarks
    
    def get_task_ids(self, benchmark: str, **kwargs) -> List[str]:
        """
        Get task IDs for a specific benchmark
        
        Args:
            benchmark: 'webarena' or 'assistantbench'
            **kwargs: Additional parameters (e.g., split for AssistantBench)
        """
        if benchmark == "webarena" and self.webarena_available:
            return self.webarena_extractor.get_all_task_ids()
        elif benchmark == "assistantbench" and self.assistantbench_available:
            split = kwargs.get("split", None)
            return self.assistantbench_extractor.get_all_task_ids(split)
        else:
            return []
    
    def get_task(self, benchmark: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information for a specific benchmark and task ID"""
        if benchmark == "webarena" and self.webarena_available:
            task_data = self.webarena_extractor.prepare_task_for_agent(int(task_id))
            if task_data:
                task_data["benchmark"] = "webarena"
            return task_data
        elif benchmark == "assistantbench" and self.assistantbench_available:
            task_data = self.assistantbench_extractor.prepare_task_for_agent(task_id)
            if task_data:
                task_data["benchmark"] = "assistantbench"
            return task_data
        else:
            return None
    
    def evaluate_result(self, benchmark: str, task_id: str, agent_answer: str, page: playwright.sync_api.Page = None) -> UnifiedTaskResult:
        """Evaluate agent result for any benchmark"""
        if benchmark == "webarena" and self.webarena_available:
            score, is_done, message, info = self.webarena_evaluator.evaluate_task_result(
                int(task_id), agent_answer, page_state=page
            )
            return UnifiedTaskResult(
                task_id=str(task_id),
                benchmark="webarena",
                agent_answer=agent_answer,
                score=score,
                success=score > 0,
                error_message=message if message else None,
                metadata=info
            )
        elif benchmark == "assistantbench" and self.assistantbench_available:
            accuracy, is_done, message, info = self.assistantbench_evaluator.evaluate_task_result(
                task_id, agent_answer
            )
            return UnifiedTaskResult(
                task_id=task_id,
                benchmark="assistantbench",
                agent_answer=agent_answer,
                score=accuracy,
                success=accuracy > 0,
                gold_answer=info.get("gold_answer"),
                error_message=message if message else None,
                metadata=info
            )
        else:
            return UnifiedTaskResult(
                task_id=task_id,
                benchmark=benchmark,
                agent_answer=agent_answer,
                score=0.0,
                success=False,
                error_message=f"Benchmark {benchmark} not available"
            )
    
    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get statistics for all available benchmarks"""
        stats = {}
        
        if self.webarena_available:
            webarena_tasks = self.webarena_extractor.get_all_task_ids()
            stats["webarena"] = {
                "total_tasks": len(webarena_tasks),
                "task_range": f"{min(webarena_tasks)}-{max(webarena_tasks)}" if webarena_tasks else "N/A",
                "available": True
            }
        else:
            stats["webarena"] = {"available": False}
        
        if self.assistantbench_available:
            validation_tasks = self.assistantbench_extractor.get_all_task_ids("validation")
            test_tasks = self.assistantbench_extractor.get_all_task_ids("test")
            imp_tasks = self.assistantbench_extractor.get_all_task_ids("imp")
            
            stats["assistantbench"] = {
                "validation_tasks": len(validation_tasks),
                "test_tasks": len(test_tasks),
                "implementation_tasks": len(imp_tasks),
                "total_tasks": len(validation_tasks) + len(test_tasks) + len(imp_tasks),
                "available": True
            }
        else:
            stats["assistantbench"] = {"available": False}
        
        return stats