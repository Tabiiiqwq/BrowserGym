import os
import sys
import fire
import argparse
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

def extract_tasks(benchmark, output_file, max_tasks):
    print(f"\nExtract {benchmark} tasks to file...")
    
    try:
        if benchmark == "webarena":
            from benchmark_extractor.webarena_task_extractor import WebArenaTaskExtractor
            extractor = WebArenaTaskExtractor()
            task_ids = extractor.get_all_task_ids()[:max_tasks]
            
            tasks = []
            for task_id in task_ids:
                task_data = extractor.prepare_task_for_agent(task_id)
                if task_data:
                    clean_task = {
                        "task_id": task_id,
                        "goal": task_data["goal"],
                        "start_urls": task_data["start_urls"],
                        "primary_url": task_data["primary_url"],
                        "credentials": task_data["credentials"],
                        "sites": task_data["sites"],
                        "benchmark": "webarena"
                    }
                    tasks.append(clean_task)
        
        elif benchmark == "assistantbench":
            from benchmark_extractor.assistantbench_task_extractor import AssistantBenchTaskExtractor
            extractor = AssistantBenchTaskExtractor()
            task_ids = extractor.get_all_task_ids("validation")[:max_tasks]
            
            tasks = []
            for task_id in task_ids:
                task_data = extractor.prepare_task_for_agent(task_id)
                if task_data:
                    clean_task = {
                        "task_id": task_id,
                        "description": task_data["description"],
                        "start_url": task_data["start_url"],
                        "benchmark": "assistantbench"
                    }
                    tasks.append(clean_task)
        
        else:
            print(f"❌ Not support benchmark: {benchmark}")
            return
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Save{len(tasks)}tasks to file: {output_file}")
        
    except Exception as e:
        print(f"❌ Extract fail: {e}")


def main(benchmark: str, output: str, max_tasks: int):
    extract_tasks(benchmark, output, max_tasks)


if __name__ == "__main__":
    fire.Fire(main)