from typing import List, Any
from .transformer import TransformerEngine
from .web_search import WebSearchTool
from .trajectory import TrajectoryInference
from .clustering import Clustering

class Agent:
    def __init__(self):
        self.transformer = TransformerEngine()
        self.web_search = WebSearchTool()
        self.trajectory = TrajectoryInference()
        self.clustering = Clustering()

    def run(self, ehr_list: List[str], lab_tests: List[List[Any]], progress_callback=None):
        """
        Main entry point for the agent to process data.
        
        Args:
            ehr_list: List of EHR strings/records.
            lab_tests: 2D list of lab test data.
            progress_callback: Optional callable(step_name, status, data) to report progress.
        """
        def report(step, status, details=None):
            if progress_callback:
                progress_callback(step, status, details)

        results = {}

        # 1. Transformer Inference
        report("Transformer Inference", "running")
        print("Agent: Starting Transformer Inference...")
        transformer_result = self.transformer.predict(ehr_list, lab_tests)
        results['transformer_prediction'] = transformer_result
        report("Transformer Inference", "completed", transformer_result)

        # 2. Web Search & Reasoning
        # Use the transformer result as context for reasoning
        report("Web Search & Reasoning", "running")
        print("Agent: Starting Web Search & Reasoning...")
        query = f"reasoning for prediction score {transformer_result.get('prediction_score')}"
        reasoning_result = self.web_search.search_and_reason(query, context=str(transformer_result))
        results['reasoning'] = reasoning_result
        report("Web Search & Reasoning", "completed", reasoning_result)

        # 3. Single-cell Trajectory Inference
        # (Assuming the input data or derived data is used here; passing full input for now)
        report("Trajectory Inference", "running")
        print("Agent: Running Trajectory Inference...")
        trajectory_result = self.trajectory.run_inference(lab_tests)
        results['trajectory_inference'] = trajectory_result
        report("Trajectory Inference", "completed", trajectory_result)

        # 4. Clustering
        report("Clustering", "running")
        print("Agent: Running Clustering...")
        clustering_result = self.clustering.perform_clustering(lab_tests)
        results['clustering'] = clustering_result
        report("Clustering", "completed", clustering_result)

        return results

