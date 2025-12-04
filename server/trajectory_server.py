import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from server.base import MCPServer


class TrajectoryAlgorithm(Enum):
    """Supported trajectory inference algorithms."""
    MONOCLE3 = "monocle3"
    SLINGSHOT = "slingshot"
    PAGA = "paga"
    VELOCYTO = "velocyto"
    SCORPIUS = "scorpius"


@dataclass
class TrajectoryNode:
    """Represents a node in the trajectory graph."""
    id: str
    name: str
    cell_count: int
    pseudotime: float
    cluster_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "cell_count": self.cell_count,
            "pseudotime": self.pseudotime,
            "cluster_id": self.cluster_id,
            "metadata": self.metadata or {}
        }


@dataclass
class TrajectoryEdge:
    """Represents an edge in the trajectory graph."""
    source: str
    target: str
    weight: float
    transition_probability: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "transition_probability": self.transition_probability,
            "metadata": self.metadata or {}
        }


class TrajectoryMCPServer(MCPServer):
    """
    MCP Server for Single-Cell Trajectory Inference.
    
    This server handles:
    - Pseudotime calculation
    - Trajectory graph construction
    - Branching analysis
    - Lineage inference
    - Trajectory visualization data generation
    """
    
    SERVER_KEY = "trajectory"
    
    def __init__(
        self,
        algorithm: str = "monocle3",
        n_neighbors: int = 15,
        min_branch_length: int = 10,
        root_selection_method: str = "automatic",
        **kwargs
    ) -> None:
        """Initialize the Trajectory MCP Server."""
        self.algorithm = TrajectoryAlgorithm(algorithm)
        self.n_neighbors = n_neighbors
        self.min_branch_length = min_branch_length
        self.root_selection_method = root_selection_method
        
        # Default Redis keys
        task_queue_key = kwargs.pop("task_queue_key", "mochiagent:trajectory:task")
        result_queue_key = kwargs.pop("result_queue_key", "mochiagent:trajectory:result")
        
        super().__init__(
            task_queue_key=task_queue_key,
            result_queue_key=result_queue_key,
            **kwargs
        )
    
    def _register_tools(self) -> None:
        """Register trajectory inference tools."""
        self.register_tool(
            "infer_trajectory",
            self._tool_infer_trajectory,
            "Infer single-cell trajectory from expression data"
        )
        self.register_tool(
            "compute_pseudotime",
            self._tool_compute_pseudotime,
            "Compute pseudotime ordering of cells"
        )
        self.register_tool(
            "detect_branches",
            self._tool_detect_branches,
            "Detect branching points in trajectory"
        )
        self.register_tool(
            "lineage_analysis",
            self._tool_lineage_analysis,
            "Perform lineage analysis"
        )
        self.register_tool(
            "trajectory_diff_expression",
            self._tool_trajectory_diff_expression,
            "Find differentially expressed genes along trajectory"
        )
        self.register_tool(
            "generate_visualization",
            self._tool_generate_visualization,
            "Generate trajectory visualization data"
        )
    
    def _register_resources(self) -> None:
        """Register trajectory resources."""
        self.register_resource(
            "trajectory://config",
            {
                "algorithm": self.algorithm.value,
                "n_neighbors": self.n_neighbors,
                "min_branch_length": self.min_branch_length,
                "root_selection_method": self.root_selection_method
            }
        )
        self.register_resource(
            "trajectory://algorithms",
            [alg.value for alg in TrajectoryAlgorithm]
        )
    
    def _register_prompts(self) -> None:
        """Register trajectory prompts."""
        self.register_prompt(
            "trajectory_interpretation",
            "Interpret the following trajectory analysis results:\n{results}\n\nBiological context: {context}"
        )
        self.register_prompt(
            "branch_annotation",
            "Annotate the following trajectory branches:\n{branches}\n\nCell type markers: {markers}"
        )
    
    def _build_knn_graph(self, data: np.ndarray) -> Dict[str, Any]:
        """Build k-nearest neighbor graph from data."""
        n_samples = data.shape[0]
        n_neighbors = min(self.n_neighbors, n_samples - 1)
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        try:
            distances = cdist(data, data, metric='euclidean')
        except:
            # Fallback to manual computation
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    distances[i, j] = np.linalg.norm(data[i] - data[j])
        
        # Find k-nearest neighbors
        knn_indices = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
        knn_distances = np.take_along_axis(distances, knn_indices, axis=1)
        
        # Build adjacency list
        adjacency = {}
        for i in range(n_samples):
            adjacency[i] = [
                {"neighbor": int(knn_indices[i, j]), "distance": float(knn_distances[i, j])}
                for j in range(n_neighbors)
            ]
        
        return {
            "n_samples": n_samples,
            "n_neighbors": n_neighbors,
            "adjacency": adjacency
        }
    
    def _compute_diffusion_pseudotime(self, knn_graph: Dict, root_cell: int = 0) -> np.ndarray:
        """Compute pseudotime using diffusion-based method."""
        n_samples = knn_graph["n_samples"]
        adjacency = knn_graph["adjacency"]
        
        # Build transition matrix
        transition_matrix = np.zeros((n_samples, n_samples))
        for i, neighbors in adjacency.items():
            i = int(i)
            total_weight = sum(1.0 / (n["distance"] + 1e-10) for n in neighbors)
            for n in neighbors:
                weight = (1.0 / (n["distance"] + 1e-10)) / total_weight
                transition_matrix[i, n["neighbor"]] = weight
        
        # Compute pseudotime via random walk from root
        pseudotime = np.zeros(n_samples)
        current_prob = np.zeros(n_samples)
        current_prob[root_cell] = 1.0
        
        # Iterate diffusion
        for step in range(100):
            current_prob = transition_matrix.T @ current_prob
            pseudotime += current_prob * step
        
        # Normalize to [0, 1]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-10)
        
        return pseudotime
    
    def _detect_branch_points(self, pseudotime: np.ndarray, knn_graph: Dict) -> List[Dict]:
        """Detect branching points in the trajectory."""
        n_samples = len(pseudotime)
        branch_points = []
        
        # Bin cells by pseudotime
        n_bins = min(20, n_samples // 5)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pseudotime, bins) - 1
        
        # Detect branches by analyzing connectivity changes
        for b in range(1, n_bins - 1):
            cells_in_bin = np.where(bin_indices == b)[0]
            cells_before = np.where(bin_indices == b - 1)[0]
            cells_after = np.where(bin_indices == b + 1)[0]
            
            if len(cells_in_bin) > 0 and len(cells_after) > len(cells_before) * 1.5:
                # Potential branch point
                branch_points.append({
                    "pseudotime": float(bins[b]),
                    "cell_count": len(cells_in_bin),
                    "downstream_expansion": len(cells_after) / (len(cells_before) + 1),
                    "confidence": min(1.0, len(cells_in_bin) / self.min_branch_length)
                })
        
        return branch_points
    
    def _build_trajectory_graph(
        self,
        pseudotime: np.ndarray,
        branch_points: List[Dict]
    ) -> Tuple[List[TrajectoryNode], List[TrajectoryEdge]]:
        """Build trajectory graph from pseudotime and branch points."""
        n_samples = len(pseudotime)
        
        # Create nodes for major pseudotime segments
        n_segments = max(5, len(branch_points) + 2)
        segment_boundaries = np.linspace(0, 1, n_segments + 1)
        
        nodes = []
        for i in range(n_segments):
            seg_start, seg_end = segment_boundaries[i], segment_boundaries[i + 1]
            cells_in_segment = np.sum((pseudotime >= seg_start) & (pseudotime < seg_end))
            
            node = TrajectoryNode(
                id=f"node_{i}",
                name=f"State_{i}",
                cell_count=int(cells_in_segment),
                pseudotime=float((seg_start + seg_end) / 2),
                cluster_id=f"cluster_{i % 3}",
                metadata={"segment_range": [float(seg_start), float(seg_end)]}
            )
            nodes.append(node)
        
        # Create edges between consecutive nodes
        edges = []
        for i in range(len(nodes) - 1):
            # Check if there's a branch point between nodes
            is_branch = any(
                nodes[i].pseudotime <= bp["pseudotime"] <= nodes[i+1].pseudotime
                for bp in branch_points
            )
            
            edge = TrajectoryEdge(
                source=nodes[i].id,
                target=nodes[i+1].id,
                weight=1.0 / (abs(nodes[i].pseudotime - nodes[i+1].pseudotime) + 0.1),
                transition_probability=0.9 if not is_branch else 0.7,
                metadata={"is_branch_transition": is_branch}
            )
            edges.append(edge)
            
            # Add alternative branch if detected
            if is_branch and i + 2 < len(nodes):
                alt_edge = TrajectoryEdge(
                    source=nodes[i].id,
                    target=nodes[i+2].id,
                    weight=0.5,
                    transition_probability=0.3,
                    metadata={"is_alternative_branch": True}
                )
                edges.append(alt_edge)
        
        return nodes, edges
    
    def _tool_infer_trajectory(
        self,
        data: List[List[float]],
        cell_ids: Optional[List[str]] = None,
        root_cell: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Infer single-cell trajectory from expression data.
        
        Args:
            data: 2D expression matrix (cells x genes)
            cell_ids: Optional cell identifiers
            root_cell: Optional root cell index
            
        Returns:
            Trajectory inference results
        """
        self._log_info(f"Inferring trajectory using {self.algorithm.value}")
        
        # Convert to numpy
        data_array = np.array(data, dtype=np.float32)
        n_cells, n_features = data_array.shape
        
        if cell_ids is None:
            cell_ids = [f"cell_{i}" for i in range(n_cells)]
        
        # Select root cell
        if root_cell is None:
            if self.root_selection_method == "automatic":
                # Use cell with minimum mean expression as root
                root_cell = int(np.argmin(np.mean(data_array, axis=1)))
            else:
                root_cell = 0
        
        # Build KNN graph
        knn_graph = self._build_knn_graph(data_array)
        
        # Compute pseudotime
        pseudotime = self._compute_diffusion_pseudotime(knn_graph, root_cell)
        
        # Detect branch points
        branch_points = self._detect_branch_points(pseudotime, knn_graph)
        
        # Build trajectory graph
        nodes, edges = self._build_trajectory_graph(pseudotime, branch_points)
        
        return {
            "status": "success",
            "algorithm": self.algorithm.value,
            "n_cells": n_cells,
            "n_features": n_features,
            "root_cell": root_cell,
            "pseudotime": {
                "values": pseudotime.tolist(),
                "cell_ids": cell_ids,
                "min": float(pseudotime.min()),
                "max": float(pseudotime.max())
            },
            "branch_points": branch_points,
            "graph": {
                "nodes": [n.to_dict() for n in nodes],
                "edges": [e.to_dict() for e in edges],
                "type": "trajectory_graph"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_compute_pseudotime(
        self,
        data: List[List[float]],
        root_cell: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute pseudotime ordering of cells."""
        data_array = np.array(data, dtype=np.float32)
        root_cell = root_cell or 0
        
        knn_graph = self._build_knn_graph(data_array)
        pseudotime = self._compute_diffusion_pseudotime(knn_graph, root_cell)
        
        # Compute pseudotime statistics
        ordering = np.argsort(pseudotime)
        
        return {
            "pseudotime": pseudotime.tolist(),
            "ordering": ordering.tolist(),
            "statistics": {
                "mean": float(np.mean(pseudotime)),
                "std": float(np.std(pseudotime)),
                "quartiles": [float(np.percentile(pseudotime, q)) for q in [25, 50, 75]]
            },
            "root_cell": root_cell
        }
    
    def _tool_detect_branches(
        self,
        data: List[List[float]],
        pseudotime: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Detect branching points in trajectory."""
        data_array = np.array(data, dtype=np.float32)
        
        if pseudotime is None:
            knn_graph = self._build_knn_graph(data_array)
            pseudotime_array = self._compute_diffusion_pseudotime(knn_graph)
        else:
            pseudotime_array = np.array(pseudotime)
        
        branch_points = self._detect_branch_points(pseudotime_array, self._build_knn_graph(data_array))
        
        return {
            "branch_points": branch_points,
            "n_branches": len(branch_points),
            "main_trajectory_length": 1.0,
            "branch_summary": {
                "early_branches": len([bp for bp in branch_points if bp["pseudotime"] < 0.3]),
                "mid_branches": len([bp for bp in branch_points if 0.3 <= bp["pseudotime"] < 0.7]),
                "late_branches": len([bp for bp in branch_points if bp["pseudotime"] >= 0.7])
            }
        }
    
    def _tool_lineage_analysis(
        self,
        data: List[List[float]],
        cell_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform lineage analysis."""
        data_array = np.array(data, dtype=np.float32)
        n_cells = data_array.shape[0]
        
        if cell_types is None:
            cell_types = [f"type_{i % 3}" for i in range(n_cells)]
        
        # Infer trajectory
        trajectory = self._tool_infer_trajectory(data.copy())
        pseudotime = np.array(trajectory["pseudotime"]["values"])
        
        # Analyze lineage composition
        unique_types = list(set(cell_types))
        lineage_composition = {}
        
        for cell_type in unique_types:
            type_indices = [i for i, t in enumerate(cell_types) if t == cell_type]
            type_pseudotime = pseudotime[type_indices]
            
            lineage_composition[cell_type] = {
                "count": len(type_indices),
                "mean_pseudotime": float(np.mean(type_pseudotime)),
                "std_pseudotime": float(np.std(type_pseudotime)),
                "pseudotime_range": [float(np.min(type_pseudotime)), float(np.max(type_pseudotime))]
            }
        
        # Sort by mean pseudotime to get lineage order
        lineage_order = sorted(unique_types, key=lambda t: lineage_composition[t]["mean_pseudotime"])
        
        # Infer transitions
        transitions = []
        for i in range(len(lineage_order) - 1):
            transitions.append({
                "from": lineage_order[i],
                "to": lineage_order[i + 1],
                "pseudotime_gap": lineage_composition[lineage_order[i + 1]]["mean_pseudotime"] - 
                                  lineage_composition[lineage_order[i]]["mean_pseudotime"]
            })
        
        return {
            "lineage_order": lineage_order,
            "lineage_composition": lineage_composition,
            "transitions": transitions,
            "n_cell_types": len(unique_types),
            "trajectory_summary": {
                "n_branch_points": len(trajectory["branch_points"]),
                "total_cells": n_cells
            }
        }
    
    def _tool_trajectory_diff_expression(
        self,
        data: List[List[float]],
        gene_names: Optional[List[str]] = None,
        n_top_genes: int = 20
    ) -> Dict[str, Any]:
        """Find differentially expressed genes along trajectory."""
        data_array = np.array(data, dtype=np.float32)
        n_cells, n_genes = data_array.shape
        
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Compute pseudotime
        knn_graph = self._build_knn_graph(data_array)
        pseudotime = self._compute_diffusion_pseudotime(knn_graph)
        
        # Compute correlation of each gene with pseudotime
        correlations = []
        for g in range(n_genes):
            corr = np.corrcoef(data_array[:, g], pseudotime)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append({
                "gene": gene_names[g],
                "correlation": float(corr),
                "abs_correlation": abs(float(corr)),
                "direction": "increasing" if corr > 0 else "decreasing"
            })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
        top_genes = correlations[:n_top_genes]
        
        return {
            "top_dynamic_genes": top_genes,
            "increasing_genes": [g for g in top_genes if g["direction"] == "increasing"],
            "decreasing_genes": [g for g in top_genes if g["direction"] == "decreasing"],
            "summary": {
                "total_genes": n_genes,
                "genes_analyzed": n_genes,
                "top_n_returned": n_top_genes
            }
        }
    
    def _tool_generate_visualization(
        self,
        data: List[List[float]],
        method: str = "umap"
    ) -> Dict[str, Any]:
        """Generate trajectory visualization data."""
        data_array = np.array(data, dtype=np.float32)
        n_cells = data_array.shape[0]
        
        # Infer trajectory
        trajectory = self._tool_infer_trajectory(data.copy())
        pseudotime = trajectory["pseudotime"]["values"]
        
        # Generate 2D coordinates (placeholder - would use UMAP/t-SNE in production)
        np.random.seed(42)
        coords_2d = np.random.randn(n_cells, 2)
        
        # Order by pseudotime for visualization
        ordering = np.argsort(pseudotime)
        
        # Create visualization data
        viz_data = {
            "coordinates": {
                "x": coords_2d[:, 0].tolist(),
                "y": coords_2d[:, 1].tolist()
            },
            "pseudotime": pseudotime,
            "graph": trajectory["graph"],
            "branch_points": trajectory["branch_points"],
            "visualization_config": {
                "method": method,
                "colorby": "pseudotime",
                "point_size": 3,
                "line_width": 1.5
            },
            "trajectory_path": {
                "ordered_cells": ordering.tolist(),
                "path_coordinates": {
                    "x": coords_2d[ordering, 0].tolist(),
                    "y": coords_2d[ordering, 1].tolist()
                }
            }
        }
        
        return viz_data
    
    def _handle_custom_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom methods for trajectory server."""
        if method == "trajectory/infer":
            return self._tool_infer_trajectory(
                params.get("data", []),
                params.get("cell_ids"),
                params.get("root_cell")
            )
        elif method == "trajectory/pseudotime":
            return self._tool_compute_pseudotime(
                params.get("data", []),
                params.get("root_cell")
            )
        elif method == "trajectory/branches":
            return self._tool_detect_branches(
                params.get("data", []),
                params.get("pseudotime")
            )
        elif method == "trajectory/lineage":
            return self._tool_lineage_analysis(
                params.get("data", []),
                params.get("cell_types")
            )
        elif method == "trajectory/diff_expression":
            return self._tool_trajectory_diff_expression(
                params.get("data", []),
                params.get("gene_names"),
                params.get("n_top_genes", 20)
            )
        elif method == "trajectory/visualize":
            return self._tool_generate_visualization(
                params.get("data", []),
                params.get("method", "umap")
            )
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    server = TrajectoryMCPServer()
    server.run()

