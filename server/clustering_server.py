import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from server.base import MCPServer


class ClusteringAlgorithm(Enum):
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    LEIDEN = "leiden"
    LOUVAIN = "louvain"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"


@dataclass
class ClusterInfo:
    """Information about a cluster."""
    id: str
    name: str
    size: int
    centroid: List[float]
    members: List[int]
    statistics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "centroid": self.centroid,
            "members": self.members,
            "statistics": self.statistics
        }


class ClusteringMCPServer(MCPServer):
    """
    MCP Server for Clustering Analysis.
    
    This server handles:
    - K-means and hierarchical clustering
    - Community detection (Leiden/Louvain)
    - Cluster quality metrics
    - Cluster visualization data
    - Differential expression between clusters
    """
    
    SERVER_KEY = "clustering"
    
    def __init__(
        self,
        algorithm: str = "leiden",
        resolution: float = 1.0,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ) -> None:
        """Initialize the Clustering MCP Server."""
        self.algorithm = ClusteringAlgorithm(algorithm)
        self.resolution = resolution
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Default Redis keys
        task_queue_key = kwargs.pop("task_queue_key", "mochiagent:clustering:task")
        result_queue_key = kwargs.pop("result_queue_key", "mochiagent:clustering:result")
        
        super().__init__(
            task_queue_key=task_queue_key,
            result_queue_key=result_queue_key,
            **kwargs
        )
    
    def _register_tools(self) -> None:
        """Register clustering tools."""
        self.register_tool(
            "cluster",
            self._tool_cluster,
            "Perform clustering on data"
        )
        self.register_tool(
            "kmeans",
            self._tool_kmeans,
            "Perform K-means clustering"
        )
        self.register_tool(
            "hierarchical",
            self._tool_hierarchical,
            "Perform hierarchical clustering"
        )
        self.register_tool(
            "evaluate_clustering",
            self._tool_evaluate_clustering,
            "Evaluate clustering quality"
        )
        self.register_tool(
            "find_markers",
            self._tool_find_markers,
            "Find marker features for each cluster"
        )
        self.register_tool(
            "generate_visualization",
            self._tool_generate_visualization,
            "Generate cluster visualization data"
        )
    
    def _register_resources(self) -> None:
        """Register clustering resources."""
        self.register_resource(
            "clustering://config",
            {
                "algorithm": self.algorithm.value,
                "resolution": self.resolution,
                "n_clusters": self.n_clusters,
                "random_state": self.random_state
            }
        )
        self.register_resource(
            "clustering://algorithms",
            [alg.value for alg in ClusteringAlgorithm]
        )
    
    def _register_prompts(self) -> None:
        """Register clustering prompts."""
        self.register_prompt(
            "cluster_interpretation",
            "Interpret the following clustering results:\n{results}\n\nContext: {context}"
        )
        self.register_prompt(
            "marker_annotation",
            "Annotate clusters based on their marker features:\n{markers}"
        )
    
    def _compute_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n_samples = data.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(data[i] - data[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _kmeans_clustering(
        self,
        data: np.ndarray,
        n_clusters: int,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform K-means clustering."""
        n_samples, n_features = data.shape
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        centroid_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = data[centroid_indices].copy()
        
        labels = np.zeros(n_samples, dtype=int)
        
        for iteration in range(max_iterations):
            # Assign samples to nearest centroid
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = [np.linalg.norm(data[i] - c) for c in centroids]
                new_labels[i] = np.argmin(distances)
            
            # Check for convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update centroids
            for k in range(n_clusters):
                cluster_points = data[labels == k]
                if len(cluster_points) > 0:
                    centroids[k] = np.mean(cluster_points, axis=0)
        
        return labels, centroids
    
    def _hierarchical_clustering(
        self,
        data: np.ndarray,
        n_clusters: int,
        linkage: str = "average"
    ) -> np.ndarray:
        """Perform hierarchical clustering."""
        n_samples = data.shape[0]
        
        # Start with each sample as its own cluster
        cluster_assignments = np.arange(n_samples)
        current_n_clusters = n_samples
        
        # Compute initial distance matrix
        distances = self._compute_distance_matrix(data)
        np.fill_diagonal(distances, np.inf)
        
        while current_n_clusters > n_clusters:
            # Find closest clusters
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            c1, c2 = min_idx
            
            # Merge clusters
            cluster_assignments[cluster_assignments == c2] = c1
            
            # Update distance matrix
            for i in range(n_samples):
                if i != c1 and i != c2:
                    if linkage == "average":
                        new_dist = (distances[c1, i] + distances[c2, i]) / 2
                    elif linkage == "single":
                        new_dist = min(distances[c1, i], distances[c2, i])
                    else:  # complete
                        new_dist = max(distances[c1, i], distances[c2, i])
                    distances[c1, i] = new_dist
                    distances[i, c1] = new_dist
            
            distances[c2, :] = np.inf
            distances[:, c2] = np.inf
            
            current_n_clusters -= 1
        
        # Relabel clusters to be consecutive
        unique_clusters = np.unique(cluster_assignments)
        label_map = {old: new for new, old in enumerate(unique_clusters)}
        labels = np.array([label_map[c] for c in cluster_assignments])
        
        return labels
    
    def _leiden_clustering(
        self,
        data: np.ndarray,
        resolution: float = 1.0
    ) -> np.ndarray:
        """Simulate Leiden clustering (placeholder for actual implementation)."""
        # Build KNN graph
        n_samples = data.shape[0]
        k = min(15, n_samples - 1)
        
        distances = self._compute_distance_matrix(data)
        knn = np.argsort(distances, axis=1)[:, 1:k+1]
        
        # Initialize each node as its own community
        communities = np.arange(n_samples)
        
        # Iterative community detection (simplified)
        for _ in range(10):
            for i in range(n_samples):
                neighbor_communities = communities[knn[i]]
                unique, counts = np.unique(neighbor_communities, return_counts=True)
                
                # Move to most common neighbor community with probability based on resolution
                if len(unique) > 0:
                    best_community = unique[np.argmax(counts)]
                    if np.random.random() < resolution * counts.max() / k:
                        communities[i] = best_community
        
        # Relabel to consecutive integers
        unique_communities = np.unique(communities)
        label_map = {old: new for new, old in enumerate(unique_communities)}
        labels = np.array([label_map[c] for c in communities])
        
        return labels
    
    def _compute_cluster_statistics(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compute statistics for each cluster."""
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_data = data[mask]
            
            cluster_stats[int(label)] = {
                "size": int(np.sum(mask)),
                "mean": np.mean(cluster_data, axis=0).tolist(),
                "std": np.std(cluster_data, axis=0).tolist(),
                "min": np.min(cluster_data, axis=0).tolist(),
                "max": np.max(cluster_data, axis=0).tolist()
            }
        
        return cluster_stats
    
    def _compute_silhouette_score(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute silhouette score for clustering quality."""
        n_samples = data.shape[0]
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        silhouette_values = []
        distances = self._compute_distance_matrix(data)
        
        for i in range(n_samples):
            # Compute a(i) - mean intra-cluster distance
            same_cluster = labels == labels[i]
            same_cluster[i] = False
            if np.sum(same_cluster) > 0:
                a_i = np.mean(distances[i, same_cluster])
            else:
                a_i = 0
            
            # Compute b(i) - mean nearest-cluster distance
            b_i = np.inf
            for label in unique_labels:
                if label != labels[i]:
                    other_cluster = labels == label
                    if np.sum(other_cluster) > 0:
                        mean_dist = np.mean(distances[i, other_cluster])
                        b_i = min(b_i, mean_dist)
            
            if b_i == np.inf:
                b_i = 0
            
            # Compute silhouette
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            silhouette_values.append(s_i)
        
        return float(np.mean(silhouette_values))
    
    def _tool_cluster(
        self,
        data: List[List[float]],
        algorithm: Optional[str] = None,
        n_clusters: Optional[int] = None,
        resolution: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform clustering on data.
        
        Args:
            data: 2D data matrix
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (for K-means, hierarchical)
            resolution: Resolution parameter (for Leiden/Louvain)
            
        Returns:
            Clustering results
        """
        self._log_info("Performing clustering analysis")
        
        data_array = np.array(data, dtype=np.float32)
        n_samples, n_features = data_array.shape
        
        algorithm = algorithm or self.algorithm.value
        n_clusters = n_clusters or self.n_clusters or min(10, n_samples // 5)
        resolution = resolution or self.resolution
        
        # Perform clustering based on algorithm
        if algorithm in ["kmeans", ClusteringAlgorithm.KMEANS.value]:
            labels, centroids = self._kmeans_clustering(data_array, n_clusters)
        elif algorithm in ["hierarchical", ClusteringAlgorithm.HIERARCHICAL.value]:
            labels = self._hierarchical_clustering(data_array, n_clusters)
            centroids = np.array([
                np.mean(data_array[labels == k], axis=0)
                for k in range(n_clusters)
            ])
        elif algorithm in ["leiden", ClusteringAlgorithm.LEIDEN.value]:
            labels = self._leiden_clustering(data_array, resolution)
            n_clusters = len(np.unique(labels))
            centroids = np.array([
                np.mean(data_array[labels == k], axis=0)
                for k in range(n_clusters)
            ])
        else:
            # Default to K-means
            labels, centroids = self._kmeans_clustering(data_array, n_clusters)
        
        # Compute statistics
        cluster_stats = self._compute_cluster_statistics(data_array, labels)
        silhouette = self._compute_silhouette_score(data_array, labels)
        
        # Build cluster info
        clusters = []
        for k in range(len(np.unique(labels))):
            cluster_members = np.where(labels == k)[0].tolist()
            cluster = ClusterInfo(
                id=f"cluster_{k}",
                name=f"Cluster {k}",
                size=len(cluster_members),
                centroid=centroids[k].tolist() if k < len(centroids) else [],
                members=cluster_members,
                statistics=cluster_stats.get(k, {})
            )
            clusters.append(cluster)
        
        return {
            "status": "success",
            "algorithm": algorithm,
            "n_clusters": len(clusters),
            "n_samples": n_samples,
            "labels": labels.tolist(),
            "clusters": [c.to_dict() for c in clusters],
            "centroids": centroids.tolist(),
            "quality_metrics": {
                "silhouette_score": silhouette,
                "inertia": float(np.sum([
                    np.sum((data_array[labels == k] - centroids[k]) ** 2)
                    for k in range(len(centroids))
                ]))
            },
            "graph": {
                "clusters": {c.id: c.members for c in clusters},
                "centroids": centroids.tolist(),
                "type": "cluster_graph"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_kmeans(
        self,
        data: List[List[float]],
        n_clusters: int = 5,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Perform K-means clustering."""
        return self._tool_cluster(data, algorithm="kmeans", n_clusters=n_clusters)
    
    def _tool_hierarchical(
        self,
        data: List[List[float]],
        n_clusters: int = 5,
        linkage: str = "average"
    ) -> Dict[str, Any]:
        """Perform hierarchical clustering."""
        return self._tool_cluster(data, algorithm="hierarchical", n_clusters=n_clusters)
    
    def _tool_evaluate_clustering(
        self,
        data: List[List[float]],
        labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate clustering quality."""
        data_array = np.array(data, dtype=np.float32)
        labels_array = np.array(labels, dtype=int)
        
        silhouette = self._compute_silhouette_score(data_array, labels_array)
        
        # Compute additional metrics
        n_clusters = len(np.unique(labels_array))
        cluster_sizes = [int(np.sum(labels_array == k)) for k in range(n_clusters)]
        
        # Compute intra-cluster variance
        intra_variances = []
        for k in range(n_clusters):
            cluster_data = data_array[labels_array == k]
            if len(cluster_data) > 0:
                variance = np.mean(np.var(cluster_data, axis=0))
                intra_variances.append(float(variance))
        
        return {
            "silhouette_score": silhouette,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "size_balance": float(np.std(cluster_sizes) / np.mean(cluster_sizes)) if cluster_sizes else 0,
            "mean_intra_cluster_variance": float(np.mean(intra_variances)) if intra_variances else 0,
            "quality_assessment": "good" if silhouette > 0.5 else "moderate" if silhouette > 0.25 else "poor"
        }
    
    def _tool_find_markers(
        self,
        data: List[List[float]],
        labels: List[int],
        feature_names: Optional[List[str]] = None,
        n_top_markers: int = 10
    ) -> Dict[str, Any]:
        """Find marker features for each cluster."""
        data_array = np.array(data, dtype=np.float32)
        labels_array = np.array(labels, dtype=int)
        n_features = data_array.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        unique_labels = np.unique(labels_array)
        cluster_markers = {}
        
        for label in unique_labels:
            cluster_mask = labels_array == label
            other_mask = ~cluster_mask
            
            cluster_data = data_array[cluster_mask]
            other_data = data_array[other_mask]
            
            # Compute fold change and significance for each feature
            markers = []
            for f in range(n_features):
                cluster_mean = np.mean(cluster_data[:, f])
                other_mean = np.mean(other_data[:, f])
                
                fold_change = (cluster_mean + 1e-10) / (other_mean + 1e-10)
                log_fc = np.log2(fold_change)
                
                # Simple t-statistic
                cluster_std = np.std(cluster_data[:, f])
                other_std = np.std(other_data[:, f])
                n1, n2 = len(cluster_data), len(other_data)
                
                pooled_se = np.sqrt((cluster_std**2 / n1) + (other_std**2 / n2) + 1e-10)
                t_stat = (cluster_mean - other_mean) / pooled_se
                
                markers.append({
                    "feature": feature_names[f],
                    "feature_index": f,
                    "cluster_mean": float(cluster_mean),
                    "other_mean": float(other_mean),
                    "log2_fold_change": float(log_fc),
                    "t_statistic": float(t_stat),
                    "is_upregulated": log_fc > 0
                })
            
            # Sort by absolute t-statistic
            markers.sort(key=lambda x: abs(x["t_statistic"]), reverse=True)
            cluster_markers[int(label)] = markers[:n_top_markers]
        
        return {
            "cluster_markers": cluster_markers,
            "n_clusters": len(unique_labels),
            "n_top_markers": n_top_markers,
            "total_features": n_features
        }
    
    def _tool_generate_visualization(
        self,
        data: List[List[float]],
        labels: Optional[List[int]] = None,
        method: str = "pca"
    ) -> Dict[str, Any]:
        """Generate cluster visualization data."""
        data_array = np.array(data, dtype=np.float32)
        n_samples = data_array.shape[0]
        
        # Perform clustering if labels not provided
        if labels is None:
            result = self._tool_cluster(data)
            labels = result["labels"]
        
        # Generate 2D coordinates (simplified PCA)
        centered = data_array - np.mean(data_array, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            top_2_eigenvectors = eigenvectors[:, idx[:2]]
            coords_2d = centered @ top_2_eigenvectors
        except:
            # Fallback to random projection
            np.random.seed(self.random_state)
            coords_2d = np.random.randn(n_samples, 2)
        
        # Compute cluster centroids in 2D
        unique_labels = np.unique(labels)
        centroids_2d = []
        for label in unique_labels:
            mask = np.array(labels) == label
            centroid = np.mean(coords_2d[mask], axis=0)
            centroids_2d.append(centroid.tolist())
        
        return {
            "coordinates": {
                "x": coords_2d[:, 0].tolist(),
                "y": coords_2d[:, 1].tolist()
            },
            "labels": labels if isinstance(labels, list) else labels.tolist(),
            "centroids_2d": centroids_2d,
            "visualization_config": {
                "method": method,
                "colorby": "cluster",
                "point_size": 5,
                "show_centroids": True
            },
            "cluster_colors": {
                int(label): f"hsl({int(label * 360 / len(unique_labels))}, 70%, 50%)"
                for label in unique_labels
            }
        }
    
    def _handle_custom_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom methods for clustering server."""
        if method == "clustering/cluster":
            return self._tool_cluster(
                params.get("data", []),
                params.get("algorithm"),
                params.get("n_clusters"),
                params.get("resolution")
            )
        elif method == "clustering/kmeans":
            return self._tool_kmeans(
                params.get("data", []),
                params.get("n_clusters", 5),
                params.get("max_iterations", 100)
            )
        elif method == "clustering/hierarchical":
            return self._tool_hierarchical(
                params.get("data", []),
                params.get("n_clusters", 5),
                params.get("linkage", "average")
            )
        elif method == "clustering/evaluate":
            return self._tool_evaluate_clustering(
                params.get("data", []),
                params.get("labels", [])
            )
        elif method == "clustering/markers":
            return self._tool_find_markers(
                params.get("data", []),
                params.get("labels", []),
                params.get("feature_names"),
                params.get("n_top_markers", 10)
            )
        elif method == "clustering/visualize":
            return self._tool_generate_visualization(
                params.get("data", []),
                params.get("labels"),
                params.get("method", "pca")
            )
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    server = ClusteringMCPServer()
    server.run()

