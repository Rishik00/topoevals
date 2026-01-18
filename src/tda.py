"""
TDA pipeline for analyzing point clouds from hidden states.
Uses Ripser for Rips complex filtration and persistence diagram computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not installed. Run: pip install ripser")

try:
    from persim import plot_diagrams, wasserstein, bottleneck
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False
    warnings.warn("persim not installed. Run: pip install persim")


@dataclass 
class PersistenceResult:
    """Container for persistence computation results."""
    diagrams: List[np.ndarray]  # List of diagrams for each homology dimension
    num_points: int
    point_cloud_dim: int
    max_dim: int
    
    def get_lifetimes(self, dim: int = 0) -> np.ndarray:
        """Get lifetimes (death - birth) for features in given dimension."""
        if dim >= len(self.diagrams):
            return np.array([])
        dgm = self.diagrams[dim]
        # Filter out infinite death times
        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) == 0:
            return np.array([])
        return finite_dgm[:, 1] - finite_dgm[:, 0]
    
    def get_birth_death(self, dim: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get birth and death times separately."""
        if dim >= len(self.diagrams):
            return np.array([]), np.array([])
        dgm = self.diagrams[dim]
        return dgm[:, 0], dgm[:, 1]
    
    def summary_stats(self, dim: int = 0) -> Dict:
        """Compute summary statistics for a homology dimension."""
        lifetimes = self.get_lifetimes(dim)
        if len(lifetimes) == 0:
            return {
                'num_features': 0,
                'mean_lifetime': 0,
                'max_lifetime': 0,
                'std_lifetime': 0,
                'total_persistence': 0,
            }
        return {
            'num_features': len(lifetimes),
            'mean_lifetime': float(np.mean(lifetimes)),
            'max_lifetime': float(np.max(lifetimes)),
            'std_lifetime': float(np.std(lifetimes)),
            'total_persistence': float(np.sum(lifetimes)),
        }


class TDAAnalyzer:
    """
    Computes Rips filtration and persistence diagrams on point clouds.
    """
    
    def __init__(self, max_dim: int = 1, max_edge_length: float = np.inf, n_threads: int = -1):
        """
        Args:
            max_dim: Maximum homology dimension to compute (0 = components, 1 = loops, 2 = voids)
            max_edge_length: Maximum edge length to include in Rips complex
            n_threads: Number of threads for computation (-1 = all)
        """
        if not RIPSER_AVAILABLE:
            raise ImportError("ripser is required. Install with: pip install ripser")
        
        self.max_dim = max_dim
        self.max_edge_length = max_edge_length
        self.n_threads = n_threads
    
    def compute_persistence(self, point_cloud: np.ndarray) -> PersistenceResult:
        """
        Compute persistence diagrams for a point cloud.
        
        Args:
            point_cloud: np.ndarray of shape (num_points, dimension)
        
        Returns:
            PersistenceResult containing diagrams and metadata
        """
        num_points, point_dim = point_cloud.shape
        
        # Run Ripser
        result = ripser(
            point_cloud,
            maxdim=self.max_dim,
            thresh=self.max_edge_length,
        )
        
        return PersistenceResult(
            diagrams=result['dgms'],
            num_points=num_points,
            point_cloud_dim=point_dim,
            max_dim=self.max_dim,
        )
    
    def compare_diagrams(
        self, 
        result1: PersistenceResult, 
        result2: PersistenceResult,
        dim: int = 0,
        metric: str = 'wasserstein'
    ) -> float:
        """
        Compare two persistence diagrams using Wasserstein or bottleneck distance.
        
        Args:
            result1, result2: PersistenceResult objects to compare
            dim: Homology dimension to compare
            metric: 'wasserstein' or 'bottleneck'
        
        Returns:
            Distance between diagrams
        """
        if not PERSIM_AVAILABLE:
            raise ImportError("persim is required for diagram comparison. Install with: pip install persim")
        
        dgm1 = result1.diagrams[dim]
        dgm2 = result2.diagrams[dim]
        
        if metric == 'wasserstein':
            return wasserstein(dgm1, dgm2)
        elif metric == 'bottleneck':
            return bottleneck(dgm1, dgm2)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def analyze_extraction(
    extraction_result: Dict,
    layers: Optional[List[int]] = None,
    max_dim: int = 1,
) -> Dict[int, PersistenceResult]:
    """
    Convenience function to analyze an extraction result.
    
    Args:
        extraction_result: Output from HiddenStateExtractor.extract()
        layers: Which layers to analyze (None = all available)
        max_dim: Maximum homology dimension
    
    Returns:
        Dict mapping layer_idx to PersistenceResult
    """
    analyzer = TDAAnalyzer(max_dim=max_dim)
    
    hidden_states = extraction_result['hidden_states']
    if layers is None:
        layers = list(hidden_states.keys())
    
    results = {}
    for layer_idx in layers:
        if layer_idx not in hidden_states:
            warnings.warn(f"Layer {layer_idx} not in extraction result")
            continue
        
        point_cloud = hidden_states[layer_idx]
        results[layer_idx] = analyzer.compute_persistence(point_cloud)
    
    return results


def print_analysis_summary(
    persistence_results: Dict[int, PersistenceResult],
    prompt: str = "",
):
    """Print a summary of persistence analysis across layers."""
    print(f"\n{'='*60}")
    if prompt:
        print(f"Prompt: {prompt[:50]}...")
    print('='*60)
    
    for layer_idx in sorted(persistence_results.keys()):
        result = persistence_results[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Point cloud: {result.num_points} points in R^{result.point_cloud_dim}")
        
        for dim in range(result.max_dim + 1):
            stats = result.summary_stats(dim)
            dim_name = {0: 'H0 (components)', 1: 'H1 (loops)', 2: 'H2 (voids)'}.get(dim, f'H{dim}')
            print(f"  {dim_name}:")
            print(f"    Features: {stats['num_features']}")
            print(f"    Mean lifetime: {stats['mean_lifetime']:.4f}")
            print(f"    Max lifetime: {stats['max_lifetime']:.4f}")
            print(f"    Total persistence: {stats['total_persistence']:.4f}")


# Quick test
if __name__ == "__main__":
    # Test with random point cloud
    print("Testing TDA pipeline with random point cloud...")
    
    np.random.seed(42)
    # Simulate a point cloud: 50 points in 100 dimensions (scaled down from 2560 for testing)
    test_cloud = np.random.randn(50, 100)
    
    analyzer = TDAAnalyzer(max_dim=1)
    result = analyzer.compute_persistence(test_cloud)
    
    print(f"Point cloud shape: {test_cloud.shape}")
    print(f"H0 features: {result.summary_stats(0)}")
    print(f"H1 features: {result.summary_stats(1)}")
    
    print("\nTDA pipeline working correctly!")
