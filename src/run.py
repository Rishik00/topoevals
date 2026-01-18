"""
Main runner for topoevals experiments.
Extracts hidden states and runs TDA analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from extract import ExtractionConfig, HiddenStateExtractor
from tda import TDAAnalyzer, analyze_extraction, print_analysis_summary, PersistenceResult


def run_experiment(
    prompts: List[str],
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    layers: Optional[List[int]] = None,
    max_new_tokens: int = 512,
    max_homology_dim: int = 1,
    output_dir: str = "results",
) -> Dict:
    """
    Run a full extraction + TDA experiment.
    
    Args:
        prompts: List of prompts to test
        model_name: HuggingFace model identifier
        layers: Which layers to capture (None = first/last 5)
        max_new_tokens: Max tokens to generate per prompt
        max_homology_dim: 0 = components only, 1 = + loops, 2 = + voids
        output_dir: Where to save results
    
    Returns:
        Dict with all results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize extractor
    config = ExtractionConfig(
        model_name=model_name,
        layers_to_capture=layers,
    )
    extractor = HiddenStateExtractor(config)
    
    # Initialize TDA analyzer
    tda = TDAAnalyzer(max_dim=max_homology_dim)
    
    # Run experiments
    all_results = {
        'metadata': {
            'model_name': model_name,
            'timestamp': timestamp,
            'layers_captured': extractor.layers_to_capture,
            'max_homology_dim': max_homology_dim,
            'num_layers': extractor.num_layers,
            'hidden_dim': extractor.hidden_dim,
        },
        'experiments': []
    }
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print('='*60)
        
        # Extract hidden states
        extraction = extractor.extract(prompt, max_new_tokens=max_new_tokens)
        print(f"Generated {extraction['num_tokens']} tokens")
        
        # Run TDA on each layer
        persistence_results = analyze_extraction(
            extraction,
            layers=extractor.layers_to_capture,
            max_dim=max_homology_dim,
        )
        
        # Print summary
        print_analysis_summary(persistence_results, prompt)
        
        # Collect results for this prompt
        experiment_result = {
            'prompt': prompt,
            'generated_text': extraction['generated_text'],
            'num_tokens': extraction['num_tokens'],
            'tda_results': {}
        }
        
        for layer_idx, pers_result in persistence_results.items():
            experiment_result['tda_results'][layer_idx] = {
                'num_points': pers_result.num_points,
                'point_cloud_dim': pers_result.point_cloud_dim,
            }
            for dim in range(max_homology_dim + 1):
                stats = pers_result.summary_stats(dim)
                experiment_result['tda_results'][layer_idx][f'H{dim}'] = stats
        
        all_results['experiments'].append(experiment_result)
    
    # Save results
    results_path = Path(output_dir) / f"results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return all_results


def compare_prompts(results: Dict) -> None:
    """
    Compare TDA results across different prompts.
    Prints a comparison table.
    """
    experiments = results['experiments']
    if len(experiments) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    print("\n" + "="*80)
    print("COMPARISON ACROSS PROMPTS")
    print("="*80)
    
    # Get layers
    first_exp = experiments[0]
    layers = sorted(first_exp['tda_results'].keys())
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        print(f"{'Prompt':<50} {'H0 mean':<12} {'H0 max':<12} {'H1 mean':<12} {'H1 max':<12}")
        print("-" * 98)
        
        for exp in experiments:
            prompt_short = exp['prompt'][:47] + "..." if len(exp['prompt']) > 50 else exp['prompt']
            tda = exp['tda_results'].get(layer, {})
            
            h0 = tda.get('H0', {})
            h1 = tda.get('H1', {})
            
            h0_mean = h0.get('mean_lifetime', 0)
            h0_max = h0.get('max_lifetime', 0)
            h1_mean = h1.get('mean_lifetime', 0)
            h1_max = h1.get('max_lifetime', 0)
            
            print(f"{prompt_short:<50} {h0_mean:<12.4f} {h0_max:<12.4f} {h1_mean:<12.4f} {h1_max:<12.4f}")


# Default test prompts
DEFAULT_PROMPTS = [
    # Simple factual (likely memorized)
    "What is the capital of France?",
    
    # Reasoning required
    "If Alice is taller than Bob and Bob is taller than Carol, who is shortest?",
    
    # More complex reasoning
    "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
    
    # Open-ended (no clear right answer)
    "What are some interesting properties of prime numbers?",
]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run topoevals experiments")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompts to test")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--max-dim", type=int, default=1, help="Max homology dimension (0, 1, or 2)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    
    results = run_experiment(
        prompts=prompts,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        max_homology_dim=args.max_dim,
        output_dir=args.output_dir,
    )
    
    compare_prompts(results)
