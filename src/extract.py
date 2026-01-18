"""
Hidden state extraction for TDA analysis on reasoning models.
Target: Qwen/Qwen3-4B-Thinking-2507
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ExtractionConfig:
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    layers_to_capture: Optional[List[int]] = None  # None = all layers
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16


class HiddenStateExtractor:
    """
    Extracts hidden states from specified layers during generation.
    Each token's hidden state vector becomes a point in the point cloud.
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.hidden_states: Dict[int, List[torch.Tensor]] = {}  # layer_idx -> list of hidden states per token
        self._hooks = []
        
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map="auto"
        )
        self.model.eval()
        
        # Get model info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        print(f"Model loaded: {self.num_layers} layers, hidden dim {self.hidden_dim}")
        
        # Default: first 5 and last 5 layers
        if config.layers_to_capture is None:
            self.layers_to_capture = list(range(5)) + list(range(self.num_layers - 5, self.num_layers))
        else:
            self.layers_to_capture = config.layers_to_capture
        
        print(f"Capturing layers: {self.layers_to_capture}")
    
    def _make_hook(self, layer_idx: int):
        """Create a hook that captures the hidden state output of a layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # hidden shape: (batch, seq_len, hidden_dim)
            # We only care about the last token during generation
            last_token_hidden = hidden[:, -1, :].detach().cpu()
            
            if layer_idx not in self.hidden_states:
                self.hidden_states[layer_idx] = []
            self.hidden_states[layer_idx].append(last_token_hidden)
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        self._clear_hooks()
        
        for layer_idx in self.layers_to_capture:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def _clear_hidden_states(self):
        """Clear stored hidden states."""
        self.hidden_states = {}
    
    def extract(self, prompt: str, max_new_tokens: int = 512) -> Dict[str, any]:
        """
        Run generation and extract hidden states.
        
        Returns:
            Dict with:
                - 'generated_text': the full generated text
                - 'hidden_states': Dict[layer_idx, np.ndarray of shape (num_tokens, hidden_dim)]
                - 'num_tokens': number of generated tokens
        """
        self._clear_hidden_states()
        self._register_hooks()
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_len = model_inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )
        
        self._clear_hooks()
        
        # Decode output
        output_ids = generated_ids[0][input_len:].tolist()
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Convert hidden states to numpy arrays
        # Shape per layer: (num_generated_tokens, hidden_dim)
        hidden_states_np = {}
        for layer_idx, states in self.hidden_states.items():
            # states is a list of tensors, each (1, hidden_dim)
            stacked = torch.cat(states, dim=0).numpy()
            hidden_states_np[layer_idx] = stacked
        
        num_tokens = len(output_ids)
        
        return {
            'generated_text': generated_text,
            'hidden_states': hidden_states_np,
            'num_tokens': num_tokens,
            'prompt': prompt,
        }
    
    def get_point_cloud(self, extraction_result: Dict, layer_idx: int) -> np.ndarray:
        """
        Get the point cloud for a specific layer.
        Each row is a point (one token's hidden state).
        
        Returns:
            np.ndarray of shape (num_tokens, hidden_dim)
        """
        return extraction_result['hidden_states'][layer_idx]


def main():
    """Quick test."""
    config = ExtractionConfig()
    extractor = HiddenStateExtractor(config)
    
    # Test prompts
    prompts = [
        "What is the capital of France?",  # Simple factual
        "If Alice is taller than Bob and Bob is taller than Carol, who is shortest?",  # Reasoning
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print('='*60)
        
        result = extractor.extract(prompt, max_new_tokens=256)
        
        print(f"Generated {result['num_tokens']} tokens")
        print(f"Response preview: {result['generated_text'][:200]}...")
        
        # Show point cloud shapes per layer
        for layer_idx in sorted(result['hidden_states'].keys()):
            pc = extractor.get_point_cloud(result, layer_idx)
            print(f"  Layer {layer_idx}: point cloud shape {pc.shape}")


if __name__ == "__main__":
    main()
