from typing import Dict, List, Optional, Iterator
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import itertools

class ModelManager:
    """Manages different language models from Hugging Face."""
    
    def __init__(self):
        self.models: Dict[str, HuggingFacePipeline] = {}
        self.model_configs: Dict[str, Dict] = {}
        self._model_iterator = None
        
    def add_model(self, 
                 name: str, 
                 model_id: str, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 temperature: float = 0.2,
                 max_length: int = 512,
                 batch_size: int = 1) -> None:
        """Add a new model to the manager.
        
        Args:
            name: Unique identifier for the model
            model_id: Hugging Face model ID
            device: Device to run the model on
            temperature: Sampling temperature
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            batch_size=batch_size,
            device=0 if device == "cuda" else -1
        )
        
        # Create LangChain wrapper
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Store model and config
        self.models[name] = llm
        self.model_configs[name] = {
            "model_id": model_id,
            "device": device,
            "temperature": temperature,
            "max_length": max_length,
            "batch_size": batch_size
        }
        
        # Reset iterator when models change
        self._reset_iterator()
    
    def get_model(self, name: str) -> Optional[HuggingFacePipeline]:
        """Get a model by name."""
        return self.models.get(name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            self._reset_iterator()
    
    def clear(self) -> None:
        """Clear all models from the manager."""
        self.models.clear()
        self.model_configs.clear()
        self._reset_iterator()
    
    def _reset_iterator(self) -> None:
        """Reset the model iterator."""
        if self.models:
            self._model_iterator = itertools.cycle(self.models.keys())
        else:
            self._model_iterator = None
    
    def get_next_model(self) -> Optional[HuggingFacePipeline]:
        """Get the next model in the round-robin sequence.
        
        Returns:
            The next model in the sequence, or None if no models are available.
        """
        if not self.models:
            return None
            
        if self._model_iterator is None:
            self._reset_iterator()
            
        next_model_name = next(self._model_iterator)
        return self.models[next_model_name]
    
    def get_model_iterator(self) -> Iterator[HuggingFacePipeline]:
        """Get an iterator that cycles through all available models.
        
        Returns:
            An iterator that yields models in a round-robin fashion.
        """
        # Create a fresh iterator
        self._reset_iterator()
        
        # Return an iterator that yields models
        while self._model_iterator:
            try:
                model_name = next(self._model_iterator)
                yield self.models[model_name]
            except StopIteration:
                return

# Example usage:
if __name__ == "__main__":
    # Initialize model manager
    manager = ModelManager()
    
    # Add some popular open-source models
    manager.add_model(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-v0.1",
        temperature=0.2
    )
    
    manager.add_model(
        name="llama2-7b",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.2
    )
    
    manager.add_model(
        name="zephyr-7b",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.2
    )
    
    # Print available models
    print("Available models:", manager.get_available_models())
    
    # Demo the round-robin assignment
    for i in range(5):
        model = manager.get_next_model()
        print(f"Agent {i} gets model: {model}") 