from typing import Dict, List, Optional, Iterator
from langchain_community.llms import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

class ModelManager:
    """Manages different language models from Hugging Face."""
    
    def __init__(self):
        self.models: Dict[str, HuggingFaceEndpoint] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.iterator: Optional[Iterator[HuggingFaceEndpoint]] = None
        
    def add_model(self, 
                 name: str, 
                 model_id: str, 
                 temperature: float = 0.2,
                 max_length: int = 512,
                 batch_size: int = 1) -> None:
        """Add a new model to the manager.
        
        Args:
            name: Unique identifier for the model
            model_id: Hugging Face model ID
            temperature: Sampling temperature
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        print(f"Loading model {model_id}...")
        
        # Create a wrapper class for InferenceClient that is compatible with LangChain
        class InferenceClientWrapper:
            def __init__(self, model_id, api_token, temperature, max_new_tokens):
                self.client = InferenceClient(model=model_id, token=api_token)
                self.temperature = temperature
                self.max_new_tokens = max_new_tokens
                
            def invoke(self, messages):
                # Format the prompt from messages
                if isinstance(messages, list):
                    prompt = self._format_messages(messages)
                else:
                    prompt = messages
                    
                # Call the text_generation method directly
                response = self.client.text_generation(
                    prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    return_full_text=False,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
                return response
                
            def _format_messages(self, messages):
                formatted = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        formatted.append(f"System: {msg.content}")
                    elif isinstance(msg, HumanMessage):
                        formatted.append(f"Human: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        formatted.append(f"AI: {msg.content}")
                return "\n\n".join(formatted)
        
        # Create wrapper client
        llm = InferenceClientWrapper(
            model_id=model_id,
            api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            temperature=temperature,
            max_new_tokens=max_length
        )
        
        # Store model and config
        self.models[name] = llm
        self.model_configs[name] = {
            "model_id": model_id,
            "temperature": temperature,
            "max_length": max_length,
            "batch_size": batch_size
        }
        
        # Reset iterator when models change
        self._reset_iterator()
        
        print(f"Model {name} loaded successfully")
    
    def get_model(self, name: str) -> Optional[object]:
        """Get a model by name."""
        return self.models.get(name)
    
    def get_next_model(self) -> object:
        """Get the next model in the round-robin sequence."""
        if not self.models:
            raise ValueError("No models available")
            
        if self.iterator is None:
            self._reset_iterator()
            
        try:
            return next(self.iterator)
        except StopIteration:
            self._reset_iterator()
            return next(self.iterator)
    
    def _reset_iterator(self) -> None:
        """Reset the model iterator."""
        self.iterator = iter(self.models.values())
    
    def get_model_iterator(self) -> Iterator[object]:
        """Get an iterator over all models."""
        return iter(self.models.values())
    
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
        self.iterator = None

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