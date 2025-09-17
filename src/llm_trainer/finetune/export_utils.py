"""
Model export utilities for saving to various formats.

This module provides utilities for exporting trained models to different formats
including HuggingFace, GGUF, and Ollama integration.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import warnings

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Export functionality will be limited.")

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class ModelExporter:
    """Handles exporting models to various formats."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: Optional[str] = None
    ):
        """
        Initialize model exporter.
        
        Args:
            model: Model to export
            tokenizer: Tokenizer
            model_name: Optional model name for metadata
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or "fine-tuned-model"
    
    def save_huggingface(
        self,
        save_directory: Union[str, Path],
        merge_lora: bool = True,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save model in HuggingFace format.
        
        Args:
            save_directory: Directory to save to
            merge_lora: Whether to merge LoRA weights
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: Repository ID for Hub upload
            **kwargs: Additional save arguments
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Handle PEFT models
        if _PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            if merge_lora:
                print("🔄 Merging LoRA weights...")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(save_directory, **kwargs)
                print("✅ Saved merged model")
            else:
                print("💾 Saving LoRA adapters...")
                self.model.save_pretrained(save_directory, **kwargs)
                # Also save base model
                base_model_dir = save_directory / "base_model"
                self.model.base_model.save_pretrained(base_model_dir, **kwargs)
                print("✅ Saved LoRA adapters and base model")
        else:
            # Regular model
            self.model.save_pretrained(save_directory, **kwargs)
            print("✅ Saved model")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        print("✅ Saved tokenizer")
        
        # Create model card
        self._create_model_card(save_directory)
        
        # Push to hub if requested
        if push_to_hub:
            self._push_to_hub(save_directory, repo_id)
    
    def save_gguf(
        self,
        save_directory: Union[str, Path],
        quantization: str = "q4_k_m",
        merge_lora: bool = True
    ) -> None:
        """
        Export model to GGUF format for llama.cpp.
        
        Args:
            save_directory: Directory to save to
            quantization: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)
            merge_lora: Whether to merge LoRA weights first
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # First save as HuggingFace format
        hf_dir = save_directory / "hf_model"
        self.save_huggingface(hf_dir, merge_lora=merge_lora)
        
        # Convert to GGUF
        gguf_path = save_directory / f"{self.model_name}-{quantization}.gguf"
        
        try:
            # Try to use llama.cpp convert script
            self._convert_to_gguf(hf_dir, gguf_path, quantization)
            print(f"✅ Exported to GGUF: {gguf_path}")
        except Exception as e:
            print(f"❌ Failed to convert to GGUF: {e}")
            print("💡 Make sure llama.cpp is installed and in PATH")
    
    def save_ollama(
        self,
        model_name: str,
        save_directory: Union[str, Path],
        quantization: str = "q4_k_m",
        template: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Export model for Ollama.
        
        Args:
            model_name: Name for Ollama model
            save_directory: Directory to save to
            quantization: Quantization type
            template: Chat template
            system_prompt: System prompt
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # First export to GGUF
        self.save_gguf(save_directory, quantization)
        
        # Create Modelfile
        modelfile_path = save_directory / "Modelfile"
        gguf_file = save_directory / f"{self.model_name}-{quantization}.gguf"
        
        self._create_ollama_modelfile(
            modelfile_path,
            gguf_file,
            template,
            system_prompt
        )
        
        # Import to Ollama
        try:
            self._import_to_ollama(model_name, modelfile_path)
            print(f"✅ Imported to Ollama as: {model_name}")
        except Exception as e:
            print(f"❌ Failed to import to Ollama: {e}")
            print(f"💡 You can manually import with: ollama create {model_name} -f {modelfile_path}")
    
    def _create_model_card(self, save_directory: Path) -> None:
        """Create a model card README."""
        model_card = f"""---
license: apache-2.0
base_model: {getattr(self.model.config, '_name_or_path', 'unknown')}
tags:
- fine-tuned
- llm-trainer
library_name: transformers
---

# {self.model_name}

This model was fine-tuned using [LLM Trainer](https://github.com/HelpingAI/llm-trainer).

## Model Details

- **Base Model**: {getattr(self.model.config, '_name_or_path', 'unknown')}
- **Model Type**: {self.model.config.model_type}
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.model_name}")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

This model was fine-tuned using LLM Trainer with the following optimizations:
- Memory-efficient training techniques
- Gradient checkpointing
- Mixed precision training
- Custom kernel optimizations

## License

This model is licensed under the Apache 2.0 License.
"""
        
        readme_path = save_directory / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
    
    def _convert_to_gguf(
        self,
        hf_dir: Path,
        gguf_path: Path,
        quantization: str
    ) -> None:
        """Convert HuggingFace model to GGUF."""
        # Try different conversion methods
        conversion_commands = [
            # llama.cpp python script
            ["python", "-m", "llama_cpp.convert", str(hf_dir), "--outfile", str(gguf_path)],
            # llama.cpp convert script
            ["convert.py", str(hf_dir), "--outfile", str(gguf_path)],
            # Alternative convert script
            ["convert-hf-to-gguf.py", str(hf_dir), "--outfile", str(gguf_path)]
        ]
        
        for cmd in conversion_commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Quantize if needed
                if quantization != "f16":
                    quantize_cmd = [
                        "quantize",
                        str(gguf_path),
                        str(gguf_path.with_suffix(f".{quantization}.gguf")),
                        quantization
                    ]
                    subprocess.run(quantize_cmd, check=True, capture_output=True)
                    # Remove unquantized version
                    gguf_path.unlink()
                
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise RuntimeError("Could not find llama.cpp conversion tools")
    
    def _create_ollama_modelfile(
        self,
        modelfile_path: Path,
        gguf_file: Path,
        template: Optional[str],
        system_prompt: Optional[str]
    ) -> None:
        """Create Ollama Modelfile."""
        modelfile_content = f"FROM {gguf_file.absolute()}\n\n"
        
        if template:
            modelfile_content += f'TEMPLATE """{template}"""\n\n'
        elif hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            modelfile_content += f'TEMPLATE """{self.tokenizer.chat_template}"""\n\n'
        
        if system_prompt:
            modelfile_content += f'SYSTEM """{system_prompt}"""\n\n'
        
        # Add parameters
        modelfile_content += """PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
"""
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
    
    def _import_to_ollama(self, model_name: str, modelfile_path: Path) -> None:
        """Import model to Ollama."""
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        subprocess.run(cmd, check=True)
    
    def _push_to_hub(self, save_directory: Path, repo_id: Optional[str]) -> None:
        """Push model to HuggingFace Hub."""
        if not repo_id:
            repo_id = self.model_name
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id, exist_ok=True)
            except Exception:
                pass
            
            # Upload files
            api.upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                repo_type="model"
            )
            
            print(f"✅ Pushed to Hub: https://huggingface.co/{repo_id}")
            
        except ImportError:
            print("❌ huggingface_hub not available. Cannot push to Hub.")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")


def export_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Union[str, Path],
    formats: List[str] = ["huggingface"],
    model_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Export model to multiple formats.
    
    Args:
        model: Model to export
        tokenizer: Tokenizer
        output_dir: Output directory
        formats: List of formats to export ("huggingface", "gguf", "ollama")
        model_name: Model name
        **kwargs: Additional export arguments
    """
    output_dir = Path(output_dir)
    exporter = ModelExporter(model, tokenizer, model_name)
    
    for format_name in formats:
        format_dir = output_dir / format_name
        
        if format_name == "huggingface":
            exporter.save_huggingface(format_dir, **kwargs)
        elif format_name == "gguf":
            exporter.save_gguf(format_dir, **kwargs)
        elif format_name == "ollama":
            ollama_model_name = kwargs.get("ollama_model_name", model_name or "fine-tuned-model")
            exporter.save_ollama(ollama_model_name, format_dir, **kwargs)
        else:
            print(f"⚠️  Unknown format: {format_name}")


def create_deployment_package(
    model_dir: Union[str, Path],
    package_dir: Union[str, Path],
    include_examples: bool = True
) -> None:
    """
    Create a deployment package with model and usage examples.
    
    Args:
        model_dir: Directory containing the model
        package_dir: Directory to create package in
        include_examples: Whether to include usage examples
    """
    model_dir = Path(model_dir)
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    model_dest = package_dir / "model"
    shutil.copytree(model_dir, model_dest, dirs_exist_ok=True)
    
    if include_examples:
        # Create usage examples
        examples_dir = package_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Python example
        python_example = """#!/usr/bin/env python3
\"\"\"
Example usage of the fine-tuned model.
\"\"\"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Example generation
    prompt = "Hello, how can I help you today?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {prompt}")
    print(f"Output: {response}")

if __name__ == "__main__":
    main()
"""
        
        with open(examples_dir / "generate.py", 'w') as f:
            f.write(python_example)
        
        # Requirements file
        requirements = """torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
"""
        
        with open(package_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        # README
        readme = """# Fine-tuned Model Deployment Package

This package contains a fine-tuned language model and usage examples.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the example:
```bash
python examples/generate.py
```

## Usage

See the examples directory for usage examples in different scenarios.
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme)
    
    print(f"✅ Created deployment package: {package_dir}")
