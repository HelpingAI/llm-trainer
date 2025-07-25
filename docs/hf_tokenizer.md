# Using HuggingFace Pretrained Tokenizers in LLM Trainer

Leverage HuggingFace's `AutoTokenizer` in your LLM Trainer pipeline for maximum compatibility, flexibility, and ease of use. This guide covers everything you need to use pretrained tokenizers (like Mistral, Llama, GPT-2, etc.) with your own models and training scripts.

---

## Why Use `HFTokenizerWrapper`?

- **Reuse vocabularies** from popular open-source models.
- **Fine-tune** or continue training with the same tokenization as a base model.
- **Save time** (no need to retrain a tokenizer).
- **Experiment** with different tokenization strategies.

---

## Quick Start: Load and Use a Pretrained Tokenizer

```python
from llm_trainer.tokenizer import HFTokenizerWrapper

# Load a pretrained tokenizer from HuggingFace
hf_tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
# (Optional) Set the padding token if needed
hf_tokenizer.tokenizer.pad_token = hf_tokenizer.tokenizer.eos_token
```

---

## Example 1: Training with HuggingFace SFTTrainer

This example shows how to use a HuggingFace tokenizer and model with the HuggingFace SFTTrainer.

```python
from llm_trainer.tokenizer import HFTokenizerWrapper
from transformers import MistralConfig, MistralForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# 1. Load tokenizer
hf_tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
hf_tokenizer.tokenizer.pad_token = hf_tokenizer.tokenizer.eos_token

# 2. Configure model
model_config = MistralConfig(
    vocab_size=hf_tokenizer.tokenizer.vocab_size,
    hidden_size=2048,
    intermediate_size=7168,
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096,
    pad_token_id=hf_tokenizer.tokenizer.pad_token_id,
    bos_token_id=hf_tokenizer.tokenizer.bos_token_id,
    eos_token_id=hf_tokenizer.tokenizer.eos_token_id
)
model = MistralForCausalLM(model_config)

# 3. Load and shuffle dataset
dataset = load_dataset('HuggingFaceTB/cosmopedia-20k', split="train").shuffle(seed=42)

# 4. Formatting function
def formatting_func(sample):
    return [text for text in sample['text']]

# 5. Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=2,
    max_steps=2000,
    learning_rate=1e-4,
    logging_steps=1,
    output_dir="M_outputs",
    overwrite_output_dir=True,
    save_steps=1000,
    optim="paged_adamw_32bit",
    report_to="none"
)

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=hf_tokenizer.tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_func,
    args=training_args
)

# 7. Train and save
trainer.train()
trainer.save_model("LLM")
hf_tokenizer.tokenizer.save_pretrained("LLM")
```

---

## Example 2: Training with LLM Trainer's Own Trainer

This example shows how to use `HFTokenizerWrapper` with your own Trainer and LanguageModelingDataset.

```python
from llm_trainer.tokenizer import HFTokenizerWrapper
from llm_trainer.models import TransformerLM
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.training import Trainer
from llm_trainer.data import LanguageModelingDataset

# 1. Load tokenizer
hf_tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
hf_tokenizer.tokenizer.pad_token = hf_tokenizer.tokenizer.eos_token

# 2. Model and config
model_config = ModelConfig(
    vocab_size=hf_tokenizer.tokenizer.vocab_size,
    d_model=2048,
    n_heads=32,
    n_layers=24,
    max_seq_len=4096
)
model = TransformerLM(model_config)

# 3. Training and data config
training_config = TrainingConfig(
    batch_size=2,
    learning_rate=1e-4,
    num_epochs=1,
    checkpoint_dir="./checkpoints"
)
data_config = DataConfig(
    dataset_name="HuggingFaceTB/cosmopedia-20k",
    max_length=2048
)

# 4. Prepare dataset
train_dataset = LanguageModelingDataset(
    dataset_name=data_config.dataset_name,
    tokenizer=hf_tokenizer.tokenizer,
    max_length=data_config.max_length,
    text_column="text"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    tokenizer=hf_tokenizer.tokenizer,
    config=training_config,
    train_dataset=train_dataset,
    data_config=data_config
)

# 6. Train and save
trainer.train()
trainer.save_model("./final_model")
```

---

## Example 3: Batch Encoding, Decoding, and Advanced Features

```python
# Batch encoding
texts = ["Hello world!", "How are you?", "This is a test."]
batch = hf_tokenizer.tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
print(batch["input_ids"].shape)  # (batch_size, seq_len)

# Batch decoding
decoded = hf_tokenizer.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
print(decoded)

# Offset mapping (for NER, QA, etc.)
encoding = hf_tokenizer.tokenizer("My name is Sylvain.", return_offsets_mapping=True)
print(encoding.tokens())
print(encoding["offset_mapping"])
```

---

## Example 4: Adding Custom Special Tokens

```python
# Add a custom special token
hf_tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<custom>"]})
# If using a pretrained model, resize embeddings:
model.resize_token_embeddings(len(hf_tokenizer.tokenizer))
# Get the ID of your new token
custom_id = hf_tokenizer.tokenizer.get_vocab()["<custom>"]
```

---

## Tips and Best Practices

- Always check and set `pad_token` if your model expects padding.
- For custom tasks, add special tokens before training and resize model embeddings.
- Use `batch_decode` for efficient decoding of multiple sequences.
- Use `return_tensors="pt"` for PyTorch, `"tf"` for TensorFlow, or `"np"` for NumPy.
- Use `truncation` and `padding` options for consistent input lengths.
- For advanced tasks, explore `offset_mapping`, `word_ids()`, and other fast tokenizer features.

---

## Troubleshooting

- **Tokenizer mismatch**: Ensure your model and tokenizer use the same vocabulary and special tokens.
- **Shape errors**: Use `padding` and `truncation` to ensure consistent input shapes.
- **Missing special tokens**: Set or add them as needed (e.g., `pad_token`, `bos_token`, `eos_token`).
- **Embedding size mismatch**: If you add tokens, always call `model.resize_token_embeddings(len(tokenizer))`.

---

## References and Further Reading

- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/)
- [Transformers Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Adding custom special tokens (Medium)](https://medium.com/@raquelhortab/how-to-add-custom-special-tokens-to-a-hugging-face-tokenizer-4b49a0ed9161)
- [Fast tokenizers’ special powers (HF Course)](https://huggingface.co/learn/nlp-course/chapter6/3)

---

With `HFTokenizerWrapper`, you can seamlessly integrate HuggingFace tokenization into your LLM Trainer workflow for both research and production use cases. 