"""Data loading utilities and collators."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Callable


class DataCollator:
    """Data collator for language modeling."""

    def __init__(self,
                 pad_token_id: int = 0,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: str = "pt"):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        # Extract sequences
        input_ids = [example["input_ids"] for example in batch]
        labels = [example["labels"] for example in batch]
        attention_masks = [example.get("attention_mask") for example in batch]

        # Determine max length for this batch
        batch_max_length = max(len(seq) for seq in input_ids)  # type: ignore

        if self.max_length is not None:
            batch_max_length = min(batch_max_length, self.max_length)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            batch_max_length = (
                (batch_max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for i, (input_seq, label_seq) in enumerate(zip(input_ids, labels)):
            # Truncate if necessary
            if len(input_seq) > batch_max_length:
                input_seq = input_seq[:batch_max_length]
                label_seq = label_seq[:batch_max_length]

            # Pad sequences
            pad_length = batch_max_length - len(input_seq)

            if isinstance(input_seq, torch.Tensor):
                padded_input = torch.cat([
                    input_seq,
                    torch.full((pad_length,), self.pad_token_id, dtype=input_seq.dtype)
                ])
                padded_label = torch.cat([
                    label_seq,
                    torch.full((pad_length,), -100, dtype=label_seq.dtype)  # -100 is ignored in loss
                ])
            else:
                padded_input = input_seq + [self.pad_token_id] * pad_length
                padded_label = label_seq + [-100] * pad_length
                padded_input = torch.tensor(padded_input, dtype=torch.long)
                padded_label = torch.tensor(padded_label, dtype=torch.long)

            padded_input_ids.append(padded_input)
            padded_labels.append(padded_label)

            # Handle attention mask
            if attention_masks[i] is not None:
                attention_mask = attention_masks[i]
                if len(attention_mask) > batch_max_length:  # type: ignore
                    attention_mask = attention_mask[:batch_max_length]  # type: ignore

                if isinstance(attention_mask, torch.Tensor):
                    padded_attention = torch.cat([
                        attention_mask,
                        torch.zeros(pad_length, dtype=attention_mask.dtype)
                    ])
                else:
                    padded_attention = attention_mask + [0] * pad_length  # type: ignore
                    padded_attention = torch.tensor(padded_attention, dtype=torch.long)
            else:
                # Create attention mask (1 for real tokens, 0 for padding)
                padded_attention = torch.cat([
                    torch.ones(len(input_seq), dtype=torch.long),
                    torch.zeros(pad_length, dtype=torch.long)
                ])

            padded_attention_masks.append(padded_attention)

        # Stack into tensors
        batch_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_masks)
        }

        return batch_dict


class LanguageModelingCollator(DataCollator):
    """Specialized collator for causal language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch for causal language modeling."""
        batch_dict = super().__call__(batch)

        # For causal LM, we shift labels by one position
        # This is handled in the model's forward pass, so we keep labels as is

        return batch_dict


class PackedDataCollator:
    """Data collator for packed sequences."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate packed sequences."""
        # For packed sequences, all sequences should already be the same length
        input_ids = torch.stack([example["input_ids"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def create_dataloader(dataset: Dataset[Any],
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     drop_last: bool = True,
                     collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
                     pad_token_id: int = 0,
                     max_length: Optional[int] = None,
                     packed_sequences: bool = False) -> DataLoader:
    """Create a DataLoader for language modeling."""

    # Choose appropriate collator
    if collate_fn is None:
        if packed_sequences:
            collate_fn = PackedDataCollator(pad_token_id=pad_token_id)
        else:
            collate_fn = LanguageModelingCollator(
                pad_token_id=pad_token_id,
                max_length=max_length
            )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


class DistributedSampler:
    """Simple distributed sampler for multi-GPU training."""

    def __init__(self, dataset: Dataset[Any], num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Calculate number of samples per replica
        self.num_samples = len(dataset) // num_replicas  # type: ignore
        if len(dataset) % num_replicas != 0:  # type: ignore
            self.num_samples += 1

        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        """Generate indices for this replica."""
        if self.shuffle:
            # Generate random permutation
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # Pad to make it evenly divisible
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size

        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch


def create_distributed_dataloader(dataset: Dataset[Any],
                                 batch_size: int,
                                 num_replicas: int,
                                 rank: int,
                                 shuffle: bool = True,
                                 num_workers: int = 0,
                                 pin_memory: bool = True,
                                 drop_last: bool = True,
                                 collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
                                 pad_token_id: int = 0,
                                 max_length: Optional[int] = None,
                                 packed_sequences: bool = False) -> DataLoader:
    """Create a distributed DataLoader."""

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle
    )

    # Choose appropriate collator
    if collate_fn is None:
        if packed_sequences:
            collate_fn = PackedDataCollator(pad_token_id=pad_token_id)
        else:
            collate_fn = LanguageModelingCollator(
                pad_token_id=pad_token_id,
                max_length=max_length
            )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
