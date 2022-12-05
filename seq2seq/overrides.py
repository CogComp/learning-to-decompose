from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.data_collator import DataCollator


@dataclass
class DoNothingDataCollator(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        lm_labels = torch.tensor([f['lm_labels'] for f in features], dtype=torch.long)
        decoder_attention_mask = torch.tensor([f['decoder_attention_mask'] for f in features], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

@dataclass
class DoNothingDataCollatorForGeneration(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        lm_labels = torch.tensor([f['lm_labels'] for f in features], dtype=torch.long)
        decoder_attention_mask = []
        decoder_inputs = []
        for i in range(0, len(features)):
            decoder_inputs.append([0] * (input_ids.size()[1]))
            decoder_attention_mask.append([1, 1, 1, 1, 1] + [0] * (input_ids.size()[1] - 5))
        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs,
            "lm_labels": lm_labels,
        }
