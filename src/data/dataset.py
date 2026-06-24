"""
PyTorch Dataset 类
用于 SFT 和 DPO 训练的 DataLoader

对应 train_sft.py 和 train_dpo.py 中的 SFTDataset / DPODataset
"""

import torch
from torch.utils.data import Dataset
import json


def _tokenize_pair(tokenizer, prompt: str, response: str,
                   control_token_ids: list, max_length: int,
                   num_control_tokens: int) -> dict:
    """共享 tokenization: prompt + control tokens + response + <eos>, 带 padding/masking.

    SFTDataset 和 DPODataset 共用此逻辑 (之前 ~30 行完全重复)。
    """
    # Response budget: 1024 tokens (JSON with 176 floats needs ~890 tokens)
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length - 1024)
    # add_special_tokens=False prevents duplicate <bos> in the middle of
    # the sequence; we manually append <eos> so the model learns to stop
    # after the JSON closes instead of generating garbage at inference.
    resp_enc = tokenizer(response, truncation=True, max_length=1024,
                         add_special_tokens=False)
    resp_ids = resp_enc["input_ids"] + [tokenizer.eos_token_id]

    input_ids = prompt_enc["input_ids"] + control_token_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    prompt_len = len(prompt_enc["input_ids"])
    control_len = num_control_tokens

    # labels use resp_ids (with <eos>) so the model learns to emit <eos>
    labels = [-100] * (prompt_len + control_len) + resp_ids
    label_mask = [0] * (prompt_len + control_len) + [1] * len(resp_ids)
    control_mask = [0] * prompt_len + [1] * num_control_tokens + [0] * len(resp_ids)

    # Padding
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
        label_mask += [0] * pad_len
        control_mask += [0] * pad_len
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        label_mask = label_mask[:max_length]
        control_mask = control_mask[:max_length]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_mask": torch.tensor(label_mask, dtype=torch.float32),
        "control_mask": torch.tensor(control_mask, dtype=torch.bool),
    }


class SFTDataset(Dataset):
    """SFT 数据集 (供 train_sft.py 使用)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096,
                 num_control_tokens: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_control_tokens = num_control_tokens
        self.control_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        )
        if any(tid is None or tid == tokenizer.unk_token_id for tid in self.control_token_ids):
            raise ValueError("Control tokens must be added to the tokenizer before building SFTDataset.")

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        result = _tokenize_pair(
            self.tokenizer, item["prompt"], item["response"],
            self.control_token_ids, self.max_length, self.num_control_tokens,
        )
        result["q_current"] = torch.tensor(item.get("q_current", []), dtype=torch.float32)
        result["delta_q_target"] = torch.tensor(item.get("delta_q", []), dtype=torch.float32)
        result["delta_a_target"] = torch.tensor(item.get("delta_a", []), dtype=torch.float32)
        result["delta_p_target"] = torch.tensor(item.get("delta_p", []), dtype=torch.float32)
        return result


class DPODataset(Dataset):
    """DPO 数据集 (供 train_dpo.py 使用)"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096,
                 num_control_tokens: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_control_tokens = num_control_tokens
        self.control_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        )
        if any(tid is None or tid == tokenizer.unk_token_id for tid in self.control_token_ids):
            raise ValueError("Control tokens must be added to the tokenizer before building DPODataset.")

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def _encode_pair(self, prompt: str, response: str):
        """单个 prompt-response 对 tokenization (委托给共享 _tokenize_pair)"""
        return _tokenize_pair(
            self.tokenizer, prompt, response,
            self.control_token_ids, self.max_length, self.num_control_tokens,
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = self._encode_pair(prompt, item["chosen"])
        rejected = self._encode_pair(prompt, item["rejected"])

        result = {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "labels_chosen": chosen["labels"],
            "label_mask_chosen": chosen["label_mask"],
            "control_mask_chosen": chosen["control_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
            "labels_rejected": rejected["labels"],
            "label_mask_rejected": rejected["label_mask"],
            "control_mask_rejected": rejected["control_mask"],
        }

        # Oracle targets for control loss (from winner/best solution)
        if "delta_q" in item:
            result["q_current"] = torch.tensor(item.get("q_current", []), dtype=torch.float32)
            result["delta_q_target"] = torch.tensor(item["delta_q"], dtype=torch.float32)
            result["delta_a_target"] = torch.tensor(item["delta_a"], dtype=torch.float32)
            result["delta_p_target"] = torch.tensor(item["delta_p"], dtype=torch.float32)

        return result
