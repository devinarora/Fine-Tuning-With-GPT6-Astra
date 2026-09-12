"""Marian input/target handling and reproducible generation settings."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_model(path, device="cpu"):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, local_files_only=True, use_safetensors=True, attn_implementation="sdpa",
        dtype=torch.float32,
    ).to(device)
    model.generation_config.disable_compile = True
    return model


def collator(tokenizer, model=None):
    return DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100,
                                 pad_to_multiple_of=8, return_tensors="pt")


def generation_options(config):
    # Marian starts decoding with one start token. This permits exactly the requested
    # number of additional tokens without conflicting legacy max_length/max_new_tokens settings.
    return {"num_beams": config.evaluation.beams, "do_sample": False,
            "max_length": config.data.max_target_tokens + 1, "max_new_tokens": None,
            "use_cache": True, "disable_compile": True}


def token_features(row):
    return {key: row[key] for key in ("input_ids", "attention_mask", "labels")}
