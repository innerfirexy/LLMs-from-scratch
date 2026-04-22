# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Standalone script to generate responses for all entries in a JSON file
# using a saved finetuned GPT-2 model.

import argparse
import json
import os

import tiktoken
import torch
from tqdm import tqdm

from previous_chapters import (
    generate,
    GPTModel,
    text_to_token_ids,
    token_ids_to_text
)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def load_model(model_path, model_size, device):
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_map = {
        "124M": "gpt2-small (124M)",
        "355M": "gpt2-medium (355M)",
        "774M": "gpt2-large (774M)",
        "1558M": "gpt2-xl (1558M)",
    }

    if model_size in model_map:
        choose_model = model_map[model_size]
    elif model_size in model_configs:
        choose_model = model_size
    else:
        raise ValueError(
            f"Invalid --model_size: {model_size}. Choose from: 124M, 355M, 774M, 1558M "
            f"or full names like 'gpt2-small (124M)'"
        )

    BASE_CONFIG.update(model_configs[choose_model])

    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, BASE_CONFIG


def generate_responses(data, model, tokenizer, device, context_size):
    for i, entry in tqdm(enumerate(data), total=len(data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=context_size,
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        data[i]["model_response"] = response_text

    return data


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = tiktoken.get_encoding("gpt2")
    model, base_config = load_model(args.model_path, args.model_size, device)
    print("Loaded model from:", args.model_path)

    with open(args.input, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Generating responses for {len(data)} entries ...")
    data = generate_responses(data, model, tokenizer, device, base_config["context_length"])

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    print(f"Saved generated responses to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate model responses for entries in a JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file containing instruction entries."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model .pth file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="124M",
        help="Model size corresponding to the saved weights. Options: 124M, 355M, 774M, 1558M."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON file where responses will be saved."
    )
    args = parser.parse_args()
    main(args)
