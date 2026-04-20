# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A vLLM-based evaluation script corresponding to ollama_evaluate.py

import json
import re

from tqdm import tqdm
from vllm import LLM, SamplingParams


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def extract_score(text):
    """Extract the first integer from the model output."""
    # Look for the first sequence of digits
    match = re.search(r"\d+", text.strip())
    if match:
        return int(match.group())
    return None


def generate_model_scores(json_data, json_key, model_name, tensor_parallel_size=1):
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        seed=123,
        max_tokens=10,  # Scores are short
    )

    # Prepare all prompts
    prompts = []
    indices = []  # Keep track of which entries actually need scoring
    for idx, entry in enumerate(json_data):
        if entry[json_key] == "":
            continue
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        prompts.append(prompt)
        indices.append(idx)

    # Run batched inference
    outputs = llm.generate(prompts, sampling_params)

    # Initialize scores: 0 for empty responses
    scores = [0] * len(json_data)
    for idx, output in zip(indices, outputs):
        generated_text = output.outputs[0].text
        score = extract_score(generated_text)
        if score is not None:
            scores[idx] = score
        else:
            print(f"Could not convert score: {generated_text}")

    return scores


def main(file_path, model, tensor_parallel_size):
    with open(file_path, "r") as file:
        test_data = json.load(file)

    scores = generate_model_scores(
        test_data, "model_response", model, tensor_parallel_size=tensor_parallel_size
    )

    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    if scores:
        print(f"Average score: {sum(scores)/len(scores):.2f}\n")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate model responses with vLLM"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model name or path to use for evaluation via vLLM"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    args = parser.parse_args()

    main(
        file_path=args.file_path,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
