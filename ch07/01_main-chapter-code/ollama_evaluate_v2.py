import argparse
import json
import os
import re
import ssl
import urllib.error
import urllib.request

import psutil
from tqdm import tqdm


def get_ssl_context(cert_path=None):
    """Create an SSL context that trusts the self-signed Caddy certificate.

    If caddy-cert.pem exists in the same directory as this script, it is
    loaded so that HTTPS connections to the Caddy reverse proxy succeed.
    Otherwise a standard system-CA context is returned.
    """
    if cert_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cert_path = os.path.join(script_dir, "caddy-cert.pem")

    context = ssl.create_default_context()
    if os.path.exists(cert_path):
        context.load_verify_locations(cert_path)
    return context


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(prompt, model="gpt-oss:20b", url="http://127.0.0.1:11434/api/chat"):
    # Create the data payload as a dictionary:
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Settings required for deterministic responses:
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }
    # Convert the dictionary to JSON and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a POST request and add headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    response_data = ""

    # Send the request and capture the streaming response
    with urllib.request.urlopen(request, context=get_ssl_context()) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            # Parse each line into JSON
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


# ollama_model = "gpt-oss:20b"
# result = query_model("What is 1+2?", ollama_model)
# print(result)


def rubric_prompt(instruction, reference_answer, model_answer):
    rubric = """
        You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
        You will be given an instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing the evaluation criteria.
        Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        Please do not generate any other opening, closing, and explanations.

        Here is the rubric you should use to build your answer:
        1: The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.
        2: The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.
        3: The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.
        4: The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.
        5: The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration

        Provide your feedback as follows:

        Feedback:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
        """

    prompt = (
        f"{rubric}\n\n"
        "Now here is the instruction, the reference answer, and the response.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer:\n{model_answer}\n\n"
        "\n\nProvide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.\n"
        "Feedback:::\n"
        "Evaluation: "
    )
    return prompt


def test_rubric_prompt():
    rendered_prompt = rubric_prompt(
        instruction=(
            "If all birds can fly, and a penguin is a bird, can a penguin fly?"
        ),
        reference_answer=(
            "Yes, according to the premise that all birds can fly, a penguin can fly."
        ),
        model_answer=("Yes – under those premises a penguin would be able to fly."),
    )
    result = query_model(rendered_prompt, "gpt-oss:20b")
    print(result)


# test_rubric_prompt()


def generate_model_scores(json_data, json_key, model, url):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = rubric_prompt(
            instruction=entry["instruction"],
            reference_answer=entry["output"],
            model_answer=entry[json_key],
        )
        response = query_model(prompt, model, url=url)
        # extract the score from response string
        try:
            # find the "Total rating: " pattern
            total_rating_match = re.search(r"Total rating: (\d+)", response)
            if total_rating_match:
                scores.append(int(total_rating_match.group(1)))
            else:
                raise ValueError()
        except ValueError:
            print(f"Could not extract score from response: {response}")
            continue

    return scores


def check_ollama_reachable(base_url, timeout=5):
    """Check if an Ollama server is reachable at the given base URL."""
    # Try the /api/tags endpoint as a lightweight health check
    tags_url = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=get_ssl_context()
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def resolve_url(user_url, port=11434):
    """Build the full API chat URL from a user-provided IP/hostname."""
    user_url = user_url.strip()
    # If user already provided a full URL, ensure it ends with /api/chat
    if user_url.startswith("http://") or user_url.startswith("https://"):
        if "/api/chat" not in user_url:
            user_url = user_url.rstrip("/") + "/api/chat"
        return user_url
    # Otherwise treat it as an IP address or hostname
    return f"http://{user_url}:{port}/api/chat"


def is_localhost(url):
    """Return True if the URL points to the local machine."""
    # Normalize to lowercase for comparison
    lowered = url.lower()
    return "127.0.0.1" in lowered or "localhost" in lowered


def test_instruction_data_with_response():
    test_data = json.load(open("test-data-with-response_355M-mask0.json"))
    scores = generate_model_scores(
        test_data,
        json_key="model_response",
        model="gpt-oss:20b",
        url="http://127.0.0.1:11434/api/chat",
    )
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")


# test_instruction_data_with_response()


def main(args):
    api_url = resolve_url(args.url, args.port)

    if is_localhost(api_url):
        ollama_running = check_if_running("ollama")
        if not ollama_running:
            raise RuntimeError(
                "Ollama not running locally. Launch ollama before proceeding."
            )
        print("Ollama running locally:", check_if_running("ollama"))
    else:
        base_url = api_url.replace("/api/chat", "")
        if not check_ollama_reachable(base_url):
            raise RuntimeError(f"Ollama server is not reachable at {base_url}")
        print(f"Ollama server reachable at: {base_url}")

    test_data = json.load(open(args.file_path))
    model = args.model
    scores = generate_model_scores(
        test_data, json_key="model_response", model=model, url=api_url
    )
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss:20b",
        help="The judge model to use for evaluation",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="127.0.0.1",
        help="IP address or hostname of the Ollama server (default: 127.0.0.1). Examples: 127.0.0.1, 127.0.0.1:11434, http://127.0.0.1, http://127.0.0.1:11434, https://127.0.0.1, https://127.0.0.1:11434. If you provide a full URL, the --port will be ignored.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number of the Ollama server (default: None)",
    )
    args = parser.parse_args()
    main(args)
