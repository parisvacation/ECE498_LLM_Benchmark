import argparse
import sys
import os
import base64
import json
from openai import OpenAI
from typing import List, Optional, Tuple
import importlib.util
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel


# Import Response_structure dynamically based on task directory
def import_response_structure(task_dir):
    task_name = os.path.basename(task_dir)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    output_structure = importlib.import_module(f'tasks.{task_name}.output_structure')
    return output_structure.Response_structure


# Load key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# client = OpenAI(api_key=api_key)
client = instructor.from_openai(OpenAI(api_key=api_key))


def extract_prompt(prompt_path: str) -> str:
    """
    Extract the prompt from the LLM_prompt.txt file.
    """
    try:
        with open(prompt_path, "r") as f:
            prompt = f.read()
        return prompt.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find prompt file at {prompt_path}")
    except Exception as e:
        raise Exception(f"Error reading prompt file: {str(e)}")

def collect_image_paths(task_dir: str) -> List[str]:
    """
    Check for an 'images' folder and return list of image file paths if exists.
    """
    image_dir = os.path.join(task_dir, "images")
    if not os.path.isdir(image_dir):
        return []

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"))
    ]
    return image_files

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file as a base64 string for use in vision-enabled LLMs.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def query_llm(prompt: str, image_paths: Optional[List[str]] = None, model: str = "gpt-4o", task_dir: str = None):
    """
    Queries the LLM with optional image inputs.

    Parameters:
    - prompt: textual prompt
    - image_paths: list of image paths to attach (if using a vision model)
    - model: OpenAI model name (e.g., "gpt-4" or "gpt-4-vision-preview")
    - api_key: your OpenAI API key

    Returns:
    - the assistant's reply or error message
    """
    try:
        if image_paths:
            print("I am here, having images as input")
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful engineering assistant that can help with engineering design problems."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_path.split('.')[-1]};base64,{encode_image_to_base64(img_path)}"
                            }
                        }
                        for img_path in image_paths
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful engineering assistant that can help with engineering design problems."},
                {"role": "user", "content": prompt}
            ]

        # response = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     max_tokens=4096,
        #     temperature=1.2
        # )
        Response_structure = import_response_structure(task_dir)
        response = client.chat.completions.create(
                    model=model,
                    response_model=Response_structure,
                    messages=[{"role": "user", "content": prompt}],
                )


        # content = response.choices[0].message.content
        return response

    except Exception as e:
        return f"OpenAI API call failed: {e}"


def load_evaluator(evaluate_path):
    spec = importlib.util.spec_from_file_location("evaluator", evaluate_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator.evaluate_llm_response


def run_task_k_times(task_dir, k=1, model="gpt-4o", log_dir="logs"):
    task_name = os.path.basename(task_dir)
    prompt_path = os.path.join(task_dir, "LLM_prompt.txt")
    evaluate_py = os.path.join(task_dir, "evaluate.py")

    if not os.path.exists(prompt_path) or not os.path.exists(evaluate_py):
        print(f"Skipping incomplete task: {task_dir}")
        return {"task": task_name, "pass_count": 0, "total": k}

    prompt = extract_prompt(prompt_path)
    print(prompt)
    evaluate_fn = load_evaluator(evaluate_py)
    pass_count = 0
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name}_log.jsonl")

    with open(log_file, "w") as logf:
        for trial in range(k):
            # Check for images in task directory
            image_paths = collect_image_paths(task_dir)
            response = query_llm(prompt, image_paths=image_paths, model=model, task_dir=task_dir)
            print(response)
            if response:
                passed, detailed_result, score, confidence = evaluate_fn(response)
            else:
                passed, detailed_result, score, confidence = False, {}, 0, 0

            log_entry = {
                "trial": trial,
                "response": response,
                "passed": passed,
                "evaluation_result": detailed_result,
                "score": score,
                "confidence": confidence
            }
            logf.write(str(log_entry) + "\n")
            
            if passed:
                pass_count += 1

    return {"task": task_name, "pass_count": pass_count, "total": k}

# Add the parent directory to the Python path so we can import from evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Run task evaluation')
    parser.add_argument('--task_dir', type=str, required=True, help='Path to the task directory')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use (default: gpt-4o)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to store logs (default: logs)')
    parser.add_argument('--k', type=int, default=1, help='Number of trials to run (default: 1 as this is for test purpose)')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path
    task_dir = os.path.abspath(args.task_dir)
    
    # Ensure the task directory exists
    if not os.path.exists(task_dir):
        print(f"Error: Task directory '{task_dir}' does not exist")
        sys.exit(1)
    
    # Run the evaluation
    result = run_task_k_times(task_dir, k=args.k, model=args.model, log_dir=args.log_dir)
    print(result)
if __name__ == "__main__":
    main() 