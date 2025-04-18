from openai import OpenAI
import os
from dotenv import load_dotenv
import json


# Load key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class LLMJudge:
    def __init__(self, task_dir, llm_response, model="o3-mini"):
        """
        Initialize the LLM judge
        
        Args:
            task_dir (str): Path to the task directory containing prompt and solution files
            llm_response (str): The response to evaluate
            model (str): Name of the OpenAI model to use
        """
        self.task_dir = task_dir
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
        # Load task files
        self.question = self._load_file("LLM_prompt.txt")
        self.correct_answer = self._load_file("solution.txt")
        self.rubric = self._load_file("rubrics.txt")
        self.llm_response = llm_response
        
    def _load_file(self, filename):
        """Load content from a file in the task directory"""
        file_path = os.path.join(self.task_dir, filename)
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Could not load {filename}: {str(e)}")
            
    def _format_judge_prompt(self):
        """Format the prompt for the judge"""
        return f"""Judge whether the following [response] to [question] based on the precise and unambiguous [correct_answer] below using the [rubric].

[question]: {self.question}

[response]: {self.llm_response}

[rubric]: {self.rubric}

[correct_answer]: {self.correct_answer}

Your judgement must be in the following json format:

{{
    "reasoning": "Explain why the [response] is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.",
    "score": "The score between 0 and 100 based on the [rubric] and the [response]. Stricly follow the [rubric] and give partial credit if the [response] is partially correct.",
    "passed": "Answer 'True' if the llm response achieves score of 100 based on the [rubric]. Answer 'False' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the answer is incorrect.",
    "confidence": "The confidence score of your judgement between 0 and 100."
}}
"""

    def _parse_judgment(self, judgment):
        """Parse the judgment response from the LLM"""
        judgment = json.loads(judgment)
        passed = judgment['passed']
        details = judgment['reasoning']
        score = judgment['score']
        confidence = judgment['confidence']
                    
        return passed, details, score, confidence

    def evaluate(self):
        """
        Evaluate an LLM response using the judge
        
        Returns:
            tuple: (passed, details, score)
        """
        try:
            # Get judgment from model
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a precise and strict judge evaluating mathematical solutions."},
                    {"role": "user", "content": self._format_judge_prompt()}
                ]
            )
            
            # Parse and return results
            judgment = response.choices[0].message.content
            print(judgment)
            return self._parse_judgment(judgment)
            
        except Exception as e:
            return False, {"error": f"Evaluation failed: {str(e)}"}, 0