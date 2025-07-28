import re
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# --- Data Preparation and Parsing Functions ---

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extracts the content from between the <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for malformed XML
    if "<answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return "" # Return empty string if no answer tag is found

def extract_hash_answer(text: str) -> str | None:
    """Extracts the final answer from the original gsm8k format (e.g., ... #### 123)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    """Loads and formats the gsm8k dataset."""
    data = load_dataset('openai/gsm8k', 'main', split=split)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    # Filter out examples where the answer could not be extracted
    data = data.filter(lambda x: x['answer'] is not None)
    return data

# --- Reward Functions ---

def correctness_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Rewards the model if its extracted answer matches the ground truth. Main reward signal."""
    responses = [completion["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Debug print to see what's being compared
    q = prompts[0][-1]['content']
    print('-'*20)
    print(f"Question: {q}")
    print(f"Ground Truth Answer: {answers[0]}")
    print(f"Model Response: {responses[0]}")
    print(f"Extracted Model Answer: {extracted_responses[0]}")
    print(f"Correct: {extracted_responses[0] == answers[0]}")
    print('-'*20)
    
    # A large reward for getting the answer correct
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answers)]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model for including the basic XML tags, even if spacing is wrong."""
    pattern = r"<reasoning>.*?</reasoning>.*<answer>.*?</answer>"
    responses = [completion["content"] for completion in completions]
    # Use re.search to find the pattern anywhere in the string, re.DOTALL to match newlines
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    # A small reward for correct formatting
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Gives a fractional reward for each correctly placed XML tag."""
    def count_xml(text):
        count = 0.0
        # Check for start tags
        if text.count("<reasoning>") == 1: count += 0.1
        if text.count("<answer>") == 1: count += 0.1
        # Check for end tags
        if text.count("</reasoning>") == 1: count += 0.1
        if text.count("</answer>") == 1: count += 0.1
        # Penalize for text after the final closing tag
        if "</answer>" in text:
            trailing_text = text.split("</answer>")[-1]
            count -= len(trailing_text.strip()) * 0.01
        return count
        
    contents = [completion["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def combined_reward_function(prompts, completions, answers, **kwargs) -> list[float]:
    """Combines all reward functions into a single score for each completion."""
    correctness_rewards = correctness_reward_func(prompts, completions, answers)
    format_rewards = soft_format_reward_func(completions)
    xml_count_rewards = xmlcount_reward_func(completions)
    
    # Sum the rewards from all functions
    total_rewards = [
        c_r + f_r + x_r
        for c_r, f_r, x_r in zip(correctness_rewards, format_rewards, xml_count_rewards)
    ]
    return total_rewards