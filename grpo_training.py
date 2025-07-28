# train_grpo.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# --- Import the reward function from your separate file ---
# Ensure 'reward_functions.py' is in the same directory as this script.
from reward_functions import combined_reward_function

# --- Data Preparation Functions ---
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    """Extracts the final answer from the original gsm8k format."""
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
    data = data.filter(lambda x: x['answer'] is not None)
    return data

def main():
    """Main function to run the GRPO training process."""
    # --- Configuration ---
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Quantization Configuration ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device
    )

    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # --- Hyperparameters ---
    learning_rate = 5e-5
    n_epochs = 3
    group_size_k = 4
    num_training_samples = 100 # Adjust as needed
    output_dir = "qwen-grpo-tuned-gsm8k"

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Load dataset ---
    dataset = get_gsm8k_questions(split="train")
    sample_dataset = dataset.select(range(num_training_samples))

    # --- Main Training Loop ---
    model.train()
    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch+1}/{n_epochs} ---")
        
        for entry in tqdm(sample_dataset, desc=f"Epoch {epoch+1} Training"):
            messages = entry['prompt']
            ground_truth_answer = entry['answer']
            
            prompt_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            # 1. GENERATE A GROUP OF K RESPONSES
            with torch.no_grad():
                generated_outputs = model.generate(
                    prompt_tokens, max_new_tokens=256, num_return_sequences=group_size_k,
                    do_sample=True, temperature=0.7, top_k=50, pad_token_id=tokenizer.pad_token_id
                )
            
            responses_text = tokenizer.batch_decode(generated_outputs[:, prompt_tokens.shape[1]:], skip_special_tokens=True)
            completions = [{"content": text} for text in responses_text]
            
            # 2. CALCULATE REWARDS (using imported function)
            rewards = torch.tensor(
                combined_reward_function(
                    prompts=[messages], completions=completions, answers=[ground_truth_answer] * group_size_k
                ), device=device
            )
            
            # 3. CALCULATE GRPO WEIGHTS (SOFTMAX OF REWARDS)
            weights = F.softmax(rewards, dim=0)

            # 4. CALCULATE LOSS
            total_loss = 0
            for i in range(group_size_k):
                response_tokens = generated_outputs[i:i+1, prompt_tokens.shape[1]:]
                full_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
                
                logits = model(full_tokens).logits
                response_logits = logits[:, -response_tokens.shape[1]:, :].contiguous()
                
                log_probs = -F.cross_entropy(
                    response_logits.view(-1, response_logits.size(-1)), 
                    response_tokens.view(-1), reduction='mean'
                )
                
                total_loss += weights[i] * log_probs
                
            final_loss = -total_loss

            # 5. UPDATE MODEL
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} finished. Final loss for last prompt: {final_loss.item():.4f}")

    print(f"\n✅ Training finished! Saving LoRA adapters to '{output_dir}'.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
