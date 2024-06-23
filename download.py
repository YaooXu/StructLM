# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-72B-Chat")