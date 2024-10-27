# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")