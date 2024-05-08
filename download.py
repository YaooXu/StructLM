# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")