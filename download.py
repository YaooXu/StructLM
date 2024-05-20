# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/StructLM-7B-Mistral")
model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/StructLM-7B-Mistral")