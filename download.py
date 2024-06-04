# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModelForCausalLM.from_pretrained("FacebookAI/roberta-base")