from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')

if model_type != "HuggingFace":
    exit(0)

print("Download")
tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True)
tokenizer.save_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, force_download=True)
model.save_pretrained(model_path)

print("Load Test")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only =True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
