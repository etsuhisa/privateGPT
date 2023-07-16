from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

if model_type != "HuggingFace":
    exit(0)

if not os.path.exists("models/"+model_path):
    print("Download Model : "+model_path)
    snapshot_download(repo_id=model_path, revision="main", local_dir="models/"+model_path, local_dir_use_symlinks=False)
    print("Load Test : "+model_path)
    tokenizer = AutoTokenizer.from_pretrained("models/"+model_path)
    model = AutoModelForCausalLM.from_pretrained("models/"+model_path)
else:
    print("Model Already Downloaded : "+model_path)

if not os.path.exists("models/"+embeddings_model_name):
    print("Download Embeddings : "+embeddings_model_name)
    snapshot_download(repo_id=embeddings_model_name, revision="main", local_dir="models/"+embeddings_model_name, local_dir_use_symlinks=False)
    print("Load Test : "+embeddings_model_name)
    embeddings = HuggingFaceEmbeddings(model_name="models/"+embeddings_model_name)
else:
    print("Embeddings Already Downloaded : "+embeddings_model_name)
