import shutil
import os

cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(hf_cache_dir)
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)
