# 指定下載路徑（請改成你想存放的路徑）
local_dir = "/work/home/huangaowei/SothisAI/model/ExternalSource"
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="google/t5-v1_1-xxl",  # Change to any model you want
    cache_dir=local_dir,  # Optional: download to specific path
    local_files_only=False,
    ignore_patterns=["*.git*", "blobs", "*.h5"]  # Avoid git blobs/snapshots
)
print("Model downloaded to:", model_path)


# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", cache_dir=local_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xxl", cache_dir=local_dir)

print(f"模型已下載到：{local_dir}")
