import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B", cache_dir="my_cache")

def func(y, x, z=9):
    return x + y + z

if __name__ == "__main__":
    input = {"x": 1, "y": 2}
    print(func(**input))

    input2 = input.pop("x")
    print(f"input: {input}, input2: {input2}")

    t = torch.tensor(10)
    print(t.item())