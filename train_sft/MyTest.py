import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



def func(y, x, z=9):
    return x + y + z

if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:1972"
    os.environ["https_proxy"] = "http://127.0.0.1:1972"
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:1972'
    print("http_proxy:", os.environ.get("http_proxy"))
    print("https_proxy:", os.environ.get("https_proxy"))

    #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B", cache_dir="my_cache")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                              cache_dir="deepseek-r1-distall-qwen-1.5b-cache")
    tokenizer
    test_str = "this is an apple"
    tokens = tokenizer(test_str, return_tensors="pt")
    print(tokens)
    print(type(tokens['input_ids']))


    input = {"x": 1, "y": 2}
    print(func(**input))

    input2 = input.pop("x")
    print(f"input: {input}, input2: {input2}")

    t = torch.tensor(10)
    print(t.item())