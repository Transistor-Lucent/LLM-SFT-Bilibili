from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # 每种模型都有自己官方的chat template
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

# chat template
dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"},
          {"role": "assistant", "content": "这是由于光的散射引起的。"}]

input = tokenizer.apply_chat_template(dialog, return_tensors="pt")  # return_tensors="pt" 表示返回 PyTorch tensor 格式（否则默认返回字符串或列表）
input = {k: v.to("cuda") for k, v in input.items()}
# 此处得到的input是一个字典，默认只包含两个key。
# input = {"input_ids": 模型可以使用的 token 序列（整数张量）, "attention_mask": 注意力 mask，表示哪些 token 是 padding，哪些是实际输入}


#设置labels和inputs一致
input["labels"] = input["input_ids"].clone()  # ids指token id sequence

output = model(**input)  # input字典解包传参，为model参数中变量名与input的key: str相同的变量传递对应的value

#获取模型的loss
loss = output.loss
loss.backward()  # 算梯度
optimizer.step()  # 更新参数
optimizer.zero_grad()  # 梯度归零。torch中梯度默认是累加的。

#保存模型
model.save_pretrained("output_dir")