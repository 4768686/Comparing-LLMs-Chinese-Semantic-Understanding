from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/mnt/workspace/models/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float16
).cuda().eval()

prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=300)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

