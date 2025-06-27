from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/mnt/workspace/models/deepseek-llm-7b-chat"
prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型（使用 float16，GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda().eval()

# 编码 prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成响应
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    top_p=0.95,
    temperature=0.7
)

# 解码并输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
