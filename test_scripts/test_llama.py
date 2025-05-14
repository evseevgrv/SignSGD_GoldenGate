import gc
import torch
from transformers import LlamaConfig, AutoModel, AutoModelForCausalLM
import json
import inspect
import time

device = "cuda:1"

args = inspect.signature(LlamaConfig.__init__).parameters
with open('configs/llama_60m.json', 'r') as file:
    config_dict = json.load(file)
args_list = list(arg for arg in args)
for arg in args:
    if arg not in config_dict.keys():
        print(arg)
args_list = list(arg for arg in args)
new_config_dict = {}
for k in config_dict:
    if k not in args_list:
        print(k, config_dict[k])
    else:
        new_config_dict[k] = config_dict[k]

from utils.modeling_llama import LlamaForCausalLM as CustomLlama
from transformers import AutoConfig

def test(model, data, num_steps=50):
    print(model)
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     raise
    start = time.time()
    for _ in range(10):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(data, labels=data).loss
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
        print(time.time() - start)
        start = time.time()

    start = time.time()

    for _ in range(num_steps):
        if not _ % 5:
            print(_)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(data, labels=data).loss
        loss.backward()
        model.zero_grad()

    print((time.time()-start) / num_steps)

def test_train(model):
    torch.manual_seed(0)
    data = torch.randint(0, 32000, (256, 257), dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(50):
        if not _ % 5:
            print(_)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(data, labels=data).loss
        loss.backward()
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()
    return losses

# torch.use_deterministic_algorithms(True)

torch.manual_seed(0)
data = torch.randint(0, 32000, (256, 256), dtype=torch.long, device=device)

model_config = AutoConfig.from_pretrained("configs/llama_60m.json")
torch.manual_seed(0)
custom_model = CustomLlama(model_config).to(device)
test(custom_model, data)
del custom_model
torch.cuda.empty_cache()
gc.collect()

config = LlamaConfig(**new_config_dict, num_key_value_heads=new_config_dict["num_attention_heads"])
torch.manual_seed(0)
model_sdpa = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").to(device)
test(model_sdpa, data)
del model_sdpa
torch.cuda.empty_cache()
gc.collect()

torch.manual_seed(0)
model_flash = AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2").to(device)
test(model_flash, data)
del model_flash
torch.cuda.empty_cache()
gc.collect()

torch.manual_seed(0)
model_flash = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(device)
test(model_flash, data)
del model_flash
torch.cuda.empty_cache()
gc.collect()

for (name1, p1), (name2, p2) in zip(custom_model.named_parameters(), model_sdpa.named_parameters()):
    assert name1 == name2
    assert p1.data.eq(p2.data).all()

# assert custom_model(data).logits.eq(model_sdpa(data).logits).all()

# losses_custom = test_train(custom_model)
# losses_sdpa = test_train(model_sdpa)

# print("custom")
# print(losses_custom)
# print()
# print("sdpa")
# print(losses_sdpa)
# print()
# assert torch.tensor(losses_custom).eq(torch.tensor(losses_sdpa)).all()
