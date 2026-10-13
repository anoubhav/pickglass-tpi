import os, sys, time, json
import torch, numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_ft import PR, model_path, points_answer, parse_points
from views import view_image, guide_line_view, resize_square
from transformers import AutoModelForImageTextToText, AutoProcessor
p = model_path()
proc = AutoProcessor.from_pretrained(p, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(p, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
print(type(model).__name__, sum(x.numel() for x in model.parameters())/1e9, "B params")
names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
print("linear leaf names:", names)
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and (".0." in n or "projector" in n or "pooling" in n): print("  ", n, tuple(m.weight.shape))
print(model.config.text_config.num_hidden_layers if hasattr(model.config,'text_config') else '')
FW = os.path.expanduser(("~/" + os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass")))
crop = cv2.imread(os.path.join(FW, "data", "legacy_crops", "a1.png"))
img = guide_line_view(resize_square(crop, 644), "h")
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
msgs = [{"role": "user", "content": [dict(type="text", text=PR.get("P4h")["text"]), dict(type="image", image=pil)]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
print({k: (tuple(v.shape), v.dtype) for k, v in inp.items()})
inp = {k: v.to(0) for k, v in inp.items()}
t = time.time()
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=1536, do_sample=False)
txt = proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
print("gen", time.time()-t, "s", out.shape[1]-inp["input_ids"].shape[1], "tokens; n_points", parse_points(txt)["n_points"], txt[:200])
# answer tokenisation cost
ans = points_answer(list(np.linspace(0.01, 0.99, 100)), "h")
ids = proc.tokenizer(ans + "<|im_end|>", add_special_tokens=False)["input_ids"]
print("100-point answer tokens:", len(ids), ans[:80])
print(proc.tokenizer.convert_ids_to_tokens(ids[:30]))
# batched generation check (4 identical-length prompts)
ims = []
for nm in ["a1", "a2", "r2", "r8"]:
    c = cv2.imread(os.path.join(FW, "data", "legacy_crops", f"{nm}.png"))
    ims.append(Image.fromarray(cv2.cvtColor(guide_line_view(resize_square(c, 644), "h"), cv2.COLOR_BGR2RGB)))
convs = [[{"role": "user", "content": [dict(type="text", text=PR.get("P4h")["text"]), dict(type="image", image=im)]}] for im in ims]
try:
    b = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True)
    print("batched keys", {k: tuple(v.shape) for k, v in b.items()})
    b = {k: v.to(0) for k, v in b.items()}
    t = time.time()
    with torch.inference_mode():
        ob = model.generate(**b, max_new_tokens=1536, do_sample=False)
    print("batched gen", time.time()-t, "s")
    for i in range(4):
        tx = proc.tokenizer.decode(ob[i, b["input_ids"].shape[1]:], skip_special_tokens=True)
        print(i, parse_points(tx)["n_points"], tx[:80])
except Exception as e:
    import traceback; traceback.print_exc()
