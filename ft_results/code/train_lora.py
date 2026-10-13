"""LoRA fine-tuning of Molmo2-4B to point at yarn crossings, from SYNTHETIC data only.

Data: ft/data/train (synth_pts.py, generator v2 + exact crossing labels).  Each training example is
drawn deterministically from index i (rng = default_rng([seed, i])), so a run is resumable from the
optimizer step alone:
    20 %  P1  (whole 1-inch view, no guide)  -> {"vertical_yarns": n_h, "horizontal_yarns": n_v, "weave": c}
    30 %  P4h / P4v on the whole 1-inch view (red guide line)       -> Molmo2 <points coords=...>
    50 %  P4h / P4v on a zoomed view: central quarter, one of the 4 diagonal quarters, or one of 6
          random square windows (15-60 % of the side)
Prompts are the project's verbatim P1 / P4h / P4v texts; views are built with the project's
data_vlm.resize_square / guide_line_view (644 px).  n_h / n_v = exact number of yarn centre lines
crossing the guide lines inside the window.

Loss: token cross-entropy on the answer tokens only (+ <|im_end|>), mean per example; logits are
computed only for the answer span (logits_to_keep).  LoRA (peft) on every LLM linear layer, the ViT
blocks that feed the connector (0-24) and the connector (pooling attention + projector).

    torchrun --nproc_per_node 3 train_lora.py --out ../ckpt/run1 --max-steps 900
Resumable: re-running with the same --out continues from the newest checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_ft import PR, PROMPT_ID, count_answer, model_path, points_answer  # noqa: E402
from views import view_image  # noqa: E402

LORA_TARGETS = (r".*(transformer\.blocks\.\d+\.(self_attn\.(att_proj|attn_out)|mlp\.(ff_proj|ff_out))"
                r"|image_vit\.transformer\.resblocks\.(\d|1\d|2[0-4])\.(attention\.(wq|wk|wv|wo)|feed_forward\.(w1|w2))"
                r"|image_pooling_2d\.(wq|wk|wv|wo)|image_projector\.(w1|w2|w3))$")
QVIEWS = ["qc", "qd0", "qd1", "qd2", "qd3"] + [f"r{k}" for k in range(6)]
P_TASK = (0.20, 0.30, 0.50)   # P1 full, P4 full, P4 zoomed


def load_labels(d):
    labs = [json.loads(l) for l in open(os.path.join(d, "labels.jsonl"))]
    labs.sort(key=lambda r: r["seed"])
    return labs


class Examples(torch.utils.data.Dataset):
    """Item i -> one deterministic (crop, task, view, axis) example, tokenised for Molmo2."""

    def __init__(self, data_dir, mpath, seed, n_items, offset=0, task_probs=P_TASK):
        self.dir, self.mpath, self.seed, self.n, self.offset, self.tp = data_dir, mpath, seed, n_items, offset, task_probs
        self.labs = load_labels(data_dir)
        self.proc = None

    def __len__(self):
        return self.n

    def spec(self, i):
        rng = np.random.default_rng([self.seed, i])
        k = int(rng.integers(len(self.labs)))
        u = rng.random()
        axis = "h" if rng.random() < 0.5 else "v"
        if u < self.tp[0]:
            return k, "P1", "full", None
        if u < self.tp[0] + self.tp[1]:
            return k, "P4", "full", axis
        return k, "P4", QVIEWS[int(rng.integers(len(QVIEWS)))], axis

    def build(self, k, task, vname, axis):
        from transformers import AutoProcessor
        if self.proc is None:
            self.proc = AutoProcessor.from_pretrained(self.mpath, trust_remote_code=True)
        lab = self.labs[k]
        crop = cv2.imread(os.path.join(self.dir, "img", lab["file"]))
        V = lab["views"][vname]
        if task == "P1":
            img = view_image(crop, V["rect"], None)
            prompt = PR.get("P1")["text"]
            ans = count_answer(len(V["h"]), len(V["v"]), lab["weave_coarse"])
        else:
            img = view_image(crop, V["rect"], axis)
            prompt = PR.get(PROMPT_ID[axis])["text"]
            ans = points_answer(V[axis], axis)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [dict(type="text", text=prompt), dict(type="image", image=pil)]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
        tok = self.proc.tokenizer
        ans_ids = torch.tensor(tok(ans, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")])
        n = len(ans_ids)
        out = dict(inp)
        out["input_ids"] = torch.cat([inp["input_ids"][0], ans_ids])[None]
        out["attention_mask"] = torch.ones_like(out["input_ids"])
        out["token_type_ids"] = torch.cat([inp["token_type_ids"][0], torch.zeros(n, dtype=inp["token_type_ids"].dtype)])[None]
        out["answer_ids"] = ans_ids[None]
        out["meta"] = dict(task=task, view=vname, axis=axis, seed=lab["seed"], n_ans=n)
        return out

    def __getitem__(self, i):
        return self.build(*self.spec(i + self.offset))


def collate(b):
    assert len(b) == 1
    return b[0]


INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids", "pixel_values", "image_token_pooling",
              "image_grids", "image_num_crops")


def _ce_chunk(h, t, W):
    logits = F.linear(h, W).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), reduction="sum")


class LossModule(torch.nn.Module):
    """Answer-token CE computed from the final hidden states in 512-token chunks (each chunk's
    151936-way logits are recomputed in backward instead of stored: ~2.5 GB saved on 200-point
    answers).  Wrapped by DDP, so gradient sync hooks see this forward."""

    def __init__(self, peft_model):
        super().__init__()
        self.m = peft_model

    def forward(self, inp, tgt):
        from torch.utils.checkpoint import checkpoint
        base = self.m.get_base_model()
        hs = base.model(**inp, use_cache=False).last_hidden_state
        n = tgt.shape[1]
        h = hs[:, -(n + 1):-1]
        tot = 0.0
        for s in range(0, n, 512):
            tot = tot + checkpoint(_ce_chunk, h[:, s:s + 512], tgt[:, s:s + 512], base.lm_head.weight,
                                   use_reentrant=False)
        return tot / n


def checkpoint_vit_blocks(model):
    """Non-reentrant activation checkpointing for the ViT blocks (HF only checkpoints LLM layers)."""
    from torch.utils.checkpoint import checkpoint
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "Molmo2VisionBlock":
            fwd = mod.forward

            def wrapped(*args, _f=fwd, **kw):
                if torch.is_grad_enabled():
                    return checkpoint(_f, *args, use_reentrant=False, **kw)
                return _f(*args, **kw)
            mod.forward = wrapped
            n += 1
    return n


def forward_loss(lossmod, ex, dev):
    inp = {k: ex[k].to(dev, non_blocking=True) for k in INPUT_KEYS}
    return lossmod(inp, ex["answer_ids"].to(dev))


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.max_steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "train"))
    ap.add_argument("--val", default=os.path.join(HERE, "..", "data", "val"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--keep-every", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--max-hours", type=float, default=2.2, help="wall-clock stop (checkpoint saved)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--probe", type=int, default=0, help="run N micro-steps on long examples, report, exit")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f"train_rank{rank}.log"), "a") if rank == 0 else None

    def P(*s):
        if rank == 0:
            msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in s)
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText
    mpath = model_path()
    model = AutoModelForImageTextToText.from_pretrained(mpath, trust_remote_code=True, dtype=torch.bfloat16,
                                                        device_map={"": local})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=LORA_TARGETS,
                     bias="none")
    model = get_peft_model(model, cfg)
    # peft/HF register a hook making the embedding output require grad (needed only for REENTRANT
    # checkpointing); Molmo2 adds image features in place into that tensor -> remove the hook
    model.get_base_model().disable_input_require_grads()
    n_vit_ck = checkpoint_vit_blocks(model)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    P(f"LoRA r={a.rank} alpha={a.alpha}: trainable {n_tr/1e6:.1f} M of {n_all/1e9:.2f} B; "
      f"dtypes {sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---------------------------------------------------------------- resume
    step = 0
    ck = os.path.join(a.out, "latest.json")
    if os.path.exists(ck):
        st = json.load(open(ck))
        d = os.path.join(a.out, st["dir"])
        set_peft_model_state_dict(model, load_file(os.path.join(d, "adapter_model.safetensors"), device=f"cuda:{local}"))
        opt.load_state_dict(torch.load(os.path.join(d, "optim.pt"), map_location=dev))
        step = st["step"]
        P(f"resumed from {d} at step {step}")
    if ddp:
        dmodel = torch.nn.parallel.DistributedDataParallel(LossModule(model), device_ids=[local],
                                                           find_unused_parameters=False)
    else:
        dmodel = LossModule(model)
    lossmod_plain = dmodel.module if ddp else dmodel
    P(f"ViT blocks checkpointed: {n_vit_ck}")
    model.train()

    per_step = world * a.grad_accum
    ds = Examples(a.data, mpath, a.seed, n_items=a.max_steps * per_step, offset=0)

    if a.probe:  # memory / speed on the longest kind of example + unused-parameter check
        idx = [i for i in range(400) if ds.spec(i)[1] == "P4" and ds.spec(i)[2] == "full"][:a.probe]
        idx = sorted(idx, key=lambda i: -max(len(ds.labs[ds.spec(i)[0]]["views"]["full"]["h"]),
                                             len(ds.labs[ds.spec(i)[0]]["views"]["full"]["v"])))
        for i in idx:
            ex = ds[i]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            loss = forward_loss(dmodel, ex, dev)
            loss.backward()
            torch.cuda.synchronize()
            P(f"probe {ex['meta']} L={ex['input_ids'].shape[1]} loss={loss.item():.3f} "
              f"{time.time()-t:.2f}s peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        P(f"unused trainable params: {len(unused)} {unused[:6]}")
        return

    # ---------------------------------------------------------------- data (rank-strided, resumable)
    class Strided(torch.utils.data.Dataset):
        def __init__(self, start):
            self.idx = list(range(start * per_step + rank, a.max_steps * per_step, world))

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, j):
            return ds[self.idx[j]]

    dl = torch.utils.data.DataLoader(Strided(step), batch_size=1, num_workers=a.workers, collate_fn=collate,
                                     prefetch_factor=4, persistent_workers=False, pin_memory=True)
    val = Examples(a.val, mpath, a.seed + 1, n_items=a.n_val)
    val_idx = list(range(rank, a.n_val, world))

    def evaluate():
        model.eval()
        s = torch.zeros(4, device=dev)   # sum loss P1, n P1, sum loss P4, n P4
        with torch.no_grad():
            for i in val_idx:
                ex = val[i]
                l = forward_loss(lossmod_plain, ex, dev)
                j = 0 if ex["meta"]["task"] == "P1" else 2
                s[j] += l
                s[j + 1] += 1
        if ddp:
            dist.all_reduce(s)
        model.train()
        return (s[0] / s[1].clamp(min=1)).item(), (s[2] / s[3].clamp(min=1)).item()

    def save(step, keep=False):
        if rank == 0:
            name = f"step{step:05d}"
            d = os.path.join(a.out, name)
            model.save_pretrained(d)
            torch.save(opt.state_dict(), os.path.join(d, "optim.pt"))
            json.dump(dict(step=step, dir=name, args=vars(a), time=time.strftime("%Y-%m-%dT%H:%M:%S")),
                      open(os.path.join(d, "state.json"), "w"), indent=1)
            json.dump(dict(step=step, dir=name), open(ck + ".tmp", "w"))
            os.replace(ck + ".tmp", ck)
            # prune: keep the newest two plus every keep-every multiple
            olds = sorted(x for x in os.listdir(a.out) if x.startswith("step"))
            for x in olds[:-2]:
                s_ = int(x[4:])
                if s_ % a.keep_every and s_ != a.max_steps:
                    import shutil
                    shutil.rmtree(os.path.join(a.out, x), ignore_errors=True)
                else:
                    o = os.path.join(a.out, x, "optim.pt")
                    if os.path.exists(o) and x != name:
                        os.remove(o)
        if ddp:
            dist.barrier()

    P(f"world={world} grad_accum={a.grad_accum} -> {per_step} examples/step; steps {step}->{a.max_steps}; "
      f"lr={a.lr} warmup={a.warmup} cosine->{a.min_lr_frac}; train crops={len(ds.labs)}")
    t0 = time.time()
    run_t0 = t0
    acc = {"P1": [0.0, 0], "P4full": [0.0, 0], "P4zoom": [0.0, 0]}
    micro = 0
    ntok = 0
    stop = False
    for ex in dl:
        m = ex["meta"]
        sync = (micro + 1) % a.grad_accum == 0
        ctx = dmodel.no_sync() if (ddp and not sync) else _null()
        with ctx:
            loss = forward_loss(dmodel, ex, dev)
            (loss / a.grad_accum).backward()
        key = "P1" if m["task"] == "P1" else ("P4full" if m["view"] == "full" else "P4zoom")
        acc[key][0] += loss.item()
        acc[key][1] += 1
        ntok += ex["input_ids"].shape[1]
        micro += 1
        if not sync:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            el = time.time() - t0
            P(f"step {step} lr {lr_at(step - 1, a):.2e} gn {gn.item():.2f} "
              + " ".join(f"{k}={v[0]/max(1,v[1]):.4f}(n{v[1]})" for k, v in acc.items())
              + f" | {el/10:.1f}s/step {ntok*world/el:.0f} tok/s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            acc = {k: [0.0, 0] for k in acc}
            t0, ntok = time.time(), 0
        if step % a.eval_every == 0 or step == a.max_steps:
            v1, v4 = evaluate()
            P(f"VAL step {step} loss_P1 {v1:.4f} loss_P4 {v4:.4f}")
        hours = (time.time() - run_t0) / 3600
        flag = torch.tensor([1.0 if hours > a.max_hours else 0.0], device=dev)
        if ddp:
            dist.all_reduce(flag)
        stop = flag.item() > 0
        if step % a.save_every == 0 or step == a.max_steps or stop:
            save(step, keep=step % a.keep_every == 0)
        if step >= a.max_steps or stop:
            break
    P(f"finished at step {step} ({'time limit' if stop else 'max steps'}) after {(time.time()-run_t0)/3600:.2f} h")
    if ddp:
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *x):
        return False


if __name__ == "__main__":
    main()
