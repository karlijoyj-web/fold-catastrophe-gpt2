"""
STATISTICAL CONFABULATION TEST
==============================
Run 20 different numerical prompts through the split-brain condition.
Classify outputs and measure the dissociation: preserved semantics + lost format.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

print("=" * 60)
print("STATISTICAL CONFABULATION TEST")
print("=" * 60)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DIGIT_TOKENS = set(tok.encode(str(d))[0] for d in range(10))

struct_prompts = ["She opened the door.", "He walked in.", "The cat sat quietly.", "It was cold."]
numer_prompts = ["The distance was 26.", "It costs about 50.", "She ran exactly 5.", "He counted to 15."]

def get_hidden(text, layer):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy()

def compute_fold(layer):
    s = np.array([get_hidden(t, layer) for t in struct_prompts])
    n = np.array([get_hidden(t, layer) for t in numer_prompts])
    fold = np.mean(s, axis=0) - np.mean(n, axis=0)
    return fold / np.linalg.norm(fold)

print("Computing folds...")
fold_l3 = compute_fold(3)
fold_l8 = compute_fold(8)

# 20 diverse numerical prompts
test_prompts = [
    "The temperature was 98.",
    "The distance was 26.",
    "It costs about 50.",
    "She ran exactly 5.",
    "He counted to 15.",
    "The building has 42.",
    "The price dropped to 30.",
    "She scored 88.",
    "The population reached 200.",
    "The test result was 75.",
    "The car traveled 60.",
    "He weighed about 180.",
    "The room held 25.",
    "The budget was 100.",
    "The voltage was 12.",
    "She waited for 45.",
    "The altitude reached 8.",
    "The percentage was 95.",
    "He measured exactly 6.",
    "The countdown started at 10.",
]

def generate(prompt, alpha_l3, alpha_l8, max_tokens=15):
    """Generate tokens with split-brain injection. Returns list of token ids."""
    input_ids = tok.encode(prompt)
    generated = []
    
    for step in range(max_tokens):
        ids_tensor = torch.tensor([input_ids])
        tpos = len(input_ids) - 1
        handles = []
        
        if alpha_l3 != 0:
            fold3 = fold_l3  # capture
            def hook_l3(module, input, output, _f=fold3, _a=alpha_l3, _t=tpos):
                h = output[0].clone()
                h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
                return (h,) + output[1:]
            handles.append(model.transformer.h[3].register_forward_hook(hook_l3))
            
        if alpha_l8 != 0:
            fold8 = fold_l8
            def hook_l8(module, input, output, _f=fold8, _a=alpha_l8, _t=tpos):
                h = output[0].clone()
                h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
                return (h,) + output[1:]
            handles.append(model.transformer.h[8].register_forward_hook(hook_l8))

        with torch.no_grad():
            out = model(ids_tensor)
        for h in handles:
            h.remove()
            
        next_token = torch.argmax(out.logits[0, -1, :]).item()
        input_ids.append(next_token)
        generated.append(next_token)
        
        if len(generated) > 2 and generated[-1] == 198 and generated[-2] == 198:
            break
    
    return generated

def classify_output(tokens):
    """Classify: digits=has digit tokens, structural=has word tokens, confab=structural despite numerical prompt"""
    digit_count = sum(1 for t in tokens if t in DIGIT_TOKENS)
    word_count = len(tokens) - digit_count
    text = tok.decode(tokens)
    
    if digit_count == 0 and word_count > 0:
        return "confabulation", text
    elif digit_count > word_count:
        return "numerical", text
    elif digit_count > 0 and word_count > 0:
        return "hybrid", text
    else:
        return "structural", text

def get_period_dm(prompt, alpha_l3, alpha_l8):
    """Measure DM at the period position under injection (cleaner than counting output tokens)."""
    input_ids = tok.encode(prompt)
    ids_tensor = torch.tensor([input_ids])
    tpos = len(input_ids) - 1
    handles = []
    if alpha_l3 != 0:
        fold3 = fold_l3
        def hook_l3(module, inp, output, _f=fold3, _a=alpha_l3, _t=tpos):
            h = output[0].clone()
            h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            return (h,) + output[1:]
        handles.append(model.transformer.h[3].register_forward_hook(hook_l3))
    if alpha_l8 != 0:
        fold8 = fold_l8
        def hook_l8(module, inp, output, _f=fold8, _a=alpha_l8, _t=tpos):
            h = output[0].clone()
            h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            return (h,) + output[1:]
        handles.append(model.transformer.h[8].register_forward_hook(hook_l8))
    with torch.no_grad():
        out = model(ids_tensor)
    for h in handles:
        h.remove()
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    return sum(probs[t].item() for t in DIGIT_TOKENS)

start = time.time()

# Run all 20 prompts in all conditions
print(f"\n{'='*60}")
print("CONDITION 1: BASELINE (no injection)")
print(f"{'='*60}")
baseline_results = []
for p in test_prompts:
    tokens = generate(p, 0, 0)
    cat, text = classify_output(tokens)
    baseline_results.append(cat)
    print(f"  [{cat:>14s}] {p} →{text}")

print(f"\n{'='*60}")
print("CONDITION 2: SPLIT BRAIN (L3=-50 math, L8=+50 syntax)")
print(f"{'='*60}")
split_results = []
split_texts = []
for p in test_prompts:
    tokens = generate(p, -50, 50)
    cat, text = classify_output(tokens)
    split_results.append(cat)
    split_texts.append(text)
    print(f"  [{cat:>14s}] {p} →{text}")

print(f"\n{'='*60}")
print("CONDITION 3: UNCONSCIOUS MATH ONLY (L3=-50, L8=0)")
print(f"{'='*60}")
math_results = []
for p in test_prompts:
    tokens = generate(p, -50, 0)
    cat, text = classify_output(tokens)
    math_results.append(cat)
    print(f"  [{cat:>14s}] {p} →{text}")

elapsed = time.time() - start

# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")

for name, results in [("Baseline", baseline_results), ("Split Brain", split_results), ("Math Only", math_results)]:
    counts = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    print(f"\n  {name}:")
    for cat in ["numerical", "hybrid", "structural", "confabulation"]:
        n = counts.get(cat, 0)
        pct = n / len(results) * 100
        bar = "█" * int(pct / 3)
        print(f"    {cat:>14s}: {n:>2d}/20 ({pct:5.1f}%) {bar}")

# Key test: does split-brain produce confabulation?
split_confab = sum(1 for r in split_results if r in ["confabulation", "structural"])
split_numerical = sum(1 for r in split_results if r == "numerical")
base_numerical = sum(1 for r in baseline_results if r == "numerical")

# Secondary validation: DM at the period position (cleaner metric)
print(f"\n{'='*60}")
print("DM-BASED VALIDATION (digit mass at period position)")
print(f"{'='*60}")
base_dms = []
split_dms = []
for p in test_prompts:
    bdm = get_period_dm(p, 0, 0)
    sdm = get_period_dm(p, -50, 50)
    base_dms.append(bdm)
    split_dms.append(sdm)
    suppressed = "✓" if sdm < 0.1 else " "
    print(f"  {suppressed} {p:<35s}  base={bdm:.4f}  split={sdm:.4f}")
dm_suppressed = sum(1 for d in split_dms if d < 0.1)
print(f"\n  DM < 0.1 under split-brain: {dm_suppressed}/20")
print(f"  Mean baseline DM: {np.mean(base_dms):.4f}")
print(f"  Mean split-brain DM: {np.mean(split_dms):.4f}")

print(f"\n{'='*60}")
print("KEY FINDINGS")
print(f"{'='*60}")
print(f"  Baseline numerical outputs:     {base_numerical}/20")
print(f"  Split-brain confabulations:     {split_confab}/20")
print(f"  Split-brain numerical outputs:  {split_numerical}/20")
print(f"  Confabulation rate:             {split_confab/20*100:.0f}%")
print(f"  Format suppression successful:  {split_confab > 14}/20 (>{14})")
print(f"\n  Time: {elapsed:.1f}s")
print(f"{'='*60}")
