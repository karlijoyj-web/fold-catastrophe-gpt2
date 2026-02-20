"""
INCEPTION DOSE-RESPONSE
=======================
Sweep alpha 0→100 on fairy tale prompt, measuring digit mass sigmoid.
Also test reverse inception: structural into math prompt.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

print("=" * 60)
print("INCEPTION DOSE-RESPONSE")
print("=" * 60)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DIGIT_TOKENS = [tok.encode(str(d))[0] for d in range(10)]

struct_prompts = ["She opened the door.", "He walked in.", "The cat sat quietly.", "It was cold."]
numer_prompts = ["The distance was 26.", "It costs about 50.", "She ran exactly 5.", "He counted to 15."]

def get_hidden(text, layer):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy()

def digit_mass(probs):
    return sum(probs[t].item() for t in DIGIT_TOKENS)

# Precompute folds
print("Computing folds at all layers...")
folds = {}
for L in range(12):
    s = np.array([get_hidden(t, L) for t in struct_prompts])
    n = np.array([get_hidden(t, L) for t in numer_prompts])
    fold = np.mean(s, axis=0) - np.mean(n, axis=0)
    folds[L] = fold / np.linalg.norm(fold)

# ═══════════════════════════════════════════════════════════════
# INCEPTION DOSE-RESPONSE
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("INCEPTION DOSE-RESPONSE")
print("Inject numerical fold into fairy tale prompt at L4")
print(f"{'='*60}")

fairy_tale = "The beautiful princess walked into the grand"
fairy_ids = torch.tensor([tok.encode(fairy_tale)])
fairy_tpos = fairy_ids.shape[1] - 1

# Baseline fairy tale DM
with torch.no_grad():
    out = model(fairy_ids)
fairy_baseline = digit_mass(torch.softmax(out.logits[0, fairy_tpos, :], dim=-1))
print(f"\nFairy tale baseline DM: {fairy_baseline:.4f}")

# Also get baseline for numerical prompt  
math_prompt = "The temperature was 98."
math_ids = torch.tensor([tok.encode(math_prompt)])
math_tpos = math_ids.shape[1] - 1
with torch.no_grad():
    out = model(math_ids)
math_baseline = digit_mass(torch.softmax(out.logits[0, math_tpos, :], dim=-1))
print(f"Math prompt baseline DM: {math_baseline:.4f}")

# Sweep: inject NEGATIVE alpha (numerical direction) into fairy tale at L4
print(f"\nDose-response (negative α = push toward numerical):")
print(f"  {'α':>6s}  {'DM':>8s}  {'Bar':>30s}")
print(f"  {'-'*50}")

alphas = list(range(0, 110, 10))
inception_dms = []
for a in alphas:
    fold4 = folds[4]
    def hook(module, input, output, _f=fold4, _a=-a, _t=fairy_tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[4].register_forward_hook(hook)
    with torch.no_grad():
        out = model(fairy_ids)
    handle.remove()
    dm = digit_mass(torch.softmax(out.logits[0, fairy_tpos, :], dim=-1))
    inception_dms.append(dm)
    bar = "█" * int(dm * 40)
    print(f"  {a:>6d}  {dm:>8.4f}  {bar}")

# Find the critical alpha (first time DM crosses 0.5)
# Note: for forward inception (fairy tale → numerical), this typically never
# crosses — the structural basin is too deep. That IS the finding.
critical_alpha = None
for i in range(1, len(alphas)):
    if inception_dms[i] > 0.5 and inception_dms[i-1] <= 0.5:
        # Linear interpolation
        frac = (0.5 - inception_dms[i-1]) / (inception_dms[i] - inception_dms[i-1])
        critical_alpha = alphas[i-1] + frac * (alphas[i] - alphas[i-1])
        break

max_dm = max(inception_dms)
print(f"\n  Max DM reached: {max_dm:.4f}")
print(f"  Critical α (DM crosses 0.5): ", end="")
if critical_alpha is not None:
    print(f"{critical_alpha:.1f}")
else:
    print(f"UNREACHABLE (max DM = {max_dm:.4f} — structural basin too deep)")

# Also sweep the REVERSE: structural into math prompt
print(f"\nReverse inception (positive α = push math prompt toward structural):")
print(f"  {'α':>6s}  {'DM':>8s}  {'Bar':>30s}")
print(f"  {'-'*50}")

reverse_dms = []
for a in alphas:
    fold4 = folds[4]
    def hook(module, input, output, _f=fold4, _a=a, _t=math_tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[4].register_forward_hook(hook)
    with torch.no_grad():
        out = model(math_ids)
    handle.remove()
    dm = digit_mass(torch.softmax(out.logits[0, math_tpos, :], dim=-1))
    reverse_dms.append(dm)
    bar = "█" * int(dm * 40)
    print(f"  {a:>6d}  {dm:>8.4f}  {bar}")

# Find reverse critical alpha (first time DM drops below 0.5)
reverse_critical = None
for i in range(1, len(alphas)):
    if reverse_dms[i] < 0.5 and reverse_dms[i-1] >= 0.5:
        frac = (0.5 - reverse_dms[i-1]) / (reverse_dms[i] - reverse_dms[i-1])
        reverse_critical = alphas[i-1] + frac * (alphas[i] - alphas[i-1])
        break

print(f"\n  Reverse critical α (DM drops below 0.5): ", end="")
if reverse_critical is not None:
    print(f"{reverse_critical:.1f}")
else:
    print(f"not found in range")

# Asymmetry summary
print(f"\n{'='*60}")
print("ASYMMETRY SUMMARY")
print(f"{'='*60}")
print(f"  Forward (fairy tale → numerical):  max DM = {max_dm:.4f} at α=100")
print(f"  Reverse (math → structural):       crosses 0.5 at α ≈ {reverse_critical:.1f}" if reverse_critical is not None else f"  Reverse (math → structural):       did not cross")
if reverse_critical is not None and max_dm < 0.5:
    print(f"  Asymmetry: structural basin UNREACHABLE from fairy tale")
    print(f"             numerical basin exits at α ≈ {reverse_critical:.1f}")
    print(f"  Interpretation: structural is the deep/default basin,")
    print(f"                  numerical is shallow/contextual")

print(f"\n{'='*60}")
print("Done.")
