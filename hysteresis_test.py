"""
HYSTERESIS TEST — The Cusp's Signature Prediction
==================================================
The cusp catastrophe predicts that the forward transition (numerical→structural)
and reverse transition (structural→numerical) happen at DIFFERENT thresholds.

Method:
1. Start from baseline, sweep α upward until DM flips (forward threshold)
2. Start from a flipped state (high α), sweep α downward until DM flips back (reverse threshold)
3. If forward ≠ reverse → hysteresis → cusp geometry confirmed
4. If forward == reverse → smooth sigmoid, no cusp

Since each forward pass is independent (no recurrent state), we simulate hysteresis
by using CONTEXT as the slow variable: we test prompts that start in different basins
and measure what α is needed to cross the fold from each side.

We also do a fine-grained sweep to check whether the transition is truly discontinuous
(jump between two stable states) vs continuous (smooth sigmoid).
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

print("=" * 65)
print("HYSTERESIS TEST — Cusp Signature")
print("=" * 65)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DIGIT_TOKENS = [tok.encode(str(d))[0] for d in range(10)]

# Training prompts (same as main paper)
struct = ["She opened the door.", "He walked in.", "The cat sat quietly.", "It was cold."]
numer = ["The distance was 26.", "It costs about 50.", "She ran exactly 5.", "He counted to 15."]

def get_hidden(text, layer):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].numpy()

def compute_fold(layer):
    s = np.array([get_hidden(t, layer) for t in struct])
    n = np.array([get_hidden(t, layer) for t in numer])
    fold = np.mean(s, axis=0) - np.mean(n, axis=0)
    return fold / np.linalg.norm(fold)

def get_dm(prompt, layer, alpha, fold):
    """Get digit mass under fold injection at given layer and alpha."""
    ids = torch.tensor([tok.encode(prompt)])
    tpos = ids.shape[1] - 1
    
    def hook(module, input, output, _f=fold, _a=alpha, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    return sum(probs[t].item() for t in DIGIT_TOKENS)

print("\nComputing fold directions...")
folds = {}
for L in range(12):
    folds[L] = compute_fold(L)

start = time.time()

# ═══════════════════════════════════════════════════════════════
# TEST 1: Forward vs Reverse threshold (CREATE vs DESTROY)
# ═══════════════════════════════════════════════════════════════
# CREATE = push numerical prompt toward structural (positive α)
# DESTROY = push structural prompt toward numerical (negative α)
# Hysteresis predicts: CREATE threshold ≠ DESTROY threshold

print(f"\n{'='*65}")
print("TEST 1: Forward/Reverse Threshold Asymmetry")
print(f"{'='*65}")

layer = 3

# Numerical prompts (high baseline DM → push toward structural)
num_prompts = [
    "The temperature was 98.",
    "She scored 42.",
    "He measured exactly 3.",
    "The price was 7.",
]

# Structural prompts (low baseline DM → push toward numerical)  
struct_prompts = [
    "She opened the door.",
    "He walked in slowly.",
    "It was over.",
    "The meeting ended.",
]

fold = folds[layer]

print(f"\nCREATE direction (numerical → structural, positive α at L{layer}):")
print(f"  {'Prompt':<35s}  {'Base DM':>8s}  {'Threshold α':>12s}")
print(f"  {'-'*60}")

create_thresholds = []
for p in num_prompts:
    base = get_dm(p, layer, 0, fold)
    # Binary search for threshold where DM drops below 0.1
    lo, hi = 0, 150
    for _ in range(20):
        mid = (lo + hi) / 2
        dm = get_dm(p, layer, mid, fold)
        if dm < 0.1:
            hi = mid
        else:
            lo = mid
    threshold = (lo + hi) / 2
    create_thresholds.append(threshold)
    print(f"  {p:<35s}  {base:>8.4f}  {threshold:>12.1f}")

print(f"\n  Mean CREATE threshold: {np.mean(create_thresholds):.1f} (range: {min(create_thresholds):.0f}–{max(create_thresholds):.0f})")

print(f"\nDESTROY direction (structural → numerical, negative α at L{layer}):")
print(f"  {'Prompt':<35s}  {'Base DM':>8s}  {'Threshold α':>12s}")
print(f"  {'-'*60}")

destroy_thresholds = []
for p in struct_prompts:
    base = get_dm(p, layer, 0, fold)
    # Binary search for threshold where DM rises above 0.5
    lo, hi = 0, 200
    for _ in range(20):
        mid = (lo + hi) / 2
        dm = get_dm(p, layer, -mid, fold)  # negative = push toward numerical
        if dm > 0.5:
            hi = mid
        else:
            lo = mid
    threshold = (lo + hi) / 2
    # Check if we actually reached 0.5
    final_dm = get_dm(p, layer, -threshold, fold)
    if final_dm < 0.5:
        destroy_thresholds.append(float('inf'))
        print(f"  {p:<35s}  {base:>8.4f}  {'> 200':>12s}")
    else:
        destroy_thresholds.append(threshold)
        print(f"  {p:<35s}  {base:>8.4f}  {threshold:>12.1f}")

finite_destroy = [t for t in destroy_thresholds if t != float('inf')]
if finite_destroy:
    print(f"\n  Mean DESTROY threshold: {np.mean(finite_destroy):.1f} (range: {min(finite_destroy):.0f}–{max(finite_destroy):.0f})")
else:
    print(f"\n  DESTROY threshold: unreachable at α=200 for all prompts")

if finite_destroy:
    ratio = np.mean(finite_destroy) / np.mean(create_thresholds)
    print(f"\n  ASYMMETRY RATIO (DESTROY/CREATE): {ratio:.1f}:1")
    print(f"  Cusp prediction: ratio ≠ 1.0")
    print(f"  Result: {'HYSTERESIS CONFIRMED' if ratio > 1.5 else 'WEAK OR NO HYSTERESIS'}")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Fine-grained sweep — is the transition discontinuous?
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("TEST 2: Fine-Grained Transition Profile")
print("(Is the jump discontinuous or a smooth sigmoid?)")
print(f"{'='*65}")

prompt = "The temperature was 98."
ids = torch.tensor([tok.encode(prompt)])

# Very fine sweep through critical band
alphas = list(np.arange(0, 45, 0.5))

print(f"\nPrompt: {prompt}")
print(f"Layer: L{layer}")
print(f"{'α':>6s}  {'DM':>8s}  {'log-odds':>10s}  {'Bar':>30s}")
print(f"{'-'*60}")

dms = []
log_odds_list = []
for a in alphas:
    dm = get_dm(prompt, layer, a, fold)
    dms.append(dm)
    lo = np.log(dm / (1 - dm + 1e-12)) if 0 < dm < 1 else (10 if dm >= 1 else -10)
    log_odds_list.append(lo)
    bar = "█" * int(dm * 40)
    print(f"{a:>6.1f}  {dm:>8.4f}  {lo:>+10.3f}  {bar}")

# Find the steepest drop
max_drop = 0
max_drop_alpha = 0
for i in range(1, len(dms)):
    drop = dms[i-1] - dms[i]
    if drop > max_drop:
        max_drop = drop
        max_drop_alpha = alphas[i]

print(f"\n  Steepest single-step drop: ΔDM = {max_drop:.4f} at α = {max_drop_alpha:.1f}")
print(f"  (Step size = 0.5, so this is ΔDM per 0.5 units of α)")

# Check for bimodality proxy: is there a range where DM is between 0.2 and 0.8?
mid_range = [(a, d) for a, d in zip(alphas, dms) if 0.2 < d < 0.8]
print(f"\n  Steps with 0.2 < DM < 0.8: {len(mid_range)} (out of {len(alphas)} total)")
if mid_range:
    mid_alphas = [a for a, d in mid_range]
    print(f"  Transition band width: α = {min(mid_alphas):.1f} to {max(mid_alphas):.1f} (width = {max(mid_alphas)-min(mid_alphas):.1f})")
else:
    print(f"  NO intermediate states — transition is discontinuous (jumps over 0.2-0.8 range)")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Log-odds linearity check
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("TEST 3: Log-Odds Space Analysis")
print("(Cusp → kink in log-odds | Sigmoid → linear in log-odds)")
print(f"{'='*65}")

# In log-odds space, a smooth sigmoid should be approximately linear
# A cusp should show a kink (sudden change in slope)

# Compute numerical derivative of log-odds w.r.t. alpha
print(f"\n  {'α':>6s}  {'d(log-odds)/dα':>16s}")
print(f"  {'-'*30}")

derivatives = []
for i in range(1, len(log_odds_list)):
    dloda = (log_odds_list[i] - log_odds_list[i-1]) / (alphas[i] - alphas[i-1])
    derivatives.append(dloda)
    if i % 4 == 0:  # print every 4th for readability
        print(f"  {alphas[i]:>6.1f}  {dloda:>+16.4f}")

# Find max |derivative| (the kink)
abs_derivs = [abs(d) for d in derivatives]
max_deriv_idx = np.argmax(abs_derivs)
max_deriv_alpha = alphas[max_deriv_idx + 1]
max_deriv_val = derivatives[max_deriv_idx]

# Compare slope in three regions: pre-transition, transition, post-transition
pre = [d for a, d in zip(alphas[1:], derivatives) if a < 15]
mid = [d for a, d in zip(alphas[1:], derivatives) if 20 < a < 30]
post = [d for a, d in zip(alphas[1:], derivatives) if a > 35]

pre_mean = np.mean(pre) if pre else 0
mid_mean = np.mean(mid) if mid else 0
post_mean = np.mean(post) if post else 0

print(f"\n  Peak slope at α = {max_deriv_alpha:.1f}: {max_deriv_val:+.4f}")
print(f"\n  Mean slope by region:")
print(f"    Pre-transition  (α < 15):  {pre_mean:+.4f}")
print(f"    Critical band   (20 < α < 30): {mid_mean:+.4f}")
print(f"    Post-transition (α > 35):  {post_mean:+.4f}")

if mid:
    acceleration = abs(mid_mean / pre_mean) if pre_mean != 0 else float('inf')
    print(f"\n  Slope acceleration (critical/pre): {acceleration:.1f}×")
    print(f"  Cusp prediction: acceleration >> 1 (kink in log-odds)")
    print(f"  Sigmoid prediction: acceleration ≈ 1 (linear in log-odds)")
    print(f"  Result: {'KINK DETECTED — supports cusp' if acceleration > 3 else 'Smooth — consistent with sigmoid'}")

elapsed = time.time() - start
print(f"\n{'='*65}")
print(f"Done in {elapsed:.1f}s")
print(f"{'='*65}")
