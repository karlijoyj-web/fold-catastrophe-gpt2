"""
PYTHIA-410M FOLD TEST — Does It Scale?
======================================
Run the core fold geometry tests on EleutherAI's Pythia-410M:
  1. Baseline digit mass
  2. Sharp fold transition (fine-grained α sweep)
  3. Directional specificity (fold vs 20 random directions)
  4. Log-odds kink (cusp vs sigmoid discrimination)
  5. Basin asymmetry (CREATE vs DESTROY thresholds)

Pythia-410M: 24 layers, 1024 dimensions, GPT-NeoX architecture.
Should run in ~2-3 minutes on M4 MacBook.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("=" * 65)
print("PYTHIA-410M FOLD TEST — Does It Scale?")
print("=" * 65)

model_name = "EleutherAI/pythia-410m"
print(f"\nLoading {model_name}...")
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
n_layers = model.config.num_hidden_layers
n_dims = model.config.hidden_size
print(f"  {n_layers} layers, {n_dims} dimensions")

# Find digit tokens for this tokenizer
DIGIT_TOKENS = []
for d in range(10):
    ids = tok.encode(str(d))
    # Take the last token (some tokenizers add BOS)
    DIGIT_TOKENS.append(ids[-1])
print(f"  Digit tokens: {DIGIT_TOKENS}")
print(f"  Decoded check: {[tok.decode([t]) for t in DIGIT_TOKENS]}")

# Training prompts
struct_prompts = ["She opened the door.", "He walked in.", "The cat sat quietly.", "It was cold."]
numer_prompts = ["The distance was 26.", "It costs about 50.", "She ran exactly 5.", "He counted to 15."]

def get_hidden(text, layer):
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[layer]
    if hs.dim() == 3:
        return hs[0, -1, :].numpy()
    else:
        return hs[-1, :].numpy()

def compute_fold(layer):
    s = np.array([get_hidden(t, layer) for t in struct_prompts])
    n = np.array([get_hidden(t, layer) for t in numer_prompts])
    fold = np.mean(s, axis=0) - np.mean(n, axis=0)
    norm = np.linalg.norm(fold)
    if norm < 1e-10:
        return fold  # degenerate layer, won't be used
    return fold / norm

def get_dm_and_logits(prompt, layer, alpha, fold):
    """Get digit mass and raw logits under fold injection."""
    ids = torch.tensor([tok.encode(prompt)])
    tpos = ids.shape[1] - 1
    
    # Find the right hook point — Pythia uses gpt_neox.layers[i]
    def hook(module, input, output, _f=fold, _a=alpha, _t=tpos):
        if isinstance(output, tuple):
            h = output[0].clone()
            if h.dim() == 3:
                h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            else:
                h[_t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            return (h,) + tuple(output[i] for i in range(1, len(output)))
        else:
            h = output.clone()
            if h.dim() == 3:
                h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            else:
                h[_t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            return h
    
    handle = model.gpt_neox.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    logits = out.logits[0, tpos, :]
    probs = torch.softmax(logits, dim=-1)
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    return dm, logits

def get_dm(prompt, layer, alpha, fold):
    dm, _ = get_dm_and_logits(prompt, layer, alpha, fold)
    return dm

start = time.time()

# ═══════════════════════════════════════════════════════════════
# STEP 0: Find the commitment window
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 0: Finding Commitment Window")
print(f"{'='*65}")

prompt = "The temperature was 98."
print(f"Prompt: {prompt}")

# Compute folds at all layers and test moderate injection
print(f"\nComputing folds and testing α=30 at each layer...")
print(f"  {'Layer':>5s}  {'DM (α=0)':>9s}  {'DM (α=30)':>10s}  {'Drop':>8s}")
print(f"  {'-'*40}")

folds = {}
layer_drops = []
baseline_dm = None
for L in range(n_layers):
    folds[L] = compute_fold(L)
    # Skip degenerate layers (zero-norm fold → NaN)
    if np.any(np.isnan(folds[L])) or np.linalg.norm(folds[L]) < 1e-10:
        print(f"  L{L:>3d}  {'(degenerate — skipped)':>35s}")
        continue
    dm0 = get_dm(prompt, L, 0, folds[L])
    if baseline_dm is None or np.isnan(baseline_dm):
        baseline_dm = dm0
    dm30 = get_dm(prompt, L, 30, folds[L])
    drop = dm0 - dm30
    layer_drops.append((L, drop, dm0, dm30))
    marker = " ←" if drop > 0.3 else ""
    print(f"  L{L:>3d}  {dm0:>9.4f}  {dm30:>10.4f}  {drop:>+8.4f}{marker}")

# Find commitment layer: earliest layer with drop > 0.5 (strong transition)
# Later layers trivially inherit the effect, so we want the first one
strong_layers = [(L, drop) for L, drop, _, _ in layer_drops if drop > 0.5]
if strong_layers:
    commit_layer = strong_layers[0][0]
    print(f"\n  Baseline DM: {baseline_dm:.4f}")
    print(f"  Commitment layer (earliest with drop > 0.5): L{commit_layer}")
    print(f"  (Later layers also work but inherit the effect)")
else:
    # Fallback to max drop
    best_layer = max(layer_drops, key=lambda x: x[1])
    commit_layer = best_layer[0]
    print(f"\n  Baseline DM: {baseline_dm:.4f}")
    print(f"  Best layer for fold injection: L{commit_layer} (drop = {best_layer[1]:.4f})")

# For GPT-2: commitment at L2-3. For Pythia-160M: L3-4. 
# For Pythia-410M: predict L3-5 range.

# ═══════════════════════════════════════════════════════════════
# STEP 1: Fine-grained α sweep at the commitment layer
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"STEP 1: Fine-Grained α Sweep at L{commit_layer}")
print(f"{'='*65}")

fold = folds[commit_layer]
alphas = list(np.arange(0, 60, 1))

print(f"\n{'α':>6s}  {'DM':>8s}  {'log-odds':>10s}  {'Bar':>30s}")
print(f"{'-'*60}")

dms = []
log_odds = []
for a in alphas:
    dm = get_dm(prompt, commit_layer, a, fold)
    dms.append(dm)
    lo = np.log(dm / (1 - dm + 1e-12)) if 0 < dm < 1 else (10 if dm >= 1 else -10)
    log_odds.append(lo)
    bar = "█" * int(dm * 40)
    print(f"{a:>6.0f}  {dm:>8.4f}  {lo:>+10.3f}  {bar}")

# Find steepest drop
max_drop = 0
critical_alpha = 0
for i in range(1, len(dms)):
    drop = dms[i-1] - dms[i]
    if drop > max_drop:
        max_drop = drop
        critical_alpha = alphas[i]

print(f"\n  Steepest drop: ΔDM = {max_drop:.4f} at α = {critical_alpha}")

# Transition band
mid_range = [(a, d) for a, d in zip(alphas, dms) if 0.2 < d < 0.8]
if mid_range:
    width = max(a for a, d in mid_range) - min(a for a, d in mid_range)
    print(f"  Transition band (0.2 < DM < 0.8): width = {width:.0f} units of α")
else:
    print(f"  Transition band: DM jumps over 0.2-0.8 range (discontinuous)")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Log-odds kink test
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STEP 2: Log-Odds Kink Test")
print(f"{'='*65}")

derivatives = []
for i in range(1, len(log_odds)):
    d = (log_odds[i] - log_odds[i-1]) / (alphas[i] - alphas[i-1])
    derivatives.append(d)

pre = [d for a, d in zip(alphas[1:], derivatives) if a < critical_alpha - 10]
mid = [d for a, d in zip(alphas[1:], derivatives) if critical_alpha - 5 < a < critical_alpha + 5]
post = [d for a, d in zip(alphas[1:], derivatives) if a > critical_alpha + 10]

pre_mean = np.mean(pre) if pre else 0.001
mid_mean = np.mean(mid) if mid else 0
post_mean = np.mean(post) if post else 0

acceleration = abs(mid_mean / pre_mean) if abs(pre_mean) > 0.001 else float('inf')

print(f"  Pre-transition slope:  {pre_mean:+.4f}")
print(f"  Critical band slope:   {mid_mean:+.4f}")
print(f"  Post-transition slope: {post_mean:+.4f}")
print(f"  Slope acceleration:    {acceleration:.1f}×")
print(f"  Result: {'KINK DETECTED' if acceleration > 3 else 'No significant kink'}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Directional specificity
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"STEP 3: Directional Specificity at L{commit_layer}")
print(f"{'='*65}")

# Use the critical alpha we found
test_alpha = int(critical_alpha) + 2  # slightly past critical
fold_dm = get_dm(prompt, commit_layer, test_alpha, fold)
print(f"\n  Fold direction at α={test_alpha}: DM = {fold_dm:.4f}")

n_random = 20
random_dms = []
np.random.seed(42)
for i in range(n_random):
    rand_dir = np.random.randn(n_dims).astype(np.float32)
    rand_dir /= np.linalg.norm(rand_dir)
    rdm = get_dm(prompt, commit_layer, test_alpha, rand_dir)
    random_dms.append(rdm)

random_flips = sum(1 for d in random_dms if d < 0.1)
print(f"  Random directions at α={test_alpha} (n={n_random}):")
print(f"    Mean DM: {np.mean(random_dms):.4f} ± {np.std(random_dms):.4f}")
print(f"    Phase transitions (DM < 0.1): {random_flips}/{n_random}")
print(f"  Result: {'DIRECTIONAL — fold unique' if random_flips == 0 and fold_dm < 0.1 else 'CHECK NEEDED'}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Basin asymmetry (CREATE vs DESTROY)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"STEP 4: Basin Asymmetry at L{commit_layer}")
print(f"{'='*65}")

num_test = [
    "The temperature was 98.",
    "She scored 42.",
    "He measured exactly 3.",
]
struct_test = [
    "She opened the door.",
    "It was over.",
    "The meeting ended.",
]

print(f"\n  CREATE (numerical → structural):")
create_thresholds = []
for p in num_test:
    base = get_dm(p, commit_layer, 0, fold)
    lo, hi = 0, 150
    for _ in range(20):
        m = (lo + hi) / 2
        if get_dm(p, commit_layer, m, fold) < 0.1:
            hi = m
        else:
            lo = m
    t = (lo + hi) / 2
    create_thresholds.append(t)
    print(f"    {p:<35s}  base={base:.4f}  threshold={t:.1f}")

print(f"\n  DESTROY (structural → numerical):")
destroy_thresholds = []
for p in struct_test:
    base = get_dm(p, commit_layer, 0, fold)
    lo, hi = 0, 200
    for _ in range(20):
        m = (lo + hi) / 2
        if get_dm(p, commit_layer, -m, fold) > 0.5:
            hi = m
        else:
            lo = m
    t = (lo + hi) / 2
    final = get_dm(p, commit_layer, -t, fold)
    if final < 0.5:
        print(f"    {p:<35s}  base={base:.4f}  threshold=> 200")
        destroy_thresholds.append(float('inf'))
    else:
        print(f"    {p:<35s}  base={base:.4f}  threshold={t:.1f}")
        destroy_thresholds.append(t)

finite_d = [t for t in destroy_thresholds if t != float('inf')]
if finite_d and create_thresholds:
    ratio = np.mean(finite_d) / np.mean(create_thresholds)
    print(f"\n  Mean CREATE: {np.mean(create_thresholds):.1f}")
    print(f"  Mean DESTROY: {np.mean(finite_d):.1f}")
    print(f"  ASYMMETRY RATIO: {ratio:.1f}:1")
elif not finite_d:
    print(f"\n  Mean CREATE: {np.mean(create_thresholds):.1f}")
    print(f"  DESTROY: unreachable at α=200 (extreme asymmetry)")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - start

print(f"\n{'='*65}")
print("SUMMARY — PYTHIA-410M")
print(f"{'='*65}")
print(f"  Model: {model_name} ({n_layers} layers, {n_dims} dims)")
print(f"  Baseline DM: {baseline_dm:.4f}")
print(f"  Commitment layer: L{commit_layer}")
print(f"  Sharp transition: {'YES' if max_drop > 0.2 else 'NO'} (max ΔDM = {max_drop:.4f})")
print(f"  Log-odds kink: {'YES' if acceleration > 3 else 'NO'} ({acceleration:.1f}× acceleration)")
print(f"  Directional specificity: {random_flips}/{n_random} random flips (fold DM = {fold_dm:.4f})")
if finite_d and create_thresholds:
    print(f"  Basin asymmetry: {ratio:.1f}:1 (CREATE={np.mean(create_thresholds):.0f}, DESTROY={np.mean(finite_d):.0f})")
else:
    print(f"  Basin asymmetry: extreme (DESTROY unreachable)")
print(f"\n  Time: {elapsed:.1f}s")
print(f"{'='*65}")
