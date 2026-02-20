"""
BIMODALITY TEST — Two Attractors or One Sliding Peak?
=====================================================
At the critical α (right at the transition), examine the full probability
distribution over the vocabulary.

Cusp catastrophe predicts: BIMODAL distribution — two competing peaks
(digit tokens and structural tokens) with a valley between them.
The system is caught between two attractors.

Smooth sigmoid predicts: UNIMODAL distribution that gradually shifts
its center from digits to structural tokens.

Method:
1. For each α in a fine sweep, get full probability distribution
2. Compute probability mass in digit tokens vs top structural tokens
3. At the critical point, check if both clusters have substantial mass
   with a gap between them (bimodal) or if mass smoothly transfers (unimodal)
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

print("=" * 65)
print("BIMODALITY TEST — Two Attractors or One Sliding Peak?")
print("=" * 65)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DIGIT_TOKENS = [tok.encode(str(d))[0] for d in range(10)]
DIGIT_TOKEN_SET = set(DIGIT_TOKENS)

# Training prompts
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

print("Computing fold direction...")
layer = 3
fold = compute_fold(layer)

prompt = "The temperature was 98."
ids = torch.tensor([tok.encode(prompt)])
tpos = ids.shape[1] - 1

start = time.time()

# ═══════════════════════════════════════════════════════════════
# PART 1: Distribution snapshots at key alpha values
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PART 1: Top-Token Distribution Snapshots")
print(f"{'='*65}")

# First, find the critical alpha with a coarse sweep
coarse_alphas = list(range(0, 50))
coarse_dms = []
for a in coarse_alphas:
    def hook(module, input, output, _f=fold, _a=a, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    coarse_dms.append(dm)

# Find critical band (where DM transitions most sharply)
max_drop = 0
critical_alpha = 25  # default
for i in range(1, len(coarse_dms)):
    drop = coarse_dms[i-1] - coarse_dms[i]
    if drop > max_drop:
        max_drop = drop
        critical_alpha = coarse_alphas[i]

print(f"Critical α detected at: {critical_alpha} (steepest DM drop: {max_drop:.4f})")

# Snapshot alphas: well before, approaching, critical, just past, well past
snapshot_alphas = [0, critical_alpha - 5, critical_alpha - 2, critical_alpha - 1,
                   critical_alpha, critical_alpha + 1, critical_alpha + 2,
                   critical_alpha + 5, critical_alpha + 10]
snapshot_alphas = [a for a in snapshot_alphas if a >= 0]

for a in snapshot_alphas:
    def hook(module, input, output, _f=fold, _a=a, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    
    # Get top 20 tokens
    top_probs, top_indices = torch.topk(probs, 20)
    
    print(f"\n  α = {a} (DM = {dm:.4f}):")
    print(f"  {'Rank':>4s}  {'Token':>15s}  {'Prob':>8s}  {'Type':>8s}  {'Bar':>20s}")
    print(f"  {'-'*60}")
    
    digit_mass_in_top20 = 0
    struct_mass_in_top20 = 0
    
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tok.decode([idx.item()]).replace('\n', '\\n')
        is_digit = idx.item() in DIGIT_TOKEN_SET
        token_type = "DIGIT" if is_digit else "struct"
        bar = "█" * int(p.item() * 60)
        print(f"  {rank+1:>4d}  {token_str:>15s}  {p.item():>8.4f}  {token_type:>8s}  {bar}")
        
        if is_digit:
            digit_mass_in_top20 += p.item()
        else:
            struct_mass_in_top20 += p.item()
    
    print(f"  Top-20 digit mass: {digit_mass_in_top20:.4f} | Top-20 structural mass: {struct_mass_in_top20:.4f}")

# ═══════════════════════════════════════════════════════════════
# PART 2: Bimodality quantification across the sweep
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PART 2: Bimodality Metric Across α Sweep")
print(f"{'='*65}")

# Bimodality metric: min(digit_mass, structural_mass_top10) / max(digit_mass, structural_mass_top10)
# = 0 when one mode dominates completely (unimodal)
# = 1 when both modes are equal (maximally bimodal)
# If cusp: this metric should SPIKE at the critical point
# If sigmoid: this metric should rise smoothly and fall smoothly

fine_alphas = list(np.arange(0, 45, 0.5))
bimodality_scores = []
digit_masses = []
struct_top_masses = []

print(f"\n  {'α':>6s}  {'DM':>8s}  {'Struct Top10':>13s}  {'Bimodality':>11s}  {'Visual':>20s}")
print(f"  {'-'*65}")

for a in fine_alphas:
    def hook(module, input, output, _f=fold, _a=a, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    
    # Top-10 structural token mass (non-digit tokens with highest prob)
    all_probs = probs.numpy()
    struct_probs = [(p, i) for i, p in enumerate(all_probs) if i not in DIGIT_TOKEN_SET]
    struct_probs.sort(reverse=True)
    struct_top10 = sum(p for p, i in struct_probs[:10])
    
    # Bimodality: overlap ratio
    bimod = min(dm, struct_top10) / (max(dm, struct_top10) + 1e-12)
    
    digit_masses.append(dm)
    struct_top_masses.append(struct_top10)
    bimodality_scores.append(bimod)
    
    # Visual: show both bars
    d_bar = "D" * int(dm * 30)
    s_bar = "S" * int(struct_top10 * 30)
    
    if a % 2 == 0:  # print every 2 steps for readability
        print(f"  {a:>6.1f}  {dm:>8.4f}  {struct_top10:>13.4f}  {bimod:>11.4f}  {d_bar}{s_bar}")

# Find peak bimodality
peak_idx = np.argmax(bimodality_scores)
peak_alpha = fine_alphas[peak_idx]
peak_score = bimodality_scores[peak_idx]

print(f"\n  Peak bimodality: {peak_score:.4f} at α = {peak_alpha:.1f}")
print(f"  At peak: DM = {digit_masses[peak_idx]:.4f}, Struct Top10 = {struct_top_masses[peak_idx]:.4f}")

# Check: is the bimodality peak sharp (cusp) or broad (sigmoid)?
# Measure width at half-max
half_max = peak_score / 2
above_half = [(a, b) for a, b in zip(fine_alphas, bimodality_scores) if b > half_max]
if above_half:
    width = above_half[-1][0] - above_half[0][0]
    print(f"  Width at half-max: {width:.1f} (narrow = cusp, broad = sigmoid)")

# ═══════════════════════════════════════════════════════════════
# PART 3: Entropy analysis
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PART 3: Entropy at the Critical Point")
print(f"{'='*65}")
print("(Cusp → entropy spike at transition | Sigmoid → smooth entropy change)")

entropies = []
for a in fine_alphas:
    def hook(module, input, output, _f=fold, _a=a, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    probs = torch.softmax(out.logits[0, tpos, :], dim=-1)
    # Shannon entropy
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    entropies.append(entropy)

print(f"\n  {'α':>6s}  {'Entropy':>8s}  {'Bar':>30s}")
print(f"  {'-'*50}")
for i, (a, e) in enumerate(zip(fine_alphas, entropies)):
    if i % 4 == 0:
        bar = "█" * int(e * 5)
        print(f"  {a:>6.1f}  {e:>8.4f}  {bar}")

peak_ent_idx = np.argmax(entropies)
peak_ent_alpha = fine_alphas[peak_ent_idx]
peak_ent = entropies[peak_ent_idx]
baseline_ent = entropies[0]
post_ent = entropies[-1]

print(f"\n  Baseline entropy (α=0): {baseline_ent:.4f}")
print(f"  Peak entropy: {peak_ent:.4f} at α = {peak_ent_alpha:.1f}")
print(f"  Post-transition entropy (α={fine_alphas[-1]}): {post_ent:.4f}")
print(f"  Peak/baseline ratio: {peak_ent/baseline_ent:.2f}×")

# Check if entropy peak is at or near the critical alpha
if abs(peak_ent_alpha - critical_alpha) <= 3:
    print(f"\n  Entropy peaks NEAR critical α ({peak_ent_alpha:.1f} vs {critical_alpha})")
    print(f"  → Consistent with cusp: maximum uncertainty at the bifurcation point")
else:
    print(f"\n  Entropy peak NOT aligned with critical α ({peak_ent_alpha:.1f} vs {critical_alpha})")

# ═══════════════════════════════════════════════════════════════
# PART 4: Multi-prompt bimodality check
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PART 4: Bimodality Across Multiple Prompts")
print(f"{'='*65}")

test_prompts = [
    "The temperature was 98.",
    "She scored 42.",
    "He measured exactly 3.",
    "The price was 7.",
    "The ratio was about 1.",
    "The value of pi is approximately 3.",
]

for p in test_prompts:
    p_ids = torch.tensor([tok.encode(p)])
    p_tpos = p_ids.shape[1] - 1
    
    # Find this prompt's critical alpha
    best_drop = 0
    p_critical = 25
    prev_dm = 1.0
    for a in range(0, 60):
        def hook(module, input, output, _f=fold, _a=a, _t=p_tpos):
            h = output[0].clone()
            h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
            return (h,) + output[1:]
        handle = model.transformer.h[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(p_ids)
        handle.remove()
        probs = torch.softmax(out.logits[0, p_tpos, :], dim=-1)
        dm = sum(probs[t].item() for t in DIGIT_TOKENS)
        drop = prev_dm - dm
        if drop > best_drop:
            best_drop = drop
            p_critical = a
        prev_dm = dm
    
    # Measure bimodality at critical point
    def hook(module, input, output, _f=fold, _a=p_critical, _t=p_tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    handle = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(p_ids)
    handle.remove()
    probs = torch.softmax(out.logits[0, p_tpos, :], dim=-1)
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    
    all_p = probs.numpy()
    struct_p = sorted([(v, i) for i, v in enumerate(all_p) if i not in DIGIT_TOKEN_SET], reverse=True)
    struct_top = sum(v for v, i in struct_p[:10])
    bimod = min(dm, struct_top) / (max(dm, struct_top) + 1e-12)
    
    # Also get entropy
    log_p = torch.log(probs + 1e-12)
    ent = -(probs * log_p).sum().item()
    
    print(f"\n  {p}")
    print(f"    Critical α: {p_critical}, DM at critical: {dm:.4f}")
    print(f"    Digit mass: {dm:.4f}, Struct top-10: {struct_top:.4f}")
    print(f"    Bimodality: {bimod:.4f}, Entropy: {ent:.4f}")
    
    # Show top 5 tokens at critical point
    top5_p, top5_i = torch.topk(probs, 5)
    tokens = [tok.decode([i.item()]).replace('\n', '\\n') for i in top5_i]
    types = ["D" if i.item() in DIGIT_TOKEN_SET else "S" for i in top5_i]
    print(f"    Top 5: {', '.join(f'{t}({ty})={p:.3f}' for t, ty, p in zip(tokens, types, top5_p))}")

elapsed = time.time() - start

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  Peak bimodality score: {peak_score:.4f} at α = {peak_alpha:.1f}")
print(f"  Entropy spike at transition: {peak_ent/baseline_ent:.2f}× baseline")
print(f"  Transition width (half-max): {width:.1f} units of α" if above_half else "  Could not measure width")
print(f"\n  Cusp signature = sharp bimodality peak + entropy spike + narrow transition")
print(f"  Sigmoid signature = broad bimodality + smooth entropy + wide transition")
print(f"\n  Time: {elapsed:.1f}s")
print(f"{'='*65}")
