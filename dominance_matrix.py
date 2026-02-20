"""
DOMINANCE MATRIX — Commitment Authority Gradient
=================================================
For every layer pair (Li, Lj) where i < j, inject OPPOSING fold
directions and measure which layer wins control of the output.

Produces a dominance matrix showing the pecking order of layers.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

print("=" * 60)
print("DOMINANCE MATRIX — Commitment Authority Gradient")
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

# Precompute folds at every layer
print("\nPrecomputing folds at all 12 layers...")
folds = {}
for L in range(12):
    s = np.array([get_hidden(t, L) for t in struct_prompts])
    n = np.array([get_hidden(t, L) for t in numer_prompts])
    fold = np.mean(s, axis=0) - np.mean(n, axis=0)
    folds[L] = fold / np.linalg.norm(fold)

prompt = "The temperature was 98."
ids = torch.tensor([tok.encode(prompt)])
tpos = ids.shape[1] - 1

# Get baseline DM
with torch.no_grad():
    out = model(ids)
baseline_dm = digit_mass(torch.softmax(out.logits[0, tpos, :], dim=-1))
print(f"Baseline DM: {baseline_dm:.4f}")

# The contest: at Li (early) inject +alpha (structural direction = suppress digits)
#              at Lj (late) inject -alpha (numerical direction = preserve digits)
# DM > 0.5 → late layer Lj won (preserved digits despite early suppression)
# DM < 0.5 → early layer Li won (suppressed digits despite late preservation)

alpha = 40  # moderate strength so we see graded effects

print(f"\nContest: Li injects +{alpha} (struct/kill), Lj injects -{alpha} (numer/save)")
print(f"DM < 0.5 → early layer Li wins | DM > 0.5 → late layer Lj wins\n")

# Dominance matrix: dominance[i][j] = DM when Li=+alpha, Lj=-alpha
n_layers = 12
dominance = np.zeros((n_layers, n_layers))

start = time.time()

for Li in range(n_layers):
    for Lj in range(n_layers):
        if Li == Lj:
            dominance[Li][Lj] = 0.5  # tie by definition
            continue
        if Li > Lj:
            continue  # only compute upper triangle (Li < Lj)
            
        handles = []
        
        def make_hook(layer_idx, direction):
            fold = folds[layer_idx]
            def hook(module, input, output):
                h = output[0].clone()
                h[0, tpos, :] += torch.tensor(direction * alpha * fold, dtype=h.dtype)
                return (h,) + output[1:]
            return hook
        
        handles.append(model.transformer.h[Li].register_forward_hook(make_hook(Li, +1)))
        handles.append(model.transformer.h[Lj].register_forward_hook(make_hook(Lj, -1)))
        
        with torch.no_grad():
            out = model(ids)
        for h in handles:
            h.remove()
        
        dm = digit_mass(torch.softmax(out.logits[0, tpos, :], dim=-1))
        dominance[Li][Lj] = dm

elapsed = time.time() - start
print(f"Computed {n_layers}x{n_layers} matrix in {elapsed:.1f}s\n")

# Print the matrix (upper triangle only: i < j means Li is truly earlier)
print("DOMINANCE MATRIX (upper triangle: Li=early +struct, Lj=late -numer)")
print("  DM > 0.5 → late layer Lj won | DM < 0.5 → early layer Li won")
print(f"\n{'':>4s}", end="")
for j in range(n_layers):
    print(f"  L{j:<3d}", end="")
print()
print("     " + "------" * n_layers)

for i in range(n_layers):
    print(f"L{i:<2d} |", end="")
    for j in range(n_layers):
        if i == j:
            print(f"  --- ", end="")
        elif i > j:
            print(f"   .  ", end="")  # lower triangle: roles reversed, skip
        else:
            dm = dominance[i][j]
            # Color coding via symbols
            if dm < 0.3:
                marker = "▼"  # early layer won decisively 
            elif dm < 0.5:
                marker = "↓"  # early layer won slightly
            elif dm > 0.7:
                marker = "▲"  # late layer won decisively
            elif dm > 0.5:
                marker = "↑"  # late layer won slightly
            else:
                marker = "="  # tie
            print(f" {dm:.2f}{marker}", end="")
    print()

# Now compute "authority score" per layer: 
# How often does this layer win when it's the LATE layer (defending)?
print(f"\n{'='*60}")
print("AUTHORITY SCORES")
print(f"{'='*60}")
print("(How often each layer wins when defending as the LATE layer)\n")

for Lj in range(n_layers):
    wins = 0
    contests = 0
    for Li in range(n_layers):
        if Li == Lj:
            continue
        if Li < Lj:
            # Lj is late layer, Lj wins if DM > 0.5
            if dominance[Li][Lj] > 0.5:
                wins += 1
            contests += 1
    win_rate = wins / contests if contests > 0 else 0
    bar = "█" * int(win_rate * 30)
    print(f"  L{Lj:>2d}: {win_rate:.0%} ({wins}/{contests})  {bar}")

# Also: when attacking as early layer
print(f"\n(How often each layer wins when attacking as the EARLY layer)\n")
for Li in range(n_layers):
    wins = 0
    contests = 0
    for Lj in range(n_layers):
        if Li == Lj:
            continue
        if Li < Lj:
            # Li is early, Li wins if DM < 0.5
            if dominance[Li][Lj] < 0.5:
                wins += 1
            contests += 1
    win_rate = wins / contests if contests > 0 else 0
    bar = "█" * int(win_rate * 30)
    print(f"  L{Li:>2d}: {win_rate:.0%} ({wins}/{contests})  {bar}")

# Key finding: is there a crossover point?
print(f"\n{'='*60}")
print("CROSSOVER ANALYSIS")
print(f"{'='*60}")
print("(For adjacent layer pairs: does later layer always win?)\n")
for i in range(n_layers - 1):
    j = i + 1
    dm = dominance[i][j]
    winner = f"L{j} (late)" if dm > 0.5 else f"L{i} (early)"
    print(f"  L{i} vs L{j}: DM={dm:.4f} → {winner} wins")

print(f"\n{'='*60}")
print("Done.")
