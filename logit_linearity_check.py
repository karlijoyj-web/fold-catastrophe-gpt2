"""
LOGIT LINEARITY CHECK
=====================
Does the raw logit difference show a nonlinear kink across alpha,
or is the sharp DM transition purely a softmax artifact?

If logit difference is linear → sharpness is softmax illusion
If logit difference is nonlinear → internal geometric fold confirmed
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

DIGIT_TOKENS = [tok.encode(str(d))[0] for d in range(10)]

# Fold computation
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

fold_l3 = compute_fold(3)

prompt = "The temperature was 98."
ids = torch.tensor([tok.encode(prompt)])
tpos = ids.shape[1] - 1

# Fine-grained alpha sweep through the critical band
alphas = list(range(0, 45, 1))

print(f"{'alpha':>6s}  {'DM':>8s}  {'top_digit_logit':>16s}  {'top_struct_logit':>16s}  {'logit_diff':>11s}")
print("-" * 65)

for a in alphas:
    def hook(module, input, output, _f=fold_l3, _a=a, _t=tpos):
        h = output[0].clone()
        h[0, _t, :] += torch.tensor(_a * _f, dtype=h.dtype)
        return (h,) + output[1:]
    
    handle = model.transformer.h[3].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    
    logits = out.logits[0, tpos, :]
    probs = torch.softmax(logits, dim=-1)
    
    # DM in probability space
    dm = sum(probs[t].item() for t in DIGIT_TOKENS)
    
    # Max logit among digit tokens
    digit_logits = [logits[t].item() for t in DIGIT_TOKENS]
    top_digit_logit = max(digit_logits)
    
    # Max logit among non-digit tokens (structural)
    all_logits = logits.numpy()
    mask = np.ones(len(all_logits), dtype=bool)
    for t in DIGIT_TOKENS:
        mask[t] = False
    top_struct_logit = all_logits[mask].max()
    
    logit_diff = top_digit_logit - top_struct_logit
    
    print(f"{a:>6d}  {dm:>8.4f}  {top_digit_logit:>16.4f}  {top_struct_logit:>16.4f}  {logit_diff:>+11.4f}")

print("\nIf logit_diff changes LINEARLY with alpha → softmax is creating the sharp DM jump")
print("If logit_diff shows a KINK or acceleration → internal nonlinear fold confirmed")
