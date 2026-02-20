# I Found Catastrophe Geometry in GPT-2's Residual Stream

**tl;dr:** When GPT-2 encounters an ambiguous token — a period that could be a decimal or a sentence boundary — it resolves the ambiguity by crossing a *fold*: a low-dimensional decision boundary with the geometric properties predicted by catastrophe theory. The transition is sharp, directional, asymmetric, and context-dependent. I built this in a week using AI assistants as collaborators. Everything reproduces from a single script in 29 seconds on CPU. [GitHub](https://github.com/karlijoyj-web/fold-catastrophe-gpt2)

---

## The Setup

I have no ML background. About a week ago I started poking at GPT-2's internals with the help of Claude, ChatGPT, and Gemini — using them as coding partners, math checkers, and sounding boards. What I found was surprising enough that I wanted to put it in front of people who know this field better than I do.

The problem is simple. Consider the period in "The temperature was 98."

GPT-2 has to decide: is the next token a digit (making this 98.6, a decimal) or a word/newline (making this a sentence ending)? The model puts about 88% probability on digits — it's strongly committed to the decimal reading.

Where does that commitment happen? How? Can you reverse it? And does the *shape* of that commitment tell you something fundamental about how the model works?

## What I Did

The technique is straightforward. Compute a "fold direction" — the average difference in hidden states between structural contexts (period ends a sentence) and numerical contexts (period is a decimal point), using 4 calibration examples of each (8 total) that are never used in testing. This is computed in the same way as a steering vector (Turner et al., 2023; Zou et al., 2023). The contribution here isn't the direction — it's the geometry of the transition it induces. This gives you a single direction in the 768-dimensional residual stream. Then inject perturbations along that direction at different layers and magnitudes, and measure what happens to next-token predictions.

I use a metric called digit mass (DM): the total probability the model assigns to bare digit tokens (0–9) at the position right after the period. High DM means the model thinks it's a decimal. Low DM means the model thinks the sentence is over. (Implementation note: only bare single-character digit tokens are summed — space-prefixed digits like " 6" represent a different tokenization path, not a decimal continuation. Multi-digit tokens are excluded, which makes DM a conservative measure.)

What I found has a specific geometric structure.

## Part 1: The Fold Exists

**It's sharp.** At layer 3, injection magnitude α=20 (where α is the scalar multiplier on the fold-direction vector) barely moves digit mass (0.883 → 0.878). At α=30 it crashes to 0.001. Not a graded decline — a discontinuous jump. There is a narrow critical band where the system tips. This isn't a softmax illusion: the raw logit difference between the top digit and top structural token is nearly flat from α=0–19 (~0.01/step), then accelerates 70× through the critical band (α=20–25, measured as Δ(logit gap)/Δα between the top digit and top structural token), then decelerates again. The nonlinearity is in the network's internal representations, not just the output probabilities.

**It's directional.** I tested 100 random directions in the 768-dimensional space at the same magnitude. Zero produce a phase transition. 0/100. In 768 dimensions, random vectors are nearly orthogonal to any given direction by construction, so this is a necessary sanity check rather than a surprise — but it confirms the transition is directional rather than a generic response to perturbation magnitude.

**It's one-dimensional.** A near-norm-preserving projection swap — replacing only the single scalar projection of the hidden state onto the fold direction, while holding everything else constant (norm change ~0.1%) — takes digit mass from 0.883 to 0.002 at layer 5. One number in a 768-dimensional vector effectively determines the commitment.

These three properties together — sharp transition, directional specificity, low-dimensional boundary — are the defining features of a fold catastrophe. This isn't an analogy I'm imposing. Catastrophe theory (Thom, 1972) is a classification theorem: under standard genericity assumptions, it predicts that any smooth system with one control parameter and one state variable that undergoes a sudden qualitative change will have this geometry. As an intuition, you can think of layer depth as time and activation norm as energy; the 12-layer pass is discrete and piecewise-smooth rather than continuous, so the classical theorem doesn't apply exactly — but the classification constrains the *topology* of any transition with these qualitative features, and the question is whether the stronger predictions also hold.

## Part 2: Cusp Geometry

The cusp catastrophe is the next step — a fold with two control parameters instead of one. It makes several testable predictions. In this mapping, injection magnitude α is one control parameter, and prompt context supplies the second by shifting where the system sits on the decision surface — how deep in a basin the model starts before you push.

### Context-dependent threshold

How much force is needed to flip the model depends on how strongly the context commits to one interpretation. I ran binary search for the flip threshold across 6 test prompts at layer 3:

| Prompt | Baseline DM | Flip threshold (α) |
|--------|------------|-------------------|
| "She scored 42." | 0.782 | 13 |
| "The value of pi is approximately 3." | 0.414 | 24 |
| "The temperature was 98." | 0.883 | 26 |
| "He measured exactly 3." | 0.682 | 28 |
| "The ratio was about 1." | 0.592 | 29 |
| "The price was 7." | 0.643 | 30 |

The threshold varies 2.3× across prompts. This is the cusp prediction: context determines how deep you start in the basin, which determines how much force is needed to escape. Note that the relationship between baseline DM and threshold isn't simply monotonic — "She scored 42." has high DM (0.782) but the lowest threshold (13), while "The price was 7." has lower DM (0.643) but a much higher threshold (30). This is because baseline DM is only a measurable proxy for where the prompt sits on the contextual asymmetry axis, not the axis itself — context complexity, lexical priors, and syntactic expectations all contribute to the second control parameter. The cusp surface has two axes, not one.

### Asymmetric basin depth

This is where it gets interesting. I measured the force needed to flip in both directions, across 4 prompts each, using binary search:

**CREATE** (flip numerical → structural):
Mean threshold at L3: α = 24 (range: 19–32)

**DESTROY** (flip structural → numerical):
Mean threshold at L3: α = 100 (range: 79–118)

**Ratio: 4.1:1.** Zero overlap between the ranges. The structural basin — the sentence-ending, narrative-continuing interpretation — is dramatically deeper than the numerical basin. Think of it as the difference between a vacuum state and an excited state: structural/narrative is the network's deep default attractor, and numerical reasoning is a shallower, contextually activated mode that's easy to knock the system out of. This isn't a quirk of one prompt — it holds for every prompt I tested.

It goes further. At matched magnitude (α=±40), pushing a numerical prompt toward structural drops DM from 0.88 to 0.01. Pushing the *same prompt* toward numerical moves it from 0.88 to 0.90 (softmax compression contributes to the apparent rigidity at high DM, but the asymmetry is confirmed by the threshold measurements: α=24 vs α=100). The system is highly rigid against being pushed deeper into its current basin, but easily flipped out of it. The energy landscape is fundamentally asymmetric.

### Basin depth topography

I measured the minimum flip force at every layer, creating a layer-by-layer map of how deep the attractor basin is:

**CREATE direction** ("The temperature was 98."):

| Layer | Flip α |
|-------|--------|
| L0 | 51 |
| L1 | 42 |
| L2 | 27 |
| L3 | **25** (shallowest — the critical point) |
| L4 | 32 |
| L5 | 33 |
| L11 | 101 |

*(Intermediate layers omitted; the basin generally deepens through L11 with minor local variation.)*

*(Note: the threshold here is α needed to push DM below 0.5, a slightly looser criterion than the 50% reduction used in the context-dependent threshold table above — for a baseline of 0.883, a 50% reduction requires reaching ~0.44, while DM < 0.5 requires only reaching 0.50 — hence α=25 here vs α=26 there for the same prompt.)*

The basin is shallowest at L2–L3, with L3 as the strict minimum. That's the commitment window — the critical layers where the fold can be crossed with minimum force. L2 does the heavy computational lifting (as we'll see below), but this leaves L3 inheriting a state already teetering on the boundary — so it requires the least additional force to tip. After L3, the basin generally deepens through L11 (with minor local variation). The model doesn't just *decide* at L3 and coast. Downstream layers actively construct the attractor that makes the commitment irreversible.

I verified this with a damping experiment: scaling down the residual contribution of post-commitment layers (L5–L11). At normal operation, fold projection is +7.4. At 50% damping: +3.2. At 25% damping: **−2.7** — the sign flips. The downstream layers don't passively propagate the decision. They *build* the basin. Remove them and the commitment literally reverses.

Zooming in further: which components within the commitment window actually cross the fold? I injected the fold direction into each of the 12 individual attention heads at layer 2, isolating their contributions before the output projection mixes them. Across 6 test prompts, **attention head 1 dominates in 5 of 6** — consistently producing the largest effect. On the remaining prompt (the weakest baseline, sitting near the separatrix), no single head has much leverage. The MLP then acts as an amplifier: head 1 alone moves digit mass partway toward the fold boundary, and the layer 2 MLP drives it the rest of the way — taking DM below 0.01 on the strongest prompts. By layer 3, attention alone is sufficient to complete the flip; by layer 4, the MLP actively resists perturbation, already reinforcing the basin walls. The attractor engine is a two-stage system: L2.H1 initiates the fold crossing, L2.MLP amplifies it past the point of no return.

## Part 3: Cross-Domain Independence

Is there just one fold, or can the model support several independent ones?

I tested whether the punctuation fold interferes with a lexical fold (bat-as-animal vs. bat-as-sports-equipment). The cosine similarity between the two fold directions is −0.077 — essentially orthogonal. Injecting along the punctuation fold at the magnitude that completely flips period behavior (α=30) changes bat disambiguation by less than 0.001. Basically zero.

A 768-dimensional space can support many independent folds, each resolving a different type of ambiguity along its own axis.

## Part 4: The Dominance Matrix

If different layers have different authority over the fold, what does the pecking order look like?

For every pair of layers (Li, Lj) where i < j, I inject *opposing* fold directions — the early layer pushes structural, the late layer pushes numerical — at matched magnitude (α=40) and measure who wins. This produces an upper-triangular dominance matrix (66 contests).

Later layers almost always win. All 11 adjacent-layer contests are won by the later layer — the authority gradient is monotonic among neighbors.

But there's an exception: **L2 overrides L8** despite being 6 layers earlier (DM=0.049 — early layer wins decisively). This is the commitment window asserting itself. Once the system crosses the fold at L2, later layers merely reinforce the basin — they lack the geometric leverage to pull the system back out. This argues against the "closer to output = more authority" explanation. It's computational authority based on where you sit relative to the fold, not relative to the output.

## Part 5: Split-Brain Confabulation

This experiment is the most vivid demonstration of what the fold does.

Setup: inject opposing perturbations at two different layers simultaneously. At layer 3, amplify the numerical signal (push toward "this is math"). At layer 8, override with structural formatting (push toward "continue with prose"). This creates a split-brain condition: the early layers encode numerical meaning more strongly, but the late-layer formatting machinery hijacks the output into prose anyway.

I ran 20 different numerical prompts through this. 19 out of 20 produce confabulation. The model generates fluent prose continuations, but the semantic content is wrong or incoherent. The late layers "know" the output should be words — the format is correct — but the early layers' numerical knowledge gets overridden by the structural frame. Format wins over fact.

Digit mass at the period position drops below 0.1 for nearly all prompts under split-brain injection. Format and fact are physically dissociable. Semantic content and syntactic format are separable commitments carried by different layers, and you can selectively amplify one while overriding it with the other. This provides a geometric mechanism for hallucination: the model's latent knowledge can be correct while its output routing sends it into the wrong attractor basin.

## Part 6: Inception Asymmetry

Last experiment. Take two prompts from opposite basins — a fairy tale ("The beautiful princess walked into the grand") and a math sentence — and try to push each across the fold at layer 4 with increasing force.

**DESTROY direction (fairy tale → numerical): fails.** Even at α=100, digit mass only reaches 0.010. You cannot push the fairy tale into the numerical basin. The structural attractor is too deep. (For comparison, the DESTROY experiments in Part 2 used plain structural sentences like "It was over," which flip at α=79–118. The fairy tale is a much deeper structural context — richly narrative, strong lexical priors — which is why even α=100 barely registers. Basin depth is context-dependent, and this context is about as deep as it gets.)

**CREATE direction (math → structural): succeeds easily.** Digit mass drops below 0.5 just past α=30 and hits 0.001 by α=50.

Same fold direction, same technique, same magnitudes — and one direction is crossable while the other isn't. This independently confirms the basin asymmetry from a completely different experimental design.

## Cross-Architecture Replication

This isn't a GPT-2 artifact. The core findings replicate on Pythia-160M (EleutherAI) — different model family (GPT-NeoX-based), different training data, different tokenizer. I compute the fold direction separately within each model's residual stream. Same commitment window (shifted to L3–L4), same directional specificity, same basin asymmetry. Two models trained independently on different data converge on the same geometric solution. The one-layer shift has a structural explanation: GPT-2 processes attention and MLP serially within each layer, so L2's MLP can immediately amplify what L2's attention found. Pythia processes them in parallel, so L2's MLP can't see L2's attention output — the amplification has to wait until L3.

GPT-2 Medium (24 layers, 1024 dimensions) shows the fold gets *harder* to flip — but the power-law scaling exponent — how sharply DM collapses as α crosses the critical band — actually increases (δ≈0.57 vs δ≈0.45 for Small). Notably, these two values bracket δ=0.5, which is the classical mean-field critical exponent predicted by Landau theory for a fold transition (rigorous confirmation would require fitting in pre-softmax logit space to rule out sigmoid inflection artifacts, but the correspondence is striking). The origami must fold tighter in smaller dimensions. The framework predicts this: a constrained catastrophe in a crowded superposition space should exhibit sharper creases, and those constraints relax as dimensionality increases.

## Why This Might Matter

If the fold picture is right, it gives mechanistic interpretability a geometric vocabulary for decision-making.

**Steering vectors are fold-crossings.** They work because they push representations across a low-dimensional boundary. They're hard to undo because downstream layers deepen the basin. And they need precise aim because only the fold direction triggers the transition — random directions in the same space do nothing.

**Superposition is the sheet; folds are the creases.** Recent work showed that neural networks pack more features than dimensions into their representations (Elhage et al., 2022). That's the sheet — a high-dimensional surface compressed into a lower-dimensional space. The fold framework says this sheet isn't flat. It has sharp creases where the model commits to interpretations. The geometry of those creases — where they are, how deep they go, which direction they run — might be as important as the features projected onto the sheet.

**Small models as fruit flies.** These are small models. I used them because mapping the exact topographic depth of an attractor basin across 12 layers requires exhaustive sweeps — hundreds of forward passes per experiment — that would be expensive at 70B parameters. Consider this the proof of concept and the blueprint. Someone with more compute should run it on bigger models.

## Limitations

**Narrow empirical base.** Punctuation boundaries and one case of lexical polysemy in small transformers. The framework predicts folds for all ambiguity types. I haven't tested most of them. Additionally, the digit mass metric counts only single-character digit tokens (0–9); multi-digit tokens like "14" are excluded, which means DM slightly underestimates the model's numerical commitment on some prompts. This makes the measured thresholds conservative — the true fold is at least as sharp as reported.

**Large injection magnitudes.** α=30 is a meaningful perturbation. The near-norm-preserving swap (behavioral flip with ~0.1% norm change) is the strongest counter to the "you're just overwhelming the residual stream" objection. The 0/100 random direction result is the other counter. But this deserves more investigation. The sharp probability transition is not a softmax artifact — the raw logit difference also shows a 70× acceleration through the critical band (see Part 1) — and the dominance matrix (L2 overriding L8, impossible in a linear pass-through) and damping sign-flip (downstream layers actively constructing the basin) independently confirm internal nonlinear dynamics.

**Polysemy is inconclusive.** The punctuation fold is clean. Lexical polysemy (bat, bank) doesn't show the same crisp geometry. One possible explanation: punctuation controls syntactic routing — get it wrong and the parse tree collapses — which demands a hard discrete commitment. Lexical meaning can persist in superposition without breaking anything downstream. If so, folds should specifically govern syntactic and structural decisions, not semantic ones. This is a testable prediction, not a failure of the framework.

**Two transformer families.** GPT-2 and Pythia are both 12-layer autoregressive transformers. True architectural diversity (state-space models, encoders, mixture-of-experts) remains untested.

## Reproduce It

Everything is in the repo: [GitHub](https://github.com/karlijoyj-web/fold-catastrophe-gpt2)

```
pip install torch transformers numpy
python reproduce_all.py
```

36 automated checks. 29 seconds on CPU. No GPU needed.

The repo also includes standalone experiments for the dominance matrix, confabulation, inception dose-response, and attention head isolation, plus the full verified results document with every number in this post.

---

If you can extend this to larger models or different architectures, I'd love to see it. I'm reachable at karlijoyj@gmail.com.

*I'm not a researcher. I'm an autistic person with ADHD who fell in love with Nima Arkani-Hamed's geometric physics and the mathematics of origami, and wanted to see if the same structures exist inside neural networks. I have no ML background, no institutional affiliation, and no funding. This project was built in about a week with significant help from AI assistants (Claude, ChatGPT, and Gemini) as coding partners, math checkers, and collaborators. Julian Gough's blowtorch conjecture gave me the other half of the intuition. If this jumps out at you the way it jumped out at me, and you think you can help push it further — please reach out. I think this could be important.*

---

## References

- Elhage, N., Hume, T., Olsson, C., et al. (2022). [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html). Transformer Circuits Thread.
- Thom, R. (1972). *Stabilité structurelle et morphogénèse*. W.A. Benjamin.
- Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248). arXiv:2308.10248.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A., Goel, S., Li, N., Byun, Z., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J.Z., & Hendrycks, D. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). arXiv:2310.01405.
