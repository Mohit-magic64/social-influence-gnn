# Social Influence Prediction with GNNs — Report

## 1. What the task is

Given a social graph, a user `v`, and the binary action states of everyone around `v`,
predict the probability that `v` performs the same action. Ad views, product adoption,
reposts — one shape of problem.

The pipeline follows DeepInf (Qiu et al., KDD 2018): cut a fixed-size ego network around
`v`, encode it with a GNN, concatenate the ego's embedding with handcrafted features,
push it through an MLP, train end-to-end with binary cross-entropy.

The one constraint that governs everything: **`v`'s own action state is the label and
must never enter the input.** The sampler only admits users who are inactive at
observation time, and the model masks the ego's action channel explicitly anyway. Leak
it and you get an AUC near 1.0 and a model that has learned nothing.

## 2. The question worth asking

"Does the GNN beat the baselines" is a weak question, because the answer depends
entirely on how much of the signal is structural in the first place. If adoption is
driven purely by *how many* friends adopted, a logistic regression on that count is the
correct model and a GNN is expensive decoration.

So this project asks a sharper question: **how much genuinely structural signal exists,
and what fraction of it does message passing recover?**

Answering that requires knowing the ground truth mechanism, which is why the default
dataset is synthetic. The cascade is a generalised-threshold process where adoption
probability depends on three things:

- the **count** of active neighbours
- the **fraction** of neighbours who are active
- the number of **disconnected groups** those adopters form

The third is structural diversity, and it is a real empirical finding: Ugander et al.
(PNAS 2012) showed three adopters from three separate social circles are substantially
more persuasive than three adopters who all know each other. It is also, deliberately,
absent from the handcrafted feature list the problem statement specifies. Recovering it
requires looking at the wiring *between* a user's neighbours.

That gives a clean instrument: a **diagnostic oracle** — logistic regression handed the
exact generative feature. It bounds what any model could gain from structure, and turns
"is the GNN better" into "what share of the available headroom did it capture".

## 3. Results

Three seeds. Each seed regenerates the graph, the cascade and the split, so the spread
covers data variance as well as initialisation variance. 12,000 instances per seed
(7.2k train / 1.8k val / 3.0k test), 8,000-node graph, ego networks of 50 nodes.

### 3.1 Regime A — default cascade

| Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6693 ± 0.0071 | 0.5354 ± 0.0032 | 0.5597 |
| DeepWalk + MLP | 0.5390 ± 0.0105 | 0.3979 ± 0.0117 | 0.5313 |
| node2vec + MLP | 0.5456 ± 0.0122 | 0.4024 ± 0.0099 | 0.5341 |
| Plain GCN (no fusion) | 0.6234 ± 0.0105 | 0.4895 ± 0.0099 | 0.5386 |
| Plain GAT (no fusion) | 0.6599 ± 0.0122 | 0.5257 ± 0.0059 | 0.5556 |
| Full model: GCN + handcrafted | 0.6703 ± 0.0112 | 0.5374 ± 0.0013 | 0.5579 |
| **Full model: GAT + handcrafted (proposed)** | **0.6690 ± 0.0079** | **0.5360 ± 0.0033** | **0.5545** |
| *[diagnostic] LR + 2-hop active count* | *0.6692 ± 0.0074* | *0.5351 ± 0.0029* | *0.5603* |
| *[diagnostic] LR + true structural diversity* | *0.6793 ± 0.0032* | *0.5521 ± 0.0058* | *0.5638* |

### 3.2 Regime B — structure-dominant cascade

Same code, same pipeline. Only the cascade coefficients change: the structural term
carries most of the signal and the raw active-neighbour count carries almost none.

| Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6911 ± 0.0150 | 0.5895 ± 0.0052 | 0.5966 |
| Plain GCN (no fusion) | 0.6504 ± 0.0177 | 0.5482 ± 0.0025 | 0.5740 |
| Plain GAT (no fusion) | 0.6856 ± 0.0151 | 0.5823 ± 0.0054 | 0.5943 |
| Full model: GCN + handcrafted | 0.6929 ± 0.0116 | 0.5919 ± 0.0021 | 0.6014 |
| **Full model: GAT + handcrafted (proposed)** | **0.6967 ± 0.0145** | **0.5954 ± 0.0041** | **0.6071** |
| *[diagnostic] LR + true structural diversity* | *0.7230 ± 0.0112* | *0.6278 ± 0.0020* | *0.6199* |

### 3.3 The comparison that actually resolves it

Seed-to-seed spread is roughly ±0.015 AUC. The model differences are around ±0.005. So
comparing marginal means answers nothing — the error bars swallow the effect. But every
model shares its seed with the baseline it is compared against, so the comparison can be
**paired**, which removes the shared data variance entirely.

| Regime | Metric | per-seed delta (GNN − LR) | mean | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|
| A | AUC-ROC | −0.0012, +0.0005, −0.0004 | −0.0004 ± 0.0009 | 1/3 | +0.0100 | −4% |
| A | AUC-PR | +0.0007, +0.0004, +0.0006 | +0.0006 ± 0.0002 | 3/3 | +0.0166 | 4% |
| B | AUC-ROC | +0.0046, +0.0053, +0.0070 | **+0.0056 ± 0.0012** | **3/3** | +0.0319 | **18%** |
| B | AUC-PR | +0.0055, +0.0044, +0.0078 | **+0.0059 ± 0.0017** | **3/3** | +0.0383 | **15%** |

Three seeds is far too few for a p-value to carry weight. The evidence here is the sign
pattern and the tightness of the spread, not a significance test.

## 4. What this means

**The GNN's value is proportional to how structural the underlying mechanism is.** In
regime A it ties a well-tuned logistic regression. In regime B it wins consistently on
every seed. Same architecture, same hyperparameters, same code — only the world changed.

**Even when it wins, it recovers under a fifth of what was available.** The oracle proves
+0.032 AUC of structural signal exists in regime B; the model captures +0.0056.

The reason is not tuning. Counting connected components among active neighbours is a
*global* property of the induced subgraph, and a 2-layer message-passing network
provably cannot compute it — this is the 1-Weisfeiler-Lehman expressiveness ceiling
(Xu et al., 2019). A GNN can learn that active neighbours are mutually connected
(a local redundancy signal correlated with diversity), but not the component count
itself. The gap between +0.0056 and +0.032 is largely that.

**Unsupervised embeddings alone are near useless here**, at 0.54 AUC. Graph position
without action state says almost nothing about adoption, which is the expected result
and worth stating plainly.

## 5. Findings from building it

**The wide-and-deep skip was necessary, and I found that by measurement.** In the first
working version, concatenating 10 clean handcrafted features to a 64-dim noisy GNN
embedding and pushing the result through a dropout MLP produced a fused model that
scored **below** plain logistic regression (0.6532 vs 0.6616). The clean features were
being diluted. Adding a direct linear path from the handcrafted block to the output
logit (Cheng et al., 2016) makes the architecture literally "logistic regression plus a
learned graph correction" — the baseline becomes the floor rather than something the
optimiser has to rediscover.

**node2vec embeddings as an extra input channel actively hurt.** Near chance on their
own, but 64 dims × 50 nodes of memorisation capacity. Ablation: 0.6690 → 0.6564.
They were cut from the proposed model and kept as an ablation row.

**I removed a feature from my own baseline, and report it anyway.** An early version gave
logistic regression a 2-hop active-neighbour count. It is not in the problem statement's
feature list and it is a hand-built proxy for exactly what the GNN is supposed to learn,
so handing it over free answers a different question. It is reported separately as a
strengthened baseline so the choice is visible rather than quietly favourable — and it
made no difference at all (0.6692 vs 0.6693).

**An evaluation bug worth recording.** Instances are cut at three different observation
rounds. The oracle baseline was scoring all of them against a single shared cascade
snapshot, silently corrupting two-thirds of the rows and making structural diversity
look worthless. Fixed by tracking the observation round per instance.

**Ablations** (regime A, seed 0, full GAT at 0.6690):

| Ablation | AUC-ROC | Δ |
|---|---|---|
| − handcrafted features entirely | 0.6437 | −0.025 |
| 3 GNN layers instead of 2 | 0.6514 | −0.018 |
| + node2vec structural embeddings | 0.6564 | −0.013 |
| − instance normalisation | 0.6592 | −0.010 |
| − wide/deep skip | 0.6593 | −0.010 |
| 1 GNN layer instead of 2 | 0.6597 | −0.009 |

Handcrafted features are the single largest contributor. Three layers is worse than two
— over-smoothing, visible directly in the numbers.

**Ego network radius barely matters**: k=1 → 0.6586, k=2 → 0.6554, k=3 → 0.6624. Flat
within noise, consistent with the model not exploiting deep structure.

## 6. Limitations

- **Synthetic data.** Necessary for the oracle, but the generative process is one I chose.
  The honest framing is that regime A and regime B bracket a range, not that either is
  what a real platform looks like.
- **One graph family.** Holme–Kim powerlaw-cluster. No test on bipartite, dense, or
  strongly community-structured graphs.
- **Three seeds.** Enough to see a consistent sign, not enough for a significance claim.
- **Sampling bias.** Top-k by personalised PageRank keeps structurally close nodes, which
  may systematically drop exactly the weak ties Granovetter argued carry the most novel
  influence. The k-hop sweep probes this only indirectly.
- **No causal identification.** This predicts influence; it does not demonstrate it.
  Homophily and influence produce identical correlations in observational data — you
  adopt what your friends adopted either because they persuaded you or because you were
  always similar people. Separating them needs a shuffle test (Anagnostopoulos et al.,
  2008) or an instrument.

## 7. References

- Qiu, J. et al. (2018). *DeepInf: Social Influence Prediction with Deep Learning.* KDD.
- Perozzi, B. et al. (2014). *DeepWalk: Online Learning of Social Representations.* KDD.
- Grover, A. & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks.* KDD.
- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Veličković, P. et al. (2018). *Graph Attention Networks.* ICLR.
- Ugander, J. et al. (2012). *Structural diversity in social contagion.* PNAS.
- Xu, K. et al. (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Cheng, H.-T. et al. (2016). *Wide & Deep Learning for Recommender Systems.* DLRS.
- Anagnostopoulos, A. et al. (2008). *Influence and correlation in social networks.* KDD.
