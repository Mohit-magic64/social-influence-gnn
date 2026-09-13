# Social Influence Prediction with GNNs

## 1. The task

Given a social graph, a user `v`, and the action states of everyone around them, predict whether `v` does the same thing. Ad clicks, product adoption, reposts. One shape of problem.

The pipeline follows DeepInf (Qiu et al., KDD 2018). Cut a fixed-size ego network around `v`, encode it with a GNN, concatenate the ego's embedding with handcrafted features, push that through an MLP, train end to end with binary cross-entropy.

One constraint governs everything else. The ego's own action state is the label, so it can never enter the input. The sampler only admits users who are still inactive at observation time, and the model zeroes the ego's action channel anyway. Leak it and you get an AUC near 1.0 from a model that learned nothing.

## 2. The question I actually asked

"Does the GNN beat the baselines" is a weak question. The answer depends entirely on how much of the signal is structural in the first place.

If adoption is driven by *how many* friends adopted, then a logistic regression on that count is the right model and a GNN is expensive decoration. If adoption depends on how those friends are wired to each other, the GNN has something to do.

So I asked a sharper version. How much structural signal is in the data, and what share of it does message passing recover?

That needs a known ground truth, which is why the default dataset is synthetic. The cascade is a generalised-threshold process. Adoption probability depends on three things:

- the **count** of active neighbours
- the **fraction** of neighbours who are active
- the number of **disconnected groups** those adopters form

The third one is structural diversity. Ugander et al. (PNAS 2012) found that three adopters from three separate social circles are far more persuasive than three adopters who all know each other. It is also, on purpose, missing from the handcrafted feature list the problem statement specifies. You can only get at it by looking at the wiring between a user's neighbours.

That gave me a clean instrument. A **diagnostic oracle**: logistic regression handed the exact generative feature. It bounds what any model could possibly gain from structure. The question stops being "is the GNN better" and becomes "what share of the available headroom did it capture".

## 3. Results

Three seeds. Each seed regenerates the graph, the cascade and the split, so the spread covers data variance and not just initialisation variance. 12,000 instances per seed (7.2k train, 1.8k val, 3.0k test) on an 8,000-node graph, ego networks of 50 nodes.

### 3.1 Regime A, default cascade

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

### 3.2 Regime B, structure-dominant cascade

Same code, same pipeline, same hyperparameters. Only the cascade coefficients move. The structural term now carries most of the signal and the raw count carries almost none.

| Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6911 ± 0.0150 | 0.5895 ± 0.0052 | 0.5966 |
| Plain GCN (no fusion) | 0.6504 ± 0.0177 | 0.5482 ± 0.0025 | 0.5740 |
| Plain GAT (no fusion) | 0.6856 ± 0.0151 | 0.5823 ± 0.0054 | 0.5943 |
| Full model: GCN + handcrafted | 0.6929 ± 0.0116 | 0.5919 ± 0.0021 | 0.6014 |
| **Full model: GAT + handcrafted (proposed)** | **0.6967 ± 0.0145** | **0.5954 ± 0.0041** | **0.6071** |
| *[diagnostic] LR + true structural diversity* | *0.7230 ± 0.0112* | *0.6278 ± 0.0020* | *0.6199* |

### 3.3 The comparison that settles it

Look at the error bars above. Seed spread is about ±0.015 AUC. The model differences are around ±0.005. Comparing those marginal means answers nothing, because the noise is three times the effect.

But every model shares its seed with the baseline it is being compared against. So the comparison can be paired, and pairing cancels the shared data variance.

| Regime | Metric | per-seed delta (GNN − LR) | mean | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|
| A | AUC-ROC | −0.0012, +0.0005, −0.0004 | −0.0004 ± 0.0009 | 1/3 | +0.0100 | −4% |
| A | AUC-PR | +0.0007, +0.0004, +0.0006 | +0.0006 ± 0.0002 | 3/3 | +0.0166 | 4% |
| B | AUC-ROC | +0.0046, +0.0053, +0.0070 | **+0.0056 ± 0.0012** | **3/3** | +0.0319 | **18%** |
| B | AUC-PR | +0.0055, +0.0044, +0.0078 | **+0.0059 ± 0.0017** | **3/3** | +0.0383 | **15%** |

Three seeds is nowhere near enough for a p-value to carry weight, and I am not claiming one. The evidence here is the sign pattern and how tight the spread is.

## 4. What it means

The GNN's value tracks how structural the mechanism is. In regime A it ties a well-tuned logistic regression. In regime B it wins on every seed. The architecture never changed. The world did.

And even when it wins, it picks up under a fifth of what was on the table. The oracle says +0.032 AUC of structural signal exists in regime B. The model gets +0.0056.

That gap is not a tuning failure. Counting connected components among active neighbours is a global property of the induced subgraph, and a 2-layer message-passing network provably cannot compute it. This is the 1-Weisfeiler-Lehman expressiveness ceiling (Xu et al., 2019). A GNN can learn that a user's active neighbours are mutually connected, which is a local redundancy signal that correlates with diversity. It cannot learn the component count itself. Most of the missing 82% is that.

The unsupervised embeddings are close to useless here at 0.54 AUC. Graph position with no action state says almost nothing about whether someone adopts. That is the expected result and worth saying plainly rather than burying.

## 5. What I learned building it

**The wide-and-deep skip was necessary, and I found that by measuring, not by planning.**

The first working version concatenated 10 clean handcrafted features onto a 64-dim noisy GNN embedding and pushed the result through a dropout MLP. The fused model scored *below* plain logistic regression: 0.6532 against 0.6616. The clean features were being drowned.

Adding a direct linear path from the handcrafted block to the output logit (Cheng et al., 2016) fixed it. The architecture becomes logistic regression plus a learned graph correction, so the baseline is the floor instead of something the optimiser has to rediscover from scratch.

**node2vec embeddings as an extra input channel made things worse.**

They score near chance on their own, but they hand the model 64 dims × 50 nodes of memorisation capacity. Ablation: 0.6690 down to 0.6564. I cut them from the proposed model and kept them as an ablation row rather than quietly deleting them.

**I removed a feature from my own baseline, and I report it anyway.**

An early version gave logistic regression a 2-hop active-neighbour count. That feature is not in the problem statement's list, and it is a hand-built proxy for exactly what the GNN is supposed to learn. Handing it over for free answers a different question than the one I set up.

So I pulled it, and reported it separately as a strengthened baseline so the choice stays visible. It made no difference at all: 0.6692 against 0.6693.

**I found a bug in my own evaluation.**

Instances get cut at three different observation rounds. The oracle baseline was scoring all of them against a single shared cascade snapshot, which silently corrupted two thirds of the rows and made structural diversity look worthless. Fixed by tracking the observation round per instance.

**Ablations** (regime A, seed 0, full GAT at 0.6690):

| Ablation | AUC-ROC | Δ |
|---|---|---|
| − handcrafted features entirely | 0.6437 | −0.025 |
| 3 GNN layers instead of 2 | 0.6514 | −0.018 |
| + node2vec structural embeddings | 0.6564 | −0.013 |
| − instance normalisation | 0.6592 | −0.010 |
| − wide/deep skip | 0.6593 | −0.010 |
| 1 GNN layer instead of 2 | 0.6597 | −0.009 |

Handcrafted features are the biggest single contributor by a wide margin. Three layers is worse than two, which is over-smoothing showing up directly in the numbers.

**Ego network radius barely matters.** k=1 gives 0.6586, k=2 gives 0.6554, k=3 gives 0.6624. Flat within noise, which fits a model that is not exploiting deep structure in the first place.

## 6. Limitations

**Synthetic data.** The oracle requires it, but the generative process is one I picked. Regime A and regime B bracket a range. Neither is a claim about what a real platform looks like.

**One graph family.** Holme-Kim powerlaw-cluster. Nothing tested on bipartite, dense, or strongly community-structured graphs.

**Three seeds.** Enough to see a consistent sign. Not enough to make a significance claim.

**Sampling bias.** Top-k by personalised PageRank keeps structurally close nodes, so it may systematically drop the weak ties Granovetter argued carry the most novel influence. The k-hop sweep only probes this indirectly.

**No causal identification.** This predicts influence, it does not demonstrate it. Homophily and influence produce identical correlations in observational data: you adopt what your friends adopted either because they persuaded you, or because you were always similar people and would have adopted anyway. Separating the two needs the shuffle test from Anagnostopoulos et al. (2008), or an instrument, or an experiment.

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
