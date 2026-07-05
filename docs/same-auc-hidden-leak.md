# Same AUC, hidden leak

**One run can't tell a clean pipeline from a leaky one.**

The most dangerous bug in applied machine learning doesn't crash. It prints a
respectable number, passes review, and ships. This page reproduces that bug
twice — and then shows the only referee that reliably catches it. Every number
below comes from an executed notebook in this repository; nothing is
illustrative.

*(A 10-minute read — or run it yourself in 5 with the
[Colab quickstart](https://colab.research.google.com/github/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/00-leakage-in-5-minutes.ipynb).)*

## Two protocols, three stateful steps

The IBM Telco churn sample: 7,043 rows, 21 columns. Preprocessing for a
logistic-regression baseline takes three steps, and each one *learns numbers*
from whatever data you feed it:

```python
TELCO_STEPS = [
    {"fn": "handle_missing", "strategy": {"TotalCharges": "median"}},   # learns a median
    {"fn": "encode", "cols": CAT, "method": "onehot", "inplace": True}, # learns vocabularies
    {"fn": "scale", "cols": NUM, "method": "standard", "inplace": True},# learns mean/std
]
```

Protocol A — the one most tutorial code quietly uses — fits those statistics
on **all** rows, then splits into train and test:

```python
fe_all = dx.featpipe(data, steps=TELCO_STEPS, show=False, plot=False)
auc_wrong = test_auc(fe_all.loc[train_idx], fe_all.loc[test_idx])
```

Protocol B splits **first**, fits on train only, and replays the frozen
parameters on the held-out rows:

```python
train_fe, params = dx.featpipe(data.loc[train_idx], steps=TELCO_STEPS,
                               return_params=True)
test_fe = dx.featpipe(data.loc[test_idx], params=params, show=False, plot=False)
```

Same model, same split, same measuring device. The verdict:

```text
leaky protocol : 0.8465
clean protocol : 0.8465
difference     : +0.0000
```

Identical to four decimal places. If you stopped here — and a single run is
exactly where most projects stop — you would ship the leak with a clear
conscience.

## A single split lies in both directions

Second dataset, harder conditions: an 800-row stratified sample of the KDD Cup
churn data, with hundreds of mostly-missing numeric columns. Preprocessing now
includes feature selection (ANOVA relevance, keep 40) — a step notorious for
leaking when it is allowed to see test rows. One split, same two protocols:

```text
leaky protocol : 0.5240
clean protocol : 0.5315
inflation      : -0.0076
```

Read that again: on this split the **leaky pipeline looks worse**. A single
number doesn't just hide leakage — its noise can point either way. That is
precisely how the bug survives code review.

## The referee: repeat until the noise cancels

Ten repetitions of stratified 5-fold cross-validation (seeds 0–9). The honest
arm runs *all* preprocessing and selection inside every CV fold; the leaky arm
uses features that were imputed and selected once, on all 800 rows:

```text
      leaky_cv  honest_cv     gap
seed
0       0.7068     0.6692  0.0376
1       0.6987     0.6583  0.0404
2       0.6882     0.6507  0.0375
3       0.6221     0.5916  0.0306
4       0.6812     0.6388  0.0424
5       0.6690     0.6583  0.0107
6       0.6575     0.6651 -0.0076
7       0.6616     0.6578  0.0037
8       0.6386     0.6307  0.0079
9       0.7000     0.6418  0.0582

mean bias: +0.0261 +- 0.0210  |  leaky ahead in 9 of 10 repeats
```

The leak is real, systematic, and worth about **+0.026 AUC of fiction** —
silently reported as "model performance" in any single-run readout.

## Why this keeps happening

Because the failure is silent and the safe protocol is usually *more* code:
you must carry fitted medians, vocabularies, scalers and selected-feature
lists from train to test by hand, and nothing warns you when you don't.
Under deadline, the leaky path is the short path.

## Making the safe path the short path

That asymmetry is the design brief behind `dextra`. Every stateful step fits
with `return_params=True` and replays with `params=` — one argument each way.
The FIT call discloses everything it learned and ends with a decision
sentence:

```text
Decision: Fitted a 3-step featpipe pipeline (handle_missing -> encode -> scale);
33 new column(s) produced; combined params is a versioned, JSON-serialisable
artifact. Apply to held-out data with featpipe(df_test, params=...).
```

The artifact is plain JSON: version it next to your model, ship it, replay it.
And when the evaluation itself must be leak-proof, the same pipelines drop
into `sklearn.pipeline.Pipeline` and `cross_val_score` via `dextra.compat` —
which is exactly how the honest arm above was built.

## Reproduce everything

- **5 minutes, zero setup:** the
  [Colab quickstart](https://colab.research.google.com/github/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/00-leakage-in-5-minutes.ipynb) —
  `pip install pydextra`, both protocols, and a 10-split referee.
- **The full story:**
  [notebook 02](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/02-leakage-safe-pipeline.ipynb) —
  including this page's KDD selection experiment, verbatim.
- **The library:** `pip install pydextra` · [Getting started](getting-started.md) ·
  [API reference](api.md).

*`pydextra`'s public API is frozen since 0.6.0 — what you evaluate today is
what you run next year. Bug reports are
[read and answered](https://github.com/ahmedabdeltawab602-collab/dextra/issues/new/choose).*

*Data: IBM Telco churn public sample; KDD Cup churn dataset. Code: MIT.*
