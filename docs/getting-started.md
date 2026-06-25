# Getting started

## Installation

```bash
pip install dextra                # core: numpy, pandas, matplotlib, seaborn, scipy
pip install "dextra[ml]"          # + scikit-learn (models, model-based selectors, compat)
pip install "dextra[viz]"         # + Plotly (interactive box-plots)
pip install "dextra[dev]"         # everything needed to develop and test dextra
```

scikit-learn and Plotly are **optional**. The core library works without them;
the relevant functions raise a clear error if their extra is missing.

## A one-line tour

Everything starts with `load`, the entry layer (Phase 11): it turns a raw,
messy source into a typed `DataFrame` and discloses exactly how it parsed it.

```python
import numpy as np
import pandas as pd
import dextra as dx

# A small synthetic sales table so this whole tour is copy-paste runnable.
rng = np.random.default_rng(42)
sales = pd.DataFrame({
    "price":  rng.normal(100, 15, 500).round(2),
    "age":    rng.integers(18, 70, 500),
    "income": rng.normal(50_000, 12_000, 500).round(2),
    "churn":  rng.integers(0, 2, 500),
})
sales.to_csv("sales.csv", index=False)        # create the example file

df = dx.load("sales.csv")                     # raw file -> typed DataFrame + full disclosure
dx.describe_numeric(df)                       # rich numeric summary + plot
dx.audit(df)                              # data-quality audit
# the model steps below need the ml extra:  pip install "dextra[ml]"
dx.regress(df, y="price", method="compare")   # rank regression baselines
train, df_new = df.iloc[:400], df.iloc[400:]  # a held-out split for the apply demo
out, params = dx.classify(train, y="churn", method="forest", return_params=True)
preds = dx.classify(df_new, params=params)    # apply, no re-fit (leakage-safe)
dx.cluster(df, method="kmeans")               # unsupervised, automatic k
```

## scikit-learn interoperability

dextra's pipelines and models also expose the standard scikit-learn API via
`dextra.compat`, so they drop straight into a `Pipeline` or `GridSearchCV`:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dextra.compat import DextraFeaturePipeline, DextraClassifier

X = df[["age", "income"]]
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe = Pipeline([
    ("fe", DextraFeaturePipeline(steps=[
        {"fn": "scale", "cols": ["age", "income"], "method": "standard"}])),
    ("clf", DextraClassifier(method="forest")),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```
