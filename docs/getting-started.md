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

```python
import dextra as dx

dx.describe_numeric(df)                       # rich numeric summary + plot
dx.clean_rep(df)                              # data-quality audit
dx.regress(df, y="price", method="compare")   # rank regression baselines
out, params = dx.classify(df, y="churn", method="forest", return_params=True)
preds = dx.classify(df_new, params=params)    # apply, no re-fit (leakage-safe)
dx.cluster(df, method="kmeans")               # unsupervised, automatic k
```

## scikit-learn interoperability

dextra's pipelines and models also expose the standard scikit-learn API via
`dextra.compat`, so they drop straight into a `Pipeline` or `GridSearchCV`:

```python
from sklearn.pipeline import Pipeline
from dextra.compat import DextraFeaturePipeline, DextraClassifier

pipe = Pipeline([
    ("fe", DextraFeaturePipeline(steps=[
        {"fn": "scale", "cols": ["age", "income"], "method": "standard"}])),
    ("clf", DextraClassifier(method="forest")),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```
