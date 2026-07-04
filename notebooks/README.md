# dextra notebooks

Three end-to-end, fully-executed notebooks. Every narrative claim in them is
backed by an output produced in the same notebook; each one was run top to
bottom before publishing, and each carries its data license and attribution.

| # | Notebook | Language | Data (license) | What it teaches |
|---|----------|----------|----------------|-----------------|
| 1 | [`01-messy-csv-to-decision.ipynb`](01-messy-csv-to-decision.ipynb) | English | UCI Online Retail (CC BY 4.0) | Disclosed loading of a genuinely messy 541,909-row CSV: a naive load destroys 9,291 invoices; editing the load *plan* rescues them. Cancellations vs sales vs anomalies, duplicates, guest checkouts, and an honest seasonal read -- ending in `edareport`. |
| 2 | [`02-leakage-safe-pipeline.ipynb`](02-leakage-safe-pipeline.ipynb) | English | IBM Telco churn (public sample) + KDD Cup 09 via OpenML (Public) | Data leakage measured, not moralized: the subtle crime costs nothing on tame data (0.8465 vs 0.8465), the selection crime hides inside single-split noise, and repeated CV exposes a +0.026 systematic bias (9/10 repeats). FIT -> APPLY artifacts, `selectpipe` replay, `dextra.compat` inside CV folds. |
| 3 | [`03-egypt-food-prices-ar.ipynb`](03-egypt-food-prices-ar.ipynb) | **Arabic** (code in English) | WFP Food Prices for Egypt via HDX (CC BY-IGO) | تحليل استكشافي صادق بالعربية: حُبيبية مختلطة، فقدان بنيوي، انقطاع سلسلة يقرره اختبار ت المزدوج، سعر الصرف الضمني 5.7→52.5، التضخم بالجنيه مقابل الدولار، وشواذ حقيقية لا تُحذف. |

## Running them

Each notebook installs its own dependencies in the first cell
(`%pip install "pydextra[...]"`) and downloads its data on first run into
`notebooks/data/` (cached; never committed -- see [`data/README.md`](data/README.md)).

```bash
python -m venv .venv-nb && .venv-nb/Scripts/activate   # or source .venv-nb/bin/activate
pip install nbconvert ipykernel
python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

Notebook 3's dataset ships in this repo's data folder only after its download
cell runs; the two HDX source files total under 400 KB.

## Archive

Earlier exploratory notebooks (`dextra_usage`, `dextra_project_walkthrough`,
`dextra_full_function_test`) live in [`archive/`](archive/) -- kept for
reference, superseded by the three above.
