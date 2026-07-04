# Data sources

Files in this directory are downloaded by the notebooks on first run and are
**never committed** to the repository (see the repo `.gitignore`).

| Used by | Dataset | Source (stable link) | License |
|---------|---------|----------------------|---------|
| `01-messy-csv-to-decision.ipynb` | Online Retail — Chen, D. (2015), DOI 10.24432/C5BW33 | <https://archive.ics.uci.edu/dataset/352/online+retail> | CC BY 4.0 |
| `02-leakage-safe-pipeline.ipynb` | Telco Customer Churn (IBM sample) | <https://github.com/IBM/telco-customer-churn-on-icp4d> | Public IBM sample dataset |
| `02-leakage-safe-pipeline.ipynb` | KDD Cup 2009 — Orange churn (`fetch_openml(data_id=1112)`, cached in `openml/`) | <https://www.openml.org/d/1112> | Public (per OpenML) |
| `03-egypt-food-prices-ar.ipynb` | WFP Food Prices for Egypt (`wfp_food_prices_egy.csv`, `wfp_markets_egy.csv`) | <https://data.humdata.org/dataset/wfp-food-prices-for-egypt> | CC BY-IGO (attribution: WFP via HDX; primary sources CAPMAS & IDSC) |

Notebook 01 downloads `online_retail.zip` (22.6 MB) from the UCI static link and
converts it once to `online_retail_raw.csv` (cached; re-runs cost nothing).

Notebook 02 downloads the Telco CSV (~1 MB) and fetches KDD Cup 09 through
scikit-learn's OpenML client into `openml/` (~15 MB, one time).

Notebook 03 downloads two small HDX files (< 400 KB total) from the stable
resource links embedded in the notebook.

A local `WUZZUF Job Posts (2014-2016)/` folder may also exist here: it is
**reserved for a possible future notebook**, is licensed CC BY-NC-SA 4.0
(more restrictive than the datasets above), is not used by any committed
notebook, and -- like everything else in this directory -- is never committed.
