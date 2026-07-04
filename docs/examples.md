# Example notebooks

Three end-to-end notebooks, fully executed with real outputs saved in place.
They are the recommended reading order after [Getting started](getting-started.md).

## 1 — From a messy CSV to an actionable report

**[`notebooks/01-messy-csv-to-decision.ipynb`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/01-messy-csv-to-decision.ipynb)**
· English · data: UCI Online Retail (CC BY 4.0)

541,909 real e-commerce rows. A naive load silently destroys 9,291 invoices;
editing the disclosed load *plan* rescues every one. Cancellations, duplicates,
guest checkouts, honest seasonality, and a one-line `edareport` finale.

## 2 — Build a leakage-safe ML pipeline

**[`notebooks/02-leakage-safe-pipeline.ipynb`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/02-leakage-safe-pipeline.ipynb)**
· English · data: IBM Telco churn + KDD Cup 09 (OpenML, Public)

Leakage measured instead of moralized. Preprocessing-before-split costs
*nothing* on tame data (0.8465 vs 0.8465 -- said honestly); selection-before-split
injects a bias a single check cannot even see; 20 splits and repeated CV expose
it as systematic (+0.026, ahead in 9 of 10 repeats). Stars: `featpipe` /
`selectpipe` FIT->APPLY artifacts and `dextra.compat` inside CV folds.

## 3 — قصة الأسعار: أسعار الغذاء في مصر (بالعربية)

**[`notebooks/03-egypt-food-prices-ar.ipynb`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/03-egypt-food-prices-ar.ipynb)**
· Arabic narrative, English code · data: WFP Food Prices for Egypt via HDX (CC BY-IGO)

تحليل استكشافي كامل بالعربية على بيانات حقيقية من السوق المصري: حُبيبية
مختلطة في عمود واحد، فقدان بنيوي لا يُعوَّض، انقطاع سلسلة عام 2022 يحسمه
اختبار ت المزدوج لا المزاج، سعر الصرف الضمني من 5.7 إلى فوق 52 جنيهاً
للدولار، والتضخم بالجنيه مقابل الدولار — مع قاموس مصطلحات ثنائي اللغة.

---

Each notebook installs its own extras in its first cell and downloads its data
on first run; see
[`notebooks/README.md`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/README.md)
for run instructions and
[`notebooks/data/README.md`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/data/README.md)
for data licenses and attributions.
