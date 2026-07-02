# dextra — مكتبة لتحليل البيانات الاستكشافي

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *مساعدات خفيفة لتحليل البيانات الاستكشافي، مبنية فوق pandas وseaborn وplotly.*

**🇬🇧 English:** [README.md](README.md)

مكتبة `dextra` أداة شاملة للتحليل الاستكشافي قائمة على مبدأ واحد:
**سطر واحد من الكود ← جدول رقمي غني + رسم متعدّد اللوحات + جملة `Decision:` واحدة.**
تضمّ حالياً **68 كائناً عاماً متمايزاً** — اللودر (`load`) في الصدارة، وتسع وحدات تحليلية، وأغلفة متوافقة مع scikit-learn:

| الوحدة | الدوال | ما تغطّيه |
|---|---|---|
| `_loader` (طبقة الدخول) | `load`, `peek` (اختصارات `dload`, `dpeek`) | مصدر خام فوضوي (CSV/TSV/Excel/JSON/Parquet) ← DataFrame مُصنَّف بإفصاح كامل قابل للإعادة. |
| `stats` / `plots` | `describe_numeric`, `plot_histograms`, `plot_boxplots` | أساسيات EDA: ملخّصات غنية ورسوم افتراضية أفضل. |
| `stats_advanced` | 22 دالة (z-scores، الالتواء، الارتباط، الانحدار الخطي، فترات الثقة، اختبارات t، ANOVA، كاي-تربيع، VIF، عدم توازن الفئات، ...) | الوصفية والثنائية والاستدلال واختبارات الفروض وتشخيص ML. |
| `cleaning` | 10 دوال (`clean_report`, `handle_missing`/`impute`, `dedupe`, `clip_outliers`/`winsor`, ...) | تدقيق جودة البيانات وتنظيفها عبر مراحل DAMA-DMBOK؛ fit/apply آمن ضد التسرّب لـ`handle_missing` و`clip_outliers`. |
| `features` | 8 دوال (`transform`, `scale`, `bin`, `encode`, `dtfeats`, `cross`, `aggfeat`, `featpipe`) | هندسة ميزات آمنة ضد التسرّب بنمط fit/apply. |
| `selection` | 5 دوال (`redundancy`, `relevance`, `importance`, `rfe`, `selectpipe`) | اختيار الميزات: Filter / Embedded / Wrapper، آمن ضد التسرّب. |
| `modeling` | `regress`, `classify`, `cluster` | نماذج أساس فورية: fit / apply / compare بأثر هجين (JSON + مُقدِّر مُدرَّب). |
| `evaluation` | `confusion_report`, `roc_pr`, `residual_analysis`, `learning_curves` | تقييم نماذج متعدّد المقاييس (وضع التسميات أو الأثر المحفوظ). |
| `timeseries` | `tsdecomp`, `tsstat`, `tsfcast` | تفكيك السلاسل الزمنية واختبارات الاستقرار والتنبؤ الأساسي. |
| `report` | `edareport` | تقرير EDA تفاعلي HTML قائم بذاته (يجمع المراحل 1‑8). |
| `dashboard` | `dash` | يولّد تطبيق Streamlit تفاعلياً قائماً بذاته. |
| `compat` | `DextraFeaturePipeline`/`DextraSelectPipeline`، `DextraRegressor`/`DextraClassifier`/`DextraClusterer` | أغلفة متوافقة مع scikit-learn تدخل في `Pipeline`/`GridSearchCV`. |

نفّذ `dx.functions()` لطباعة كامل الواجهة العامة مع ملخّص سطر واحد لكل دالة.
معظم الدوال توفّر نمط `method='compare'` يعرض كل الخيارات **دون أن يقرّر عنك**،
ونمط fit/apply (عبر `params` dict) يتعلّم على التدريب ويُطبَّق حرفياً على بيانات
الاختبار — الضمان ضد تسرّب البيانات.

---

## التثبيت

من PyPI (بعد النشر):

```bash
pip install pydextra   # اسم الاستيراد كما هو: import dextra as dx
```

مباشرة من GitHub:

```bash
pip install git+https://github.com/ahmedabdeltawab602-collab/dextra.git
```

من المصدر بوضع التطوير:

```bash
git clone https://github.com/ahmedabdeltawab602-collab/dextra.git
cd dextra
pip install -e ".[dev]"
```

### إضافات اختيارية

النواة خفيفة (numpy, pandas, matplotlib, seaborn, scipy). فعّل ما تحتاجه:

```bash
pip install "pydextra[io]"       # charset-normalizer + clevercsv + openpyxl: أفضل كشف للودر + Excel
pip install "pydextra[ml]"       # scikit-learn: regress / classify / cluster، المنتقيات النموذجية، dextra.compat
pip install "pydextra[viz]"      # plotly: صناديق plot_boxplots التفاعلية
pip install "pydextra[ts]"       # statsmodels: السلاسل الزمنية STL + ADF / KPSS
pip install "pydextra[dash]"     # streamlit: لوحة المعلومات التفاعلية المولَّدة
pip install "pydextra[perf]"     # polars + pyarrow: خلفيات DataFrame بديلة
pip install "pydextra[notebook]" # jupyter + ipykernel
pip install "pydextra[docs]"     # موقع mkdocs-material
```

---

## بداية سريعة

معظم سير العمل يبدأ من **اللودر** — يحوّل ملفاً فوضوياً إلى DataFrame مُصنَّف وموثَّق:

```python
import dextra as dx

df = dx.load("your_data.csv")   # ترميز + فاصل + استدلال نوع لكل عمود، بإفصاح قابل للإعادة
```

وبقية البداية السريعة تستخدم إطاراً صغيراً في الذاكرة ليعمل المقطع كما هو:

```python
import pandas as pd
import numpy as np
import dextra as dx

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "price":    rng.normal(100, 15, 500),   # السعر
    "quantity": rng.integers(1, 20, 500),   # الكمية
    "score":    rng.beta(2, 5, 500) * 100,  # الدرجة
})

# 1) ملخّص رقمي غني
dx.describe_numeric(df)

# 2) مدرجات تكرارية مع إحصاءات جانبية
dx.plot_histograms(df, bins=30)

# 3) صناديق Plotly تفاعلية
dx.plot_boxplots(df)
```

### استرجاع النتائج بدلاً من عرضها

كل دالة تملك `show=` للتحكم بالعرض و`return_*=` لإرجاع النتائج، حتى تستخدمها في خطوة لاحقة:

```python
summary = dx.describe_numeric(df, return_df=True, raw=True, show=False)
fig, stats = dx.plot_boxplots(df, return_fig=True, return_df=True, show=False)
```

استخدم `raw=True` إذا أردت قيماً عددية غير منسَّقة (`float64`) لأغراض الحساب أو التصدير إلى Excel.

---

## مرجع الدوال

### `describe_numeric`

تُرجع ملخصاً إحصائياً غنياً يتضمّن 21 مقياساً لكل عمود: المتوسط، الانحراف المعياري، التباين، معامل الاختلاف، الأدنى، الربيع الأول، الوسيط، الفجوة بين المتوسط والوسيط، الربيع الثالث، الأعلى، المدى الرُّبَيعي، حدّا تكي العلوي والسفلي، عدد القيم الشاذة ونسبتها، العدد، المفقود، الفريد، الالتواء، التفلطح، والمنوال.

**المعاملات الرئيسية:**

- `df`: الـ DataFrame.
- `cols`: قائمة الأعمدة (افتراضياً جميع الأعمدة العددية).
- `decimals`: عدد الخانات العشرية للتنسيق.
- `iqr_multiplier`: معامل IQR لتحديد القيم الشاذة (1.5 قاعدة تكي الكلاسيكية).
- `metrics_as_rows`: لجعل المقاييس في الصفوف والأعمدة في الأعمدة.
- `show`, `return_df`, `raw`: للتحكم بالعرض والإرجاع.

### `plot_histograms`

مدرّج تكراري matplotlib مع KDE، ولوحة إحصاءات نصية (monospace) بجانب كل رسمة. المتوسط (أحمر متقطّع) والوسيط (أخضر متقطّع مُنقّط) مُبرَزان.

### `plot_boxplots`

رسم صناديق Plotly تفاعلي، صف لكل عمود، أفقي، مع خطوط متقطّعة عند حدَّي تكي وتعليق ملخّص لكل صف.

### أسماء مختصرة (متوافقة مع النسخة السابقة)

`numdesc`, `hister`, `boxpl` لا تزال تعمل وتُحوِّل للدوال الجديدة.

---

## التطوير المحلي

```bash
git clone https://github.com/ahmedabdeltawab602-collab/dextra.git
cd dextra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## المساهمة

نرحّب بطلبات السحب (Pull Requests). قبل فتح طلب، شغِّل `pytest` و`ruff check .`.

---

## الترخيص

[MIT](LICENSE) © 2026 Ahmed Abd El Tawab
