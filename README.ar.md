# dextra — مكتبة لتحليل البيانات الاستكشافي

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/%D8%A7%D9%84%D8%AA%D9%88%D8%AB%D9%8A%D9%82-%D9%85%D8%AA%D8%A7%D8%AD-brightgreen.svg)](https://ahmedabdeltawab602-collab.github.io/dextra/)

**دوال بيانات تشرح نفسها بنفسها.** سطر واحد يدخل ← جدول مقاييس غني +
رسم متعدّد اللوحات + جملة `Decision:` واحدة تقول ما الذي جرى ولماذا.
والخطوات الحسّاسة للتسرّب تُرجع أيضاً خطة `params` قابلة لإعادة التطبيق —
تعلُّم على بيانات التدريب وتطبيق حرفي على الاختبار — فيصبح تسرّب البيانات
صعب الارتكاب سهواً.

نداء حقيقي واحد، حرفياً من
[دفتر خط الأنابيب الآمن ضد التسرّب](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/02-leakage-safe-pipeline.ipynb):

```python
train_fe, params = dx.featpipe(train, steps=steps, return_params=True)
```

```text
Decision: Fitted a 3-step featpipe pipeline (handle_missing -> encode -> scale);
33 new column(s) produced; combined params is a versioned, JSON-serialisable
artifact. Apply to held-out data with featpipe(df_test, params=...).
```

```bash
pip install pydextra
```

> **اسمان لمكتبة واحدة:** اسم `dextra` لم يكن متاحاً على PyPI، لذلك
> **تُثبّت `pydextra`** و**تستورد `dextra`**. المستودع والتوثيق والاستيراد كلها `dextra`.

[**التوثيق**](https://ahmedabdeltawab602-collab.github.io/dextra/) ·
[**جرّب على Colab**](https://colab.research.google.com/github/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/00-leakage-in-5-minutes.ipynb) ·
[**دفاتر Kaggle**](https://www.kaggle.com/ahmedabdeltawab40/code) ·
[**🇬🇧 English**](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/README.md)

مكتبة `dextra` أداة «الخيارات أولاً» لكامل سير العمل الاستكشافي — مفردات من
**63 دالة عامة** زائد 5 أغلفة متوافقة مع scikit-learn، أي **68 كائناً عاماً
قابلاً للنداء إجمالاً** (لكل اسم طويل اختصارُه؛ راجع
[جدول اتساق الواجهة](https://ahmedabdeltawab602-collab.github.io/dextra/api-consistency/)):
طبقة الدخول `load` في الصدارة، وتسع وحدات تحليلية، وأغلفة `compat`:

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

الدفاتر الثلاثة الكاملة منشورة مُنفَّذة على Kaggle:
[إنقاذ 9,291 رقم فاتورة](https://www.kaggle.com/code/ahmedabdeltawab40/rescuing-9-291-invoices-from-coercion-pydextra) ·
[بروتوكول كشف التسرّب](https://www.kaggle.com/code/ahmedabdeltawab40/two-pipelines-same-auc-0-8465-one-is-lying) ·
[تحليل أسعار الغذاء في مصر بالعربية](https://www.kaggle.com/code/ahmedabdeltawab40/egypt-food-prices-2010-2026-arabic-eda-pydextra).

---

## حالة المشروع

**مكتملة الملامح ومستقرة الواجهة منذ 0.6.0 — مجمّدة عن سياسة، لا عن هجران.**

- الواجهة العامة مقفلة: لا ميزات جديدة ولا تغيير تواقيع. ما تقيّمه اليوم هو
  ما تشغّله بعد عام.
- **بلاغات الأعطال تُقرأ ويُردّ عليها.** العطل المؤكَّد يُصلح باختبار أحمر
  أولاً ثم يصدر في نسخة ترقيع — افتح بلاغاً عبر
  [قوالب القضايا](https://github.com/ahmedabdeltawab602-collab/dextra/issues/new/choose).
- قبل 0.6.0 خضعت المكتبة لتدقيق خارجي عدائي: أحد عشر عيباً مُثبَتاً بالدليل،
  أُغلق كل منها أحمر←أخضر باختبار انحدارٍ مسمّى — راجع
  [`AUDIT_REPORT.md`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/AUDIT_REPORT.md)
  و[سجل التغييرات](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/CHANGELOG.md).
- الدَّين التقني المعلَن عمداً علني:
  [‏#1 — تقييم relevance على الصفوف المكتملة لكل ميزة](https://github.com/ahmedabdeltawab602-collab/dextra/issues/1).

---

## التثبيت

من PyPI:

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
