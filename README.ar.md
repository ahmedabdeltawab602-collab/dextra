# dextra — مكتبة لتحليل البيانات الاستكشافي

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *مساعدات خفيفة لتحليل البيانات الاستكشافي، مبنية فوق pandas وseaborn وplotly.*

**🇬🇧 English:** [README.md](README.md)

مكتبة `dextra` توفّر ثلاث دوال صغيرة ومركّزة لتسريع أول عشر دقائق من أي تحليل بيانات:

| الدالة | ما تفعله |
|---|---|
| `describe_numeric` | نسخة أغنى من `df.describe()` تتضمّن IQR وعدد القيم الشاذّة وStd والانحراف/التفلطح مع تنسيق جاهز للعرض. |
| `plot_histograms` | شكل matplotlib لكل عمود، مدرّج تكراري + KDE على اليسار ولوحة إحصائيات كاملة على اليمين. |
| `plot_boxplots` | رسوم Plotly تفاعلية (صناديق أفقية مُكدَّسة) مع تعليقات إحصائية لكل متغير. |

---

## التثبيت

من PyPI (بعد النشر):

```bash
pip install dextra
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

---

## بداية سريعة

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
