# dextra — برومبت استئناف (ما بعد المرحلة 11) — انسخه في محادثة جديدة

> الغرض: استئناف العمل على dextra. **المرحلة 11 (المُحمِّل الذكي) مكتملة ومدفوعة
> إلى GitHub، والإصدار 0.4.0، وكل بنود المراجعة المحرِّرة مغلقة.**

---

## 0) دورك
أنت مطوّر/مراجع تقني للمكتبة. التزم: لا مجاملة، شرائح خضراء صغيرة
(كود + اختبارات + run_validation + commit)، وصنّف أي نقطة غير مؤكَّدة صراحةً.

## 1) الحالة الحالية (الأساس)
- **المستودع:** `D:\06 PythonProjects\dextra-project` — GitHub
  `ahmedabdeltawab602-collab/dextra`، فرع `main`، آخر commit مدفوع `8faa167`.
- **بيئة المالك:** Python 3.13.13، venv، pandas 3.0.3، numpy 2.4.5،
  scikit-learn 1.8.0، **pyarrow 24.0.0** (مثبَّت حديثاً).
  التفعيل: `cd "D:\06 PythonProjects\dextra-project"; .\.venv\Scripts\Activate.ps1`
- **الأساس الأخضر المُتحقَّق:** `run_validation.ps1` → ALL GREEN:
  ruff نظيف، **348 اختباراً** (صفر تخطيات)، تغطية **76.86%** (عتبة 68%).
- **الإصدار:** `_version.py` = **0.4.0** (غير منشور على PyPI بعد).

## 2) ما أُنجز في الجلسة السابقة (كله مدفوع)
1. **إصلاحات الإصدار** (بنود المراجعة #1–#3): 0.3.0، حذف `.bak` من التتبع،
   لوحة `dash` افتراضيها `data_format="auto"` (parquet→csv) وpickle اختياري بتحذير أمني.
2. **المرحلة 11.2 — Excel** (xlsx/xlsm عبر openpyxl كسولاً): أوراق، كتلة بيانات،
   ترويسات مدمجة متعددة الصفوف (`header_rows=`)، قيم-لا-صيغ، أنواع خلايا أصلية،
   خطط قابلة للإعادة. + إصلاح حقيقي: `peek` كان يخطئ النوع (الامتداد من الاسم المعروض).
3. **11.3a — parquet + json/ndjson**: تمرير مُنمَّط، كشف شكل JSON، تسلسل
   القيم المتداخلة مع إفصاح، replay.
4. **11.3b — SQL آمن**: جملة واحدة ببارامترات فقط (`sql=`/`sql_params=`)،
   SQLite يُفتح **read-only** (mode=ro)، حارس صفوف افتراضي 1M، أخطاء ملفوفة واضحة،
   replay. + استخراج ذيل الإفصاح المشترك `_disclose_and_finish` لكل الأنواع.
5. **إغلاق متابعات المراجعة + 0.4.0**: فحص الفاصل quote-aware ويتجاهل التمهيد
   (أغلق قيد 11.1 التجميلي)، CI: وظيفة `perf-extras` (#8) + `pandas-matrix`
   غير حاجبة (#8b)، CHANGELOG موحَّد تحت 0.4.0.

## 3) قرارات مثبَّتة (لا تُعد فتحها)
كل قرارات `LOADER_PHILOSOPHY.md` + `RESUME_PROMPT_PHASE11.md` §3 سارية.
عقد المُحمِّل الموحَّد يشمل الآن: csv/tsv + Excel + parquet + json + SQL.

## 4) تحذيرات بيئية (مجرَّبة)
1. **المزامنة تُفسد ملفات أدوات الكتابة**: اكتب/عدّل الكود عبر bash ثم تحقّق فوراً
   (`py_compile` + عدّ null bytes). لا تثق بـWrite/Edit دون تحقّق.
2. **بيئة المساعد المعزولة**: Python 3.10 + pandas 2.x، بلا pytest/ruff/sklearn/pyarrow؛
   الاختبار بعزل عبر shim + ستاب pytest مصغّر. **التحقّق الكامل على جهاز المالك**
   عبر `run_validation.ps1`. الشبكة محجوبة (لا pip ولا git push من بيئة المساعد).
3. **فروق pandas 3.0**: النصوص StringDtype (لا تستخدم `== object`؛ استخدم
   `pd.api.types.is_object_dtype/is_string_dtype`)، **`.map()` يستنتج str dtype**
   (أنشئ `pd.Series(..., dtype=object)` صراحةً)، dtype التواريخ قد يكون
   `datetime64[us]` (في الاختبارات: `startswith("datetime")`).
4. **قفل git**: إن علِق `index.lock` بعد عمليات git من بيئة المساعد، يحذفه
   المساعد بأداة حذف الملفات (أو `del .git\index.lock` على جهازك).

## 5) المعلّق / المسارات المقترحة (اختر مع المالك)
- **A — النشر**: tag `v0.4.0` + GitHub Release → `publish.yml` ينشر على PyPI
  تلقائياً. قبلها: راجع نتيجة وظائف CI الجديدة على GitHub Actions (أول تشغيل بعد الدفع).
- **B — رفع التغطية**: `cleaning.py` (61%) و`stats_advanced.py` (67%)
  و`_features_numeric.py` (65%) الأدنى؛ شرائح اختبارات مركَّزة.
- **C — مرحلة وظيفية جديدة** من ROADMAP (ما بعد المرحلة 11).
- ملاحظة أمنية: التوكن القديم كان مضمّناً في remote URL وعُولج؛ تأكد من تدويره.

## 6) أول إجراءاتك في المحادثة الجديدة
1. فعّل البيئة وأكّد الأساس: `run_validation.ps1` → يجب **348 + ALL GREEN**.
2. تحقّق من `git status` (يجب نظيفاً) و`git log --oneline -3` (HEAD = 8faa167 أو أحدث).
3. لا تبدأ تعديلاً قبل اختيار المسار. أي تعديل: bash + py_compile + عدّ null bytes
   → run_validation → commit.

## 7) ملفات مرجعية
`ROADMAP.md` ← `CHANGELOG.md` (قسم 0.4.0) ← `LOADER_PHILOSOPHY.md` ←
`src/dextra/_loader.py` (1625 سطراً، 5 أنواع مصادر) ← `tests/test_phase11.py` ←
`AUDIT_REPORT.md`.
