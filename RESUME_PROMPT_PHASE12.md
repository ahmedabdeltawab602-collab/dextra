# dextra — برومبت استئناف (ما بعد المرحلة 11: التصليب قبل النشر) — انسخه في محادثة جديدة

> الغرض: استئناف العمل على dextra. **كل المراحل الوظيفية 1–11 مكتملة، الإصدار 0.4.0،
> 348 اختباراً خضراء، مدفوعة إلى GitHub.** المتبقي: تصليب الجودة ثم **النشر آخر خطوة**.

---

## 0) دورك
أنت مطوّر/مراجع تقني للمكتبة. التزم: لا مجاملة، شرائح خضراء صغيرة
(كود + اختبارات + run_validation + commit)، وصنّف أي نقطة غير مؤكَّدة صراحةً.

## 1) الحالة الحالية (الأساس)
- **المستودع:** `D:\06 PythonProjects\dextra-project` — GitHub
  `ahmedabdeltawab602-collab/dextra`، فرع `main`. آخر commits: `8faa167` (إغلاق
  متابعات المراجعة + 0.4.0) ثم `0a3f97e` (هذا البرومبت).
- **بيئة المالك:** Python 3.13.13، venv، pandas 3.0.3، numpy 2.4.5،
  scikit-learn 1.8.0، pyarrow 24.0.0.
  التفعيل: `cd "D:\06 PythonProjects\dextra-project"; .\.venv\Scripts\Activate.ps1`
- **الأساس الأخضر المُتحقَّق:** `run_validation.ps1` → ALL GREEN: ruff نظيف،
  **348 اختباراً** (صفر تخطيات)، تغطية **76.86%** (عتبة 68%).
- **الإصدار:** 0.4.0 — **غير منشور على PyPI بعد (النشر آخر خطوة بقرار المالك).**

## 2) قرارات المالك المثبَّتة لبنود المراجعة (اختيرت للأداء وتجربة المستخدم — لا تُعد فتحها)
- **#4 العقد الموحّد:** الخيار (أ) — توسيع Phases 1–2 إلى العقد الكامل
  (`return_params` + `dextra_audit` + `df_name` + إدراجها في `_FAMILIES`)،
  **مع توافق رجعي**: يبقى المعامل القديم `name` يعمل كـalias مهجور (تحذير
  DeprecationWarning) حتى لا ينكسر كود المستخدمين.
- **#5 إكمال DRY:** كل الوحدات (~6) تستورد `now_iso`/`append_audit`/`json_safe`
  من `_utils` وتُحذف النسخ المحلية (`dashboard._now_iso` إلخ).
- **#6:** يُدرَج `dextra_audit` ضمن الـmanifest المُعاد من report/dash عند
  `return_params=True` (لا حذف للميزة).
- **#7:** معامل صريح `task=("auto"|"regression"|"classification")` في report/dash؛
  `auto` يستنتج من dtype الهدف أولاً ثم عتبة nunique الموثَّقة.
- **#9:** إضافة `binize` كاسم بديل و`bin` يبقى alias (لا كسر).
- **#10:** تقليص aliases الزائدة لـPhase 3: alias قصير واحد لكل دالة يبقى رسمياً،
  والبقية تعمل مع DeprecationWarning موثَّق في CHANGELOG.
- **#11:** `df_name`/`name` صريح في كل واجهة عامة، و`get_variable_name` رجعة أخيرة
  فقط (علّة peek في 11.2 أثبتت هشاشته).
- **#12:** توثيق افتراض الثقة في `validate_rules` (df.eval) في docstring — لا تقييد.
- **#13:** `DextraClusterer.labels_` يخزّن تسميات fit الفعلية (لا NearestCentroid).
- **#14:** يُقبل استيراد matplotlib المبكر كما هو (core dependency) — مغلق بلا تغيير.

## 3) خطة الجلسات القادمة (بالترتيب — النشر آخراً)
1. **شريحة التوثيق:** `docs/api.md` لا يذكر `load`/`peek` إطلاقاً — وثّق المرحلة 11
   كاملة (load/peek + الأنواع الخمسة + on_ambiguous + الخطة القابلة للإعادة + الأمان)
   وحدّث getting-started بسطر `df = dx.load(...)` كنقطة الدخول.
2. **شريحة العقد (#4 + #5 + #6):** التوسيع + DRY + إصلاح التدقيق المهمل. الأكبر —
   قسّمها: (5 أولاً سهلة) ثم (4) ثم (6).
3. **شريحة #7 + المنخفضة:** task= + binize + تقليص aliases + #11 + #12 + #13.
4. **شريحة التغطية:** الأدنى أولاً — `cleaning.py` 61% ثم `_features_numeric.py` 65%
   ثم `stats_advanced.py` 67% (هدف واقعي: إجمالي ≥80%).
5. **إثبات الأداء:** benchmark موثَّق على بيانات ~1M صف (load/clean/features) يُرفق
   بالتوثيق — يعالج «الإنتاجي 66/100» في المراجعة.
6. **النشر (الأخير):** تأكد من خضرة CI كاملاً (بما فيها perf-extras وpandas-matrix)،
   رفع الإصدار النهائي، ثم `git tag vX.Y.Z` + GitHub Release → `publish.yml` ينشر
   على PyPI تلقائياً.

## 4) تحذيرات بيئية (مجرَّبة — حرجة)
1. **المزامنة تُفسد الملفات بـnull bytes** (أصابت .py و`.git/config` فعلاً):
   اكتب/عدّل الكود عبر bash ثم تحقّق فوراً (`py_compile` + عدّ null bytes).
   لا تثق بـWrite/Edit دون تحقّق. إن تلف ملف git: نظّفه من `\x00` عبر python.
2. **بيئة المساعد المعزولة:** Python 3.10 + pandas 2.x، بلا pytest/ruff/sklearn/pyarrow،
   والشبكة محجوبة (لا pip/push). الاختبار بعزل عبر shim + ستاب pytest مصغّر.
   **التحقّق الحاسم دائماً على جهاز المالك** عبر `run_validation.ps1`.
3. **فروق pandas 3.0:** النصوص StringDtype (استخدم `pd.api.types.is_object_dtype/
   is_string_dtype` لا `== object`)؛ **`.map()` يستنتج str dtype** — أنشئ
   `pd.Series(..., dtype=object)` صراحةً؛ dtype التواريخ قد يكون `datetime64[us]`
   (في الاختبارات: `startswith("datetime")`).
4. **قفل git:** إن علِق `index.lock` من عمليات بيئة المساعد، يحذفه المساعد بأداة
   حذف الملفات بعد الاستئذان (أو `del .git\index.lock` على جهاز المالك).

## 5) أول إجراءاتك في المحادثة الجديدة
1. فعّل البيئة وأكّد الأساس: `run_validation.ps1` → يجب **348 + ALL GREEN**.
2. `git status` نظيف و`git log --oneline -3` (HEAD = 0a3f97e أو أحدث، مدفوع).
3. **افحص حالة GitHub Actions** لأول تشغيل لوظيفتي `perf-extras` و`pandas-matrix`
   (اطلب من المالك لقطة/نتيجة إن لم تتوفر أداة وصول) — أي أحمر هناك يُعالج قبل البدء.
4. ابدأ بالبند 1 من §3 (التوثيق) ما لم يوجّه المالك غير ذلك. أي تعديل:
   bash + py_compile + عدّ null bytes → run_validation → commit → push من جهاز المالك.

## 6) ملفات مرجعية
`ROADMAP.md` ← `CHANGELOG.md` (0.4.0) ← `AUDIT_REPORT.md` §12 (سجل المشكلات؛
القرارات أعلاه تحسمه) ← `LOADER_PHILOSOPHY.md` ← `src/dextra/_loader.py` ←
`docs/api.md` (الفجوة) ← `tests/test_phase11.py`.
