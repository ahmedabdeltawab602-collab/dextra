# dextra — Resume Prompt (انسخ هذا في محادثة جديدة)

أنت تتعاون معي على مكتبة **dextra** لتحليل البيانات وعلم البيانات. دقة المعلومات
والكود مطلوبة 100%، ونلتزم بـ«التراكم الهادئ»: أساس مُختبَر وموثَّق وأخضر في CI
قبل أي توسّع، وتقسيم العمل إلى شرائح صغيرة قابلة للتحقّق.

## أين نحن الآن (يونيو 2026)
- المشروع: `D:\06 PythonProjects\dextra-project` — مرفوع على GitHub
  (`ahmedabdeltawab602-collab/dextra`)، و**CI أخضر** على مصفوفة Python 3.9–3.12
  (وظائف: test / benchmarks / build) + workflow `docs` ينشر على GitHub Pages.
- الإصدار الحالي: **0.2.0** (لم يُنشَر على PyPI بعد — مؤجَّل بقرارنا).
- بيئة التطوير: Python 3.13، venv، pandas 3.0.3، numpy 2.4.x، scipy، scikit-learn 1.8.
  التفعيل: `cd "D:\06 PythonProjects\dextra-project"; .\.venv\Scripts\Activate.ps1`
- التحقّق المحلي: `powershell -ExecutionPolicy Bypass -File ".\run_validation.ps1"`
  (ruff --fix ثم ruff صارم ثم pytest + تغطية، العتبة **68%**، الفعلي **72.37%**).

## المُنجَز والمُختبَر
- **Phases 1–7 مكتملة (55 دالة عامة + aliases):** stats/plots، stats_advanced (22)،
  cleaning (10)، features (8)، selection (5)، modeling (regress/classify/cluster)،
  **evaluation (confusion_report/roc_pr/residual_analysis/learning_curves)**.
- **Phase 7 — `evaluation.py` (آخر إنجاز، commit `473c3a6`، مدفوع):**
  - أربع دوال + aliases `confrep/rocpr/residan/learncv` على عقد §4.7 الموروث من
    Phase 6 (سطر واحد ← جدول مقاييس + لوحات متعددة + جملة Decision + أثر تدقيقي).
  - **وضعا الإدخال:** label (df + y_true + y_pred/scores) و artifact (params من
    Phase 6؛ يشتقّ y_true/y_pred/الاحتمالات من المُقدِّر المُدرَّب).
  - report descriptor آمن-JSON عبر `return_params` (لا يبني مُقدِّراً).
  - `learning_curves` هي الوحيدة التي تُعيد التدريب — على عيّنات فرعية ضمن CV (لا تسريب).
  - وثيقة `EVALUATION_PHILOSOPHY.md`؛ اختبارات `tests/test_phase7.py` (17 اختباراً)؛
    الإجمالي **200 اختباراً خضراء**، تغطية **72.37%** (evaluation.py **89%**).
- **تقرير التقييم الحيّ:** `dextra_Evaluation_Report.docx` (مجلد «شرح بايثون...»)،
  **الإصدار 1.5** — فيه قسم 14 (تحديث Phase 7)، و"Phase 7 Deliverables"،
  و"Current Project State"، وتحديث ROADMAP، و"Recommended Next Step".

## المؤجَّل عمداً
- **النشر على PyPI**: جاهز عبر `publish.yml` (Trusted Publishing/OIDC). يتطلّب فعلي:
  2FA على PyPI، pending publisher (project=dextra, repo، workflow=publish.yml,
  environment=pypi)، بيئة `pypi` على GitHub، ثم Release بوسم `v0.2.0`. مؤجَّل حتى
  تكتمل المكتبة.
- **سبرنت أداء أعمق**: chunking/out-of-core، تقليل النسخ وتتجيه الحلقات، بوابة
  anti-regression على benchmarks، و`n_jobs`. (الأداء الآن 8.5/10.)

## المطلوب التالي: Phase 8 — `timeseries.py`
أساسيات السلاسل الزمنية على نفس العقد الموحّد: `ts_decompose` (تفكيك اتجاه/موسمية/
بواقي)، `ts_stationarity` (اختبار سكون: ADF/KPSS + فروقات مقترحة)، `ts_quick_forecast`
(تنبؤ خط أساس بنافذة تحقّق). نفس §4.7: سطر واحد ← جدول مقاييس + رسم متعدّد اللوحات +
جملة Decision + أثر تدقيقي. التبعيات: numpy/pandas/scipy؛ و`statsmodels` اختياري عند
اللزوم (بموافقة صريحة، استيراد كسول كنمط sklearn في Phases 5–7).

## أول إجراء مطلوب منك في المحادثة الجديدة
1. اقرأ بالترتيب: `ROADMAP.md`، `MODELING_PHILOSOPHY.md` (§4.7 العقد)،
   `EVALUATION_PHILOSOPHY.md` (نمط Phase 7 الأحدث)، `CHANGELOG.md`، وبنية
   `src/dextra/` (لاحظ `evaluation.py` + `features` حزمة وحدات + façade).
2. اعرض تصميم `TIMESERIES_PHILOSOPHY.md` المختصر (الحدود + المبادئ على نسق ملفات
   الفلسفة) ثم توقيعات الدوال الثلاث للموافقة السريعة — وحدّد بوضوح أي تبعية
   اختيارية جديدة (statsmodels) قبل اعتمادها.
3. نفّذ مرحلياً بنفس الانضباط: دالة-دالة، مع اختبارات تُبقي التغطية فوق 68%، وتشغيل
   `run_validation.ps1` قبل كل commit.

## تحذيرات تقنية مكتسبة (مهمة جداً — سبّبت أعطالاً فعلية)
1. **شغّل `run_validation.ps1` قبل كل `git commit`.** تخطّيه أحمَر CI: ruff **E302**
   (سطران فارغان قبل كل def)، **I001** (ترتيب الاستيرادات)، **F821** (اسم غير معرّف).
   وانتبه لبوّابة **التغطية**: أي وحدة جديدة بلا اختبارات تُسقِط الإجمالي تحت 68%
   (حدث فعلاً في Phase 7 قبل إضافة `test_phase7.py`).
2. **المزامنة تحقن null bytes** في ملفات .py أحياناً. عند «source code string cannot
   contain null bytes»: `open(f,'rb').read().replace(b'\x00',b'')` ثم أعد التحقق.
3. **لا يمكن حذف/إعادة تسمية ملفات على المجلد المتزامن** (Operation not permitted).
   أنشئ ملفات جديدة + اكتب فوق القائمة + نمط façade لإعادة التصدير؛ لا تعتمد الحذف.
4. **أدوات Edit/Write قد تقطع ملفات المزامنة القائمة** (قُطِع `__init__.py` فعلاً عند
   الكتابة فوقه). الأأمن للملفات الموجودة: الكتابة عبر **bash** (cat heredoc أو python)
   ثم التحقّق: `py_compile` + `ast.parse` + `wc -l` + عدّ null bytes. (الملفات الجديدة
   عبر Write نجحت بلا قطع.)
5. **بيئة Claude المعزولة بلا إنترنت وبلا scipy/sklearn/plotly/pytest**؛ التحقّق الكامل
   على جهازك/CI. للتحقّق offline: shim بـ numpy، أو تحليل `symtable`/`ast` للأسماء غير
   المُحلّاة (يكشف NameError بلا تشغيل) + py_compile.
6. **تحرير docx الموجود:** فُكّ عبر `scripts/office/unpack.py`، عدّل XML (العناوين تنسيق
   مباشر RTL: pBdr + bidi + bold لون 1F4E79 حجم 34 + rtl؛ النقاط نمط `ListBullet`؛
   الجداول `TableGrid` + bidiVisual + jc=right، رأس fill `1F4E79`/أبيض، بيانات fill
   `EEF3F8`)، ثم أعد الحزم عبر `scripts/office/pack.py --original` للتحقّق. حذار:
   `<w:trPr>` يطابق بادئة `<w:tr`، فاستخدم `<w:tr[ >]` عند تحديد بداية صف.
7. المصادقة لـ git عبر Git Credential Manager (PAT بصلاحيات repo + workflow)؛ شغّل git
   من مجلد المشروع. (أول `git push` قد يفشل ثم ينجح بعد مطالبة الاعتماد.)

## مراجع داخل المشروع (اقرأها أولاً)
`ROADMAP.md` ← `MODELING_PHILOSOPHY.md` ← `EVALUATION_PHILOSOPHY.md` ← `CHANGELOG.md`
← `dextra_Evaluation_Report.docx` (v1.5) ← `src/dextra/modeling.py` و`evaluation.py`
كنماذج للعقد، وبنية `src/dextra/_features_*.py`.


