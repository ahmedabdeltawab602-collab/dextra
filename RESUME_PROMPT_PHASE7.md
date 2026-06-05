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
  (ruff --fix ثم ruff صارم ثم pytest + تغطية، العتبة **68%**، الفعلي ~71%).

## المُنجَز والمُختبَر
- **Phases 1–6 مكتملة (51 دالة عامة + aliases):** stats/plots، stats_advanced (22)،
  cleaning (10)، features (8)، selection (5)، modeling (regress/classify/cluster).
- **سبرنتات رفع الدرجات 1–3 (كلها خضراء):**
  - 1 (الأداء): قبول polars/pyarrow عند المدخل عبر `_ensure_pandas` (extra `perf`)؛
    harness `benchmarks/` + وظيفة CI؛ وتقسيم `features.py` إلى 5 وحدات
    (`_features_common/_numeric/_discretize/_derive/_pipeline`) خلف façade
    `features.py` (re-export، بلا تغيير API).
  - 2 (التوزيع/التوثيق): موقع mkdocs-material منشور على Pages؛ `CITATION.cff`؛
    وظيفة build CI (sdist/wheel + twine check)؛ شارة PyPI؛ عتبة تغطية 68%.
  - 3 (الاتساق): `tests/test_api_contract.py` (تصدير/docstrings/أعلام قياسية/
    55 alias مطابِق)، `tests/test_compat_conformance.py` (clone/pickle/fit→self/repr)،
    و`dx.functions()` مجمّعة حسب phase.
- **dextra.compat:** أغلفة sklearn (`DextraFeaturePipeline/SelectPipeline`,
  `DextraRegressor/Classifier/Clusterer`) تدخل Pipeline/GridSearchCV.
- تقرير التقييم الحيّ: `dextra_Evaluation_Report.docx` (مجلد «شرح بايثون...»)،
  **الإصدار 1.4، الإجمالي 9.4/10**؛ القسم 12 فيه خطة رفع كل بند، القسم 13 ملخّص
  سبرنتات 1–3.

## المؤجَّل عمداً
- **النشر على PyPI**: جاهز عبر `publish.yml` (Trusted Publishing/OIDC). يتطلّب فعلي:
  تفعيل 2FA على PyPI، إضافة pending publisher (project=dextra, repo، workflow=publish.yml,
  environment=pypi)، إنشاء بيئة `pypi` على GitHub، ثم Release بوسم `v0.2.0`. مؤجَّل
  حتى تكتمل المكتبة.
- **سبرنت أداء أعمق**: chunking/out-of-core، تقليل النسخ وتتجيه الحلقات (مثل
  `_kmeans_1d`)، بوابة anti-regression على benchmarks، و`n_jobs`. (الأداء الآن 8.5/10.)

## المطلوب التالي: Phase 7 — `evaluation.py`
تقييم عميق متعدّد المقاييس يستهلك artifact النمذجة الهجين من Phase 6:
`confusion_report`, `roc_pr`, `residual_analysis`, `learning_curves` (+ aliases).
نفس العقد الموحّد: سطر واحد ← جدول مقاييس + رسم متعدّد اللوحات + جملة Decision + أثر
تدقيقي؛ يقبل إمّا (df + y_true + y_pred/أعمدة) أو artifact/مُقدِّر مُدرَّب.

## أول إجراء مطلوب منك في المحادثة الجديدة
1. اقرأ بالترتيب: `ROADMAP.md`، `MODELING_PHILOSOPHY.md` (خاصة §4.7 العقد)،
   `CHANGELOG.md`، وبنية `src/dextra/` (لاحظ features صار حزمة وحدات + façade).
2. اعرض تصميم `EVALUATION_PHILOSOPHY.md` المختصر (المخطّط على نسق ملفات الفلسفة)
   ثم توقيعات الدوال الأربع للموافقة السريعة.
3. نفّذ مرحلياً بنفس الانضباط.

## تحذيرات تقنية مكتسبة (مهمة جداً — سبّبت أعطالاً فعلية)
1. **شغّل `run_validation.ps1` قبل كل `git commit`.** تخطّيه أحمَر CI مرّتين بأخطاء
   أسلوب لا تظهر في الترجمة: ruff **E302** (سطران فارغان قبل كل def)، **I001** (ترتيب
   الاستيرادات)، **F821** (اسم غير معرّف في تعليق نوع بعد جعل الاستيراد كسولاً).
2. **المزامنة تحقن null bytes** في ملفات .py المعدّلة أحياناً. عند «source code string
   cannot contain null bytes»: `open(f,'rb').read().replace(b'\x00',b'')` ثم أعد التحقق.
3. **لا يمكن حذف/إعادة تسمية ملفات على المجلد المتزامن** (Operation not permitted).
   لذا: أنشئ ملفات جديدة + اكتب فوق القائمة + استخدم نمط façade لإعادة التصدير؛ لا
   تعتمد على الحذف أبداً. (ملف `src/dextra/_deltest.txt` بقي ولم أستطع حذفه — مُدرَج
   في .gitignore؛ احذفه يدوياً متى شئت.)
4. أداة Edit قد **تقطع** ملفات المزامنة؛ الأفضل الكتابة عبر bash (cat heredoc أو
   python) ثم التحقّق: `py_compile` + `ast.parse` + `wc -l` + عدّ null bytes.
5. **بيئة Claude المعزولة بلا إنترنت وبلا scipy/sklearn/plotly/pytest**؛ التحقّق
   الكامل يجري على جهازك/CI. للتحقّق offline: shim بـ numpy، أو تحليل `symtable`/`ast`
   لأسماء غير المُحلّاة (يكشف NameError وقت الاستدعاء بلا تشغيل).
6. المصادقة لـ git عبر Git Credential Manager (PAT بصلاحيات repo + workflow)؛ شغّل
   git من مجلد المشروع.

## مراجع داخل المشروع (اقرأها أولاً)
`ROADMAP.md` ← `MODELING_PHILOSOPHY.md` ← `CHANGELOG.md` ←
`dextra_Evaluation_Report.docx` (v1.4) ← `src/dextra/modeling.py` و`compat.py`
كنماذج للعقد، وبنية `src/dextra/_features_*.py`.
