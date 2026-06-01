# dextra — Resume Prompt (انسخ هذا في محادثة جديدة)

أنت تتعاون معي على مكتبة **dextra** لتحليل البيانات وعلم البيانات.
دقة المعلومات والكود مطلوبة 100%، ونلتزم بـ"التراكم الهادئ": أساس مُختبَر وموثّق قبل أي توسّع.

## أين نحن الآن
- المشروع: `D:\06 PythonProjects\dextra-project` — مرفوع على GitHub
  (`ahmedabdeltawab602-collab/dextra`)، و**CI أخضر** على مصفوفة Python 3.9–3.12.
- بيئة التطوير الفعلية: Python 3.13، venv، pandas 3.0.3، numpy 2.4.x،
  scipy، scikit-learn 1.8.0. التفعيل:
  `cd "D:\06 PythonProjects\dextra-project"; .\.venv\Scripts\Activate.ps1`
- إجمالي ما أُنجز ومُختبَر (90 اختبار pytest، تغطية ~68%، lint نظيف):
  - **Phases 1–5 مكتملة** (48 دالة عامة): stats/plots, stats_advanced (22),
    cleaning (10), features (8), selection (5).
  - **Phase 5.5 (سبرنت تثبيت):** نقل سكربتات Phases 2–5 إلى `tests/legacy/`
    وتشغيلها عبر pytest في CI؛ مزامنة README/CHANGELOG/ROADMAP؛ إصلاحات
    IMPROVEMENTS (modes، bins تلقائية، ddof، `dx.functions()`)؛ ضبط ruff.
  - **Phase 6.1 — `regress`** (alias `reg`): linear/ridge/lasso/tree/forest +
    compare. مثبّتة (CI أخضر).
  - **Phase 6.2 — `classify`** (alias `clf`): logistic/tree/forest/knn +
    compare، ثنائي + متعدّد الفئات، accuracy/F1/ROC-AUC. مثبّتة (CI أخضر).

## المطلوب التالي: Phase 6.3 — `cluster` (آخر مرحلة في Phase 6)
- الاسم: `cluster` (alias `clus`)، أوضاع fit/apply/compare، **بلا هدف y**.
- الخوارزميات (compare): kmeans، agglomerative + اختيار k (silhouette / elbow).
- المقاييس: `{"fit": {"silhouette": ..., "inertia": ..., "n_clusters": k}}`
  (بنفس بنية `{split:{metric}}` ونفس `_metrics_table`).
- pred_col = `"cluster"` (تسمية العنقود). أضف مساعِداً `_clean_x` (بلا هدف).

## العقد الموحّد (مُقفل — لا تنشئ "جزيرة معمارية" جديدة)
موثّق في `MODELING_PHILOSOPHY.md §4.7`. كل دوال Phase 6 تتشارك:
1. **fit/apply/compare**: fit يضيف عمود تنبّؤ؛ apply يتنبّأ فقط بلا إعادة تدريب
   (آمن ضد التسرّب)؛ compare يقارن بـ CV ولا يكتب شيئاً (ويرفض return_params).
2. **artifact هجين**: `params` = واصف JSON-آمن + `params['estimator']` (Pipeline
   مُدرَّب يُحفظ بـ joblib ويُحقن في sklearn Pipeline). نفس مفاتيح العقد:
   function/task/algorithm/version/fit_at/features/target/hyperparams/metrics/
   pred_col/metadata/estimator.
3. إعادة استخدام المساعِدات الموجودة في `modeling.py`: `_resolve_features`,
   `_clean_xy`, `_metrics_table`, `_fmt_metric`, `_descriptor`, `_ret_pack`,
   `_append_audit`, `_require_sklearn`, `_json_safe_num`, `_finalize_figure`.
4. سطر واحد ← جدول مقاييس + رسم متعدّد اللوحات + جملة Decision + أثر تدقيقي
   في `df.attrs['dextra_audit']`. الأصل لا يتغيّر؛ idempotency في apply.
5. أسماء عامة بلا "_" + alias قصير؛ docstring بنمط NumPy إنجليزي؛ scikit-learn
   كسول (extra `ml`)؛ بلا أي تبعية ثقيلة جديدة.
6. تنسيق المقاييس المفقودة موحّد وآمن: None/NaN ← `-` في الجداول (`_fmt_table`)
   و`n/a` في جمل Decision (`_fmt_metric`). لا تنسيق خام لأي مقياس.

## الانضباط الإلزامي لكل stage
- اختبارات pytest منذ أول commit + **edge cases** (constant/degenerate،
  انتشار NaN). `tests/test_phase6_stage3.py`.
- compare محدود ومستقر (ليس AutoML).
- تحقّق محلي قبل الدفع عبر `run_validation.ps1` (ruff --fix + ruff + pytest+cov).

## تحذيرات تقنية مكتسبة (مهمة جداً)
1. أدوات Edit/Write قد **تقطع** ملفات الـ mount؛ اكتب/عدّل عبر
   `mcp__workspace__bash` (cat >> ... << 'EOF' أو سكربت python داخل bash).
   تحقّق دائماً: `python3 -m py_compile` + `ast.parse` + `wc -l`.
2. طبقة المزامنة قد تُدخِل **null bytes** أو تقتطع ملفات أحياناً. عند ظهور
   "source code string cannot contain null bytes": جرّد البايتات الصفرية
   `open(f,'rb').read().replace(b'\x00',b'')` ثم أعد التحقق، أو استعد الملف
   النظيف من git (`git checkout -- <file>` للملفات غير المعدَّلة هذا الدور).
3. البيئة المعزولة (sandbox) **بلا إنترنت ولا scipy/sklearn/plotly/pytest**؛
   التحقق الكامل يجري على جهاز المستخدم عبر `run_validation.ps1` أو CI.
   للتحقق المنطقي داخل sandbox، استخدم نموذج sklearn وهمي صغير بـ numpy.
4. الدفع لـ GitHub: الرِّيموت `origin` مضبوط؛ المصادقة عبر نافذة Git Credential
   Manager (تبويب Token = لصق Personal Access Token بصلاحيات repo + workflow).
   شغّل أوامر git **من مجلد المشروع** لا من system32.

## مراجع داخل المشروع (اقرأها أولاً بهذا الترتيب)
1. `ROADMAP.md` (الحالة العامة لكل المراحل).
2. `MODELING_PHILOSOPHY.md` — خاصة §4.7 (العقد الرسمي + جدول العائلات).
3. `PHASE6_TEST_MATRIX.md` (تغطية العقد وحالات الحافة لـ 6.1).
4. `src/dextra/modeling.py` (regress + classify كنموذج يُحتذى لـ cluster).
5. `dextra_Evaluation_Report.docx` (تقرير التقييم الحيّ، آخر إصدار 1.1).

## أول إجراء مطلوب منك في المحادثة الجديدة
اقرأ المراجع أعلاه، ثم اعرض تصميم `cluster` المختصر (أسماء + مقاييس + توقيع
موافق للعقد) للموافقة السريعة، ثم نفّذه مرحلياً بنفس انضباط 6.1/6.2:
كود عبر bash ← تصدير في `__init__.py` ← `tests/test_phase6_stage3.py` ←
تحديث ROADMAP/CHANGELOG/MODELING_PHILOSOPHY ← المستخدم يشغّل `run_validation.ps1`
ويدفع ← CI أخضر ← تثبيت Phase 6 بالكامل.
