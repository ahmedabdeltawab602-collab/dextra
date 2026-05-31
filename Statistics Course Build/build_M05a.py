import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 05 - Hypothesis Testing"

# ───────── Cover ─────────
d = setup(); fc(d)
titlep(d, module_no=5,
       module_en="Hypothesis Testing",
       module_ar="اختبار الفرضيات",
       subtitle_en="The 5 Steps · H0 vs Ha · One-tail vs Two-tail · Type I and II Errors",
       subtitle_ar="الخمس خطوات · H0 وHa · أحادى وثنائى الذيل · أخطاء النوعَين")
end_m(d, os.path.join(OUT, "M05 - Cover.docx"))

# ============================================================
# L05-01 — Chapter Introduction
# ============================================================
L = Lesson(
    lid="M05-L01", module=5,
    title_en="Chapter Introduction - Hypothesis Testing",
    title_ar="مقدمة الفصل - اختبار الفرضيات",
    objectives_en=[
        "Understand what hypothesis testing is and when to use it.",
        "Connect this module to the previous one (p-values, alpha).",
        "Recognise hypothesis testing as the formal workhorse of business inference.",
    ],
    objectives_ar=[
        "فهم ما هو اختبار الفرضيات ومتى يُستخدم.",
        "ربط هذا الموديول بسابقه (قيم p، ألفا).",
        "إدراك أن اختبار الفرضيات هو الأداة الرسمية للاستدلال فى الأعمال.",
    ],
    sources=["05 Hypothesis Testing/Chapter Introduction.mp4",
             "Slides PDF: pages 65-67",
             "Glossary: Hypothesis Testing"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Hypothesis testing is the formal procedure that combines everything from Module 04 (p-values, alpha, sample size, CLT) into a clear, defensible decision: did our intervention work, or could the result be just chance?")
    ap(d, "اختبار الفرضيات هو الإجراء الرسمى الذى يدمج كل ما فى الموديول السابق (قيم p، ألفا، حجم العينة، CLT) فى قرار واضح يمكن الدفاع عنه: هل عمل تدخّلنا، أم أن النتيجة قد تكون مجرد صدفة؟")
    callout(d, "key",
            "Every A/B test you'll ever run, every clinical trial, every quality-control decision uses the same five-step process you'll learn in this module.",
            "كل اختبار A/B تشغّله، وكل تجربة سريرية، وكل قرار ضبط جودة، يستخدم نفس العملية ذات الخمس خطوات التى ستتعلمها فى هذا الموديول.")
L.section("intro", "1 · The Formal Procedure", "1 · الإجراء الرسمى", s1)

def s2(d):
    callout(d, "example",
            "Vacuum-store warranty case (from the slides): the store wants to know which warranty drives more revenue per customer — none, 5-year, or 7-year. Three sample groups, three sample means, then a hypothesis test: is there a real difference, or just noise?",
            "مثال متجر المكانس (من الشرائح): المتجر يريد أن يعرف أى ضمان يولّد إيراداً أعلى لكل عميل — بدون، 5 سنوات، أو 7 سنوات. ثلاث مجموعات عينية، ثلاثة متوسطات، ثم اختبار فرضيات: هل هناك فرق حقيقى، أم مجرد ضوضاء؟")
    ep(d, "We'll meet this case again in Lessons 06 and 08 when we run full two-tailed and one-tailed tests on it.")
    ap(d, "سنرى هذه الحالة مرة أخرى فى الدرسَين 06 و 08 عند تشغيل اختبارَين كاملَين عليها (ذيلَين وذيل واحد).")
L.section("example", "2 · The Running Example", "2 · المثال المرجعى", s2)

def s3(d):
    ep(d, "This module's 10 lessons:")
    ap(d, "دروس الموديول العشرة:")
    en_(d, "L01-02: Intro + the 5 steps overview.")
    en_(d, "L03: Stating hypotheses (H₀, Hₐ).")
    en_(d, "L04: Setting alpha and confidence.")
    en_(d, "L05-08: Picking and running one-tail vs two-tail tests.")
    en_(d, "L09: Type I and II errors.")
    en_(d, "L10: Reporting results professionally.")
    an_(d, "L01-02: المقدمة + نظرة عامة على الخمس خطوات.")
    an_(d, "L03: صياغة الفرضيات (H₀، Hₐ).")
    an_(d, "L04: ضبط ألفا والثقة.")
    an_(d, "L05-08: اختيار وتشغيل اختبارات ذيل-واحد وذيلَين.")
    an_(d, "L09: أخطاء النوع الأول والثانى.")
    an_(d, "L10: إعداد التقرير باحترافية.")
L.section("concept", "3 · Roadmap of the Module", "3 · خريطة الموديول", s3)

L.add_memory_aid("Hypothesis testing = the formal procedure that turns data into a defensible decision.",
                 "اختبار الفرضيات = الإجراء الرسمى الذى يحوّل البيانات إلى قرار مدعوم.")
L.summary(
    en="Hypothesis testing wraps everything you learned in Module 04 into one ten-step workflow. The same procedure powers A/B testing, clinical trials, quality control, and business experimentation. By the end of this module you will write hypotheses, choose tests, compute p-values, and report results to a non-statistical audience with confidence.",
    ar="اختبار الفرضيات يدمج كل ما تعلمته فى الموديول السابق فى سير عمل واحد. نفس الإجراء يحرّك اختبارات A/B، والتجارب السريرية، وضبط الجودة، وتجارب الأعمال. بنهاية هذا الموديول ستكتب الفرضيات، تختار الاختبار، تحسب قيمة p، وتُبلّغ النتائج لجمهور غير متخصص بثقة."
)
L.render_memory()

L.quiz_q(qid="M05-L01-Q1", qtype="MCQ",
         q_en="Hypothesis testing is best described as:",
         q_ar="اختبار الفرضيات يوصف أفضل بأنه:",
         opts_en=["A guessing game","A formal procedure to decide if an effect is real",
                  "A type of regression","Pure intuition"],
         opts_ar=["لعبة تخمين","إجراء رسمى لتحديد هل التأثير حقيقى","نوع من الانحدار","حدس بحت"],
         ans="B", exp_en="The formal-procedure framing is the right one.",
         exp_ar="صياغة 'إجراء رسمى' هى الصحيحة.")

L.quiz_q(qid="M05-L01-Q2", qtype="MCQ",
         q_en="Which is NOT typically a hypothesis-testing scenario?",
         q_ar="أى مما يلى ليس عادة سيناريو اختبار فرضيات؟",
         opts_en=["A/B testing a website","Clinical trial of a drug",
                  "Quality control on a production line","Calculating monthly KPIs"],
         opts_ar=["اختبار A/B لموقع","تجربة دواء سريرية",
                  "ضبط جودة على خط إنتاج","حساب مؤشرات الأداء الشهرية"],
         ans="D", exp_en="Calculating KPIs is descriptive; testing if they changed is inferential.",
         exp_ar="حساب المؤشرات وصفى؛ اختبار هل تغيّرت استدلالى.")

L.quiz_q(qid="M05-L01-Q3", qtype="TrueFalse",
         q_en="True or False: Hypothesis testing eliminates uncertainty.",
         q_ar="صح أم خطأ: اختبار الفرضيات يلغى عدم اليقين.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="It QUANTIFIES uncertainty (via p and alpha), it doesn't eliminate it.",
         exp_ar="يُقيس عدم اليقين، لا يلغيه.")

L.quiz_q(qid="M05-L01-Q4", qtype="MCQ",
         q_en="The five-step procedure starts with:",
         q_ar="الإجراء الخماسى يبدأ بـ:",
         opts_en=["Calculate p-value","State hypotheses","Set alpha","Pick the test"],
         opts_ar=["احسب قيمة p","صِغ الفرضيات","حدد ألفا","اختر الاختبار"],
         ans="B", exp_en="You can't compute anything until you write H₀ and Hₐ.",
         exp_ar="لا تحسب شيئاً قبل كتابة H₀ و Hₐ.")

L.quiz_q(qid="M05-L01-Q5", qtype="MCQ",
         q_en="In the warranty example, what's the underlying business question?",
         q_ar="فى مثال الضمان، ما السؤال التجارى الأساسى؟",
         opts_en=["How many vacuums to sell",
                  "Which warranty produces a different revenue per customer",
                  "How long warranties last","Price of vacuums"],
         opts_ar=["كم عدد المكانس للبيع","أى ضمان ينتج إيراداً مختلفاً لكل عميل",
                  "مدة الضمانات","سعر المكانس"], ans="B",
         exp_en="Does warranty A vs B vs none drive different revenue per customer?",
         exp_ar="هل الضمان A مقابل B مقابل لا شىء يحرّك إيراداً مختلفاً؟")
L.render_quiz(); L.save()

# ============================================================
# L05-02 — The 5 Steps
# ============================================================
L = Lesson(
    lid="M05-L02", module=5,
    title_en="The 5 Steps of Hypothesis Testing",
    title_ar="الخمس خطوات لاختبار الفرضيات",
    objectives_en=[
        "Memorise the five steps in order.",
        "Understand why the order matters.",
        "Apply the steps to a simple business example.",
    ],
    objectives_ar=[
        "حفظ الخمس خطوات بالترتيب.",
        "فهم لماذا الترتيب مهم.",
        "تطبيق الخطوات على مثال أعمال بسيط.",
    ],
    sources=["05 Hypothesis Testing/Hypothesis Testing Steps.mp4",
             "Slides PDF: page 68"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Every hypothesis test follows the same five steps. Internalise them now and you'll never get lost during a real test.")
    ap(d, "كل اختبار فرضيات يتبع نفس الخمس خطوات. احفظها الآن ولن تضيع أبداً فى اختبار حقيقى.")
L.section("intro", "1 · One Procedure to Rule Them All",
          "1 · إجراء واحد يحكمها كلها", s1)

def s2(d):
    kvtable(d, header=["Step / الخطوة","Action / الفعل","Output / المخرَج"],
        rows=[
            ("1", "State the null and alternative hypotheses.\nصِغ الصفرية والبديلة.",
             "H₀ and Hₐ on paper.\nH₀ و Hₐ مكتوبتان."),
            ("2", "Set the significance level α.\nحدد مستوى الدلالة α.",
             "α value (e.g., 0.05).\nقيمة ألفا."),
            ("3", "Select the test (one-tail / two-tail; t / z / chi).\nاختر الاختبار.",
             "Test type chosen.\nنوع الاختبار محدد."),
            ("4", "Compute the test statistic and p-value.\nاحسب إحصاء الاختبار وقيمة p.",
             "Numerical p-value.\nقيمة p رقمية."),
            ("5", "Compare p to α; reject or fail to reject H₀.\nقارن p بـ α؛ ارفض أو لا ترفض.",
             "Decision + business interpretation.\nقرار + تفسير تجارى."),
        ])
L.section("definitions", "2 · The 5 Steps Table", "2 · جدول الخمس خطوات", s2)

def s3(d):
    callout(d, "example",
            "Mini example: An e-commerce site wants to test if a new shipping policy raises average order value above the historical $80.\n• Step 1: H₀: μ = $80. Hₐ: μ > $80 (one-tail).\n• Step 2: α = 0.05.\n• Step 3: One-sample t-test (population σ unknown).\n• Step 4: From sample (n=200, x̄=$84, s=$15) → t = (84-80)/(15/√200) ≈ 3.77; p ≈ 0.0001.\n• Step 5: p < α → reject H₀. Conclude policy raised AOV.",
            "مثال صغير: موقع تجارة يريد اختبار هل سياسة شحن جديدة ترفع متوسط الطلب فوق 80 دولاراً.\n• الخطوة 1: H₀: μ = 80. Hₐ: μ > 80 (ذيل واحد).\n• الخطوة 2: α = 0.05.\n• الخطوة 3: t-test لعينة واحدة.\n• الخطوة 4: من العينة (n=200, x̄=84, s=15) ← t ≈ 3.77؛ p ≈ 0.0001.\n• الخطوة 5: p < α ← ارفض H₀. الاستنتاج: السياسة رفعت AOV.")
L.section("example", "3 · Worked Example - 5 Steps in 60 Seconds",
          "3 · مثال محلول - 5 خطوات فى 60 ثانية", s3)

def s4(d):
    callout(d, "trap",
            "Doing the steps out of order. Setting α AFTER seeing the p-value is p-hacking. Choosing one-tail AFTER seeing the direction is data-snooping. Pre-register the design.",
            "تنفيذ الخطوات خارج الترتيب. ضبط ألفا بعد رؤية p تلاعب. اختيار ذيل واحد بعد رؤية الاتجاه تجسس على البيانات. سجّل التصميم مسبقاً.")
L.section("traps", "4 · The Order Matters", "4 · الترتيب يهمّ", s4)

def s5(d):
    callout(d, "use",
            "Save this five-step checklist on a sticky note. Whether you're testing a new email subject line, a new pricing model, or a new HR policy — these are the steps. The math may be done by Excel or Python; the steps are yours.",
            "احفظ قائمة الخمس خطوات على ملصق. سواء اختبرت عنوان بريد، نموذج تسعير، سياسة موارد بشرية — هذه هى الخطوات. الرياضيات تنفّذها Excel أو Python؛ الخطوات لك.")
L.section("practice", "5 · The Sticky-Note Checklist",
          "5 · قائمة الملصق", s5)

L.add_memory_aid("STATE → SET → SELECT → COMPUTE → COMPARE.",
                 "صِغ ← حدد ← اختر ← احسب ← قارن.")
L.summary(
    en="Five steps, always in this order: state hypotheses; set alpha; select the test; compute the statistic and p-value; compare to alpha and decide. The procedure is universal across A/B tests, clinical trials, quality control, and any business experiment. Master the order before you worry about the math.",
    ar="خمس خطوات بهذا الترتيب دائماً: صِغ الفرضيات؛ حدد ألفا؛ اختر الاختبار؛ احسب الإحصاء وقيمة p؛ قارن بألفا وقرر. الإجراء عالمى عبر اختبارات A/B والتجارب السريرية وضبط الجودة وأى تجربة أعمال. أتقن الترتيب قبل الرياضيات."
)
L.render_memory()

L.quiz_q(qid="M05-L02-Q1", qtype="MCQ",
         q_en="The first step in hypothesis testing is:",
         q_ar="الخطوة الأولى فى اختبار الفرضيات:",
         opts_en=["Compute p-value","Set alpha","State H0 and Ha","Pick the test"],
         opts_ar=["احسب قيمة p","حدد ألفا","صِغ H0 و Ha","اختر الاختبار"], ans="C",
         exp_en="State hypotheses first.", exp_ar="صياغة الفرضيات أولاً.")

L.quiz_q(qid="M05-L02-Q2", qtype="MCQ",
         q_en="The final step is:",
         q_ar="الخطوة الأخيرة:",
         opts_en=["Set alpha","Compute p-value","Compare p to alpha and decide","Submit to a journal"],
         opts_ar=["حدد ألفا","احسب قيمة p","قارن p بألفا وقرر","قدّمها لمجلة"], ans="C",
         exp_en="The decision is the output.", exp_ar="القرار هو المخرج.")

L.quiz_q(qid="M05-L02-Q3", qtype="TrueFalse",
         q_en="True or False: Alpha should be chosen after looking at the p-value.",
         q_ar="صح أم خطأ: ألفا يُختار بعد النظر إلى قيمة p.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Pre-commit to alpha. Otherwise it's p-hacking.",
         exp_ar="ثبّت ألفا مسبقاً. وإلا فهو تلاعب.")

L.quiz_q(qid="M05-L02-Q4", qtype="MCQ",
         q_en="Step 4 produces:",
         q_ar="الخطوة 4 تنتج:",
         opts_en=["A hypothesis","A test choice","A test statistic and p-value","A confidence interval only"],
         opts_ar=["فرضية","اختياراً للاختبار","إحصاء اختبار وقيمة p","فاصل ثقة فقط"], ans="C",
         exp_en="Compute the statistic and p-value before deciding.",
         exp_ar="احسب الإحصاء وقيمة p قبل القرار.")

L.quiz_q(qid="M05-L02-Q5", qtype="MCQ",
         q_en="If p = 0.03 and alpha = 0.05, your decision is:",
         q_ar="إذا p = 0.03 و ألفا = 0.05، قرارك:",
         opts_en=["Reject H0","Fail to reject H0","Re-run the test","Set new alpha"],
         opts_ar=["ارفض H0","لا ترفض H0","أعد التشغيل","ضع ألفا جديد"], ans="A",
         exp_en="p < alpha → reject H0.", exp_ar="p < ألفا ← ارفض H0.")
L.render_quiz(); L.save()

# ============================================================
# L05-03 — Stating the Hypothesis
# ============================================================
L = Lesson(
    lid="M05-L03", module=5,
    title_en="Stating the Hypothesis",
    title_ar="صياغة الفرضيات",
    objectives_en=[
        "Write null (H0) and alternative (Ha) hypotheses correctly.",
        "Recognise that we test H0, not Ha.",
        "Translate a business question into a statistical hypothesis pair.",
    ],
    objectives_ar=[
        "كتابة الفرضية الصفرية (H0) والبديلة (Ha) بشكل صحيح.",
        "إدراك أننا نختبر H0، لا Ha.",
        "ترجمة سؤال أعمال إلى زوج فرضيات إحصائى.",
    ],
    sources=["05 Hypothesis Testing/Stating the Hypothesis.mp4",
             "Slides PDF: page 69",
             "Glossary: Null Hypothesis, Alternative hypothesis"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Every hypothesis test has TWO statements: the null (H0) and the alternative (Ha). They must be mutually exclusive and together cover all possibilities.")
    ap(d, "كل اختبار له بيانان: الصفرية (H0) والبديلة (Ha). يجب أن يكونا متعارضَين تماماً ومعاً يغطيان كل الاحتمالات.")
    callout(d, "key",
            "H0 = the no-effect / no-difference position. The 'innocent until proven guilty' assumption. We test whether data forces us to reject it.",
            "H0 = موقف 'لا تأثير' / 'لا فرق'. افتراض 'برىء حتى تثبت إدانته'. نختبر هل تجبرنا البيانات على رفضه.")
L.section("intro", "1 · Two Statements, Always",
          "1 · بيانان دائماً", s1)

def s2(d):
    kvtable(d, header=["Hypothesis / الفرضية","Symbol / الرمز","Says / يقول","Example / مثال"],
        rows=[
            ("Null\nالصفرية","H0","No difference / no effect.\nلا فرق / لا تأثير.",
             "μ_A = μ_B (mean revenue is equal).\nμ_A = μ_B (متوسط الإيراد متساوٍ)."),
            ("Alternative\nالبديلة","Ha or H1","There IS a difference / effect.\nيوجد فرق / تأثير.",
             "μ_A ≠ μ_B (or > or <).\nμ_A ≠ μ_B (أو > أو <)."),
        ])
L.section("definitions", "2 · The H0 / Ha Pair", "2 · زوج H0 / Ha", s2)

def s3(d):
    callout(d, "example",
            "Warranty case from the slides:\n• H0: μ_5y = μ_7y (both warranties drive the same revenue per visitor).\n• Ha: μ_5y ≠ μ_7y (they differ).\nThis is a two-tailed (≠) test. We don't yet know which warranty is better; we just suspect they're different.",
            "حالة الضمان من الشرائح:\n• H0: μ_5سنة = μ_7سنة (كلا الضمانَين يولّد نفس الإيراد لكل زائر).\n• Ha: μ_5سنة ≠ μ_7سنة (مختلفان).\nهذا اختبار ذيلَين (≠). لا نعرف بعد أيهما أفضل؛ فقط نشكّ فى الاختلاف.")
L.section("example", "3 · Worked Example - Warranty Hypotheses",
          "3 · مثال محلول - فرضيات الضمان", s3)

def s4(d):
    ep(d, "Three flavours of Ha:")
    ap(d, "ثلاثة أنواع من Ha:")
    eb(d, "Two-sided (≠): looking for any difference.")
    eb(d, "Right-sided (>): expecting an increase.")
    eb(d, "Left-sided (<): expecting a decrease.")
    ab(d, "ثنائى الجانب (≠): نبحث عن أى اختلاف.")
    ab(d, "موجب الجانب (>): نتوقع زيادة.")
    ab(d, "سالب الجانب (<): نتوقع انخفاضاً.")
    callout(d, "meaning",
            "Pick the direction BEFORE looking at the data, based on the business question. Most beginner tests are two-sided unless there's a clear directional belief.",
            "اختر الاتجاه قبل النظر للبيانات، بناءً على السؤال التجارى. معظم اختبارات المبتدئين ذات الجانبَين ما لم يكن هناك اتجاه واضح.")
L.section("concept", "4 · Direction of the Alternative",
          "4 · اتجاه البديلة", s4)

def s5(d):
    callout(d, "trap",
            "Writing Ha as 'H0 is wrong'. Be specific: 'mean increased' or 'proportion of churn fell to less than 10%'.",
            "كتابة Ha كـ 'H0 خاطئة'. كن محدداً: 'المتوسط ارتفع' أو 'نسبة المغادرة انخفضت إلى أقل من 10%'.")
    callout(d, "trap",
            "Putting the effect we hope to find IN H0. H0 is always the boring, no-effect version.",
            "وضع التأثير الذى نأمل إيجاده فى H0. H0 دائماً النسخة المملة، لا تأثير فيها.")
L.section("traps", "5 · Common Traps", "5 · فخاخ شائعة", s5)

def s6(d):
    kvtable(d, header=["Business question / سؤال أعمال","H0","Ha"],
        rows=[
            ("Did the new email subject raise open rate above 18%?\nهل رفع العنوان الجديد معدل الفتح فوق 18%؟",
             "p = 0.18","p > 0.18"),
            ("Are average load times of v2 and v1 different?\nهل أوقات تحميل v2 و v1 مختلفة؟",
             "μ_v2 = μ_v1","μ_v2 ≠ μ_v1"),
            ("Did the price increase reduce monthly orders?\nهل خفّض رفع السعر الطلبات الشهرية؟",
             "μ_after = μ_before","μ_after < μ_before"),
        ])
L.section("practice", "6 · Business Examples", "6 · أمثلة أعمال", s6)

L.add_memory_aid("H0 = no effect. Ha = the change we're testing for.",
                 "H0 = لا تأثير. Ha = التغيير الذى نختبره.")
L.add_memory_aid("Pre-commit to direction. Don't switch one-tail after seeing data.",
                 "ثبّت الاتجاه مسبقاً. لا تغيّر إلى ذيل واحد بعد رؤية البيانات.")
L.summary(
    en="The null hypothesis (H0) is the default 'no effect' statement. The alternative (Ha) is what you would conclude if H0 is rejected. Three forms of Ha exist: ≠ (two-tailed), > (right-tailed), < (left-tailed). Always pick H0/Ha — and the direction — BEFORE looking at the data. Writing them clearly is half the battle of a clean hypothesis test.",
    ar="الفرضية الصفرية (H0) هى البيان الافتراضى 'لا تأثير'. البديلة (Ha) هى ما ستستنتجه إذا رُفضت H0. ثلاثة أشكال لـ Ha: ≠ (ذيلَين)، > (ذيل أيمن)، < (ذيل أيسر). اختر دائماً H0/Ha — والاتجاه — قبل النظر للبيانات. كتابتها بوضوح نصف المعركة."
)
L.render_memory()

L.quiz_q(qid="M05-L03-Q1", qtype="MCQ",
         q_en="The null hypothesis typically states:",
         q_ar="الفرضية الصفرية تذكر عادة:",
         opts_en=["The effect we hope to find","No effect / no difference",
                  "A strong opinion","An uncertain guess"],
         opts_ar=["التأثير الذى نأمل إيجاده","لا تأثير / لا فرق",
                  "رأى قوى","تخمين"], ans="B", exp_en="H0 is the boring default.",
         exp_ar="H0 الافتراض المملّ.")

L.quiz_q(qid="M05-L03-Q2", qtype="MCQ",
         q_en="For 'Did the new homepage increase signups above 5%?', Ha is:",
         q_ar="لـ 'هل رفعت الصفحة الجديدة التسجيلات فوق 5%؟'، Ha:",
         opts_en=["p = 0.05","p ≠ 0.05","p > 0.05","p < 0.05"],
         opts_ar=["p = 0.05","p ≠ 0.05","p > 0.05","p < 0.05"], ans="C",
         exp_en="'Above' = right-sided test.", exp_ar="'فوق' = اختبار جانب أيمن.")

L.quiz_q(qid="M05-L03-Q3", qtype="MCQ",
         q_en="For 'Are revenues of A and B different?', Ha is:",
         q_ar="لـ 'هل إيرادات A و B مختلفة؟'، Ha:",
         opts_en=["μ_A = μ_B","μ_A ≠ μ_B","μ_A > μ_B","μ_A < μ_B"],
         opts_ar=["μ_A = μ_B","μ_A ≠ μ_B","μ_A > μ_B","μ_A < μ_B"], ans="B",
         exp_en="'Different' = two-tailed ≠.", exp_ar="'مختلف' = ذيلَين ≠.")

L.quiz_q(qid="M05-L03-Q4", qtype="TrueFalse",
         q_en="True or False: You should pick the test direction (one-tail / two-tail) after looking at the data.",
         q_ar="صح أم خطأ: اختيار اتجاه الاختبار بعد النظر للبيانات.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Pre-register the direction. After-the-fact picking inflates false positives.",
         exp_ar="سجّل الاتجاه مسبقاً. الاختيار اللاحق يضخّم الإيجابيات الكاذبة.")

L.quiz_q(qid="M05-L03-Q5", qtype="MCQ",
         q_en="A pharmaceutical company tests if their drug LOWERS blood pressure. The right Ha is:",
         q_ar="شركة دواء تختبر هل دواؤها يخفّض ضغط الدم. Ha الصحيحة:",
         opts_en=["μ_after = μ_before","μ_after > μ_before",
                  "μ_after < μ_before","μ_after ≠ μ_before"],
         opts_ar=["μ_after = μ_before","μ_after > μ_before",
                  "μ_after < μ_before","μ_after ≠ μ_before"], ans="C",
         exp_en="'Lowers' = left-sided.", exp_ar="'يخفض' = جانب أيسر.")
L.render_quiz(); L.save()

print("L05-01, L05-02, L05-03 done.")
