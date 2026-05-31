import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 05 - Hypothesis Testing"

# ============================================================
# L05-04 — Setting Alpha & Confidence Level
# ============================================================
L = Lesson(
    lid="M05-L04", module=5,
    title_en="Setting the Alpha and Confidence Level",
    title_ar="ضبط ألفا ومستوى الثقة",
    objectives_en=[
        "Choose an alpha appropriate to the business stakes.",
        "Connect alpha to Type I error and confidence level.",
        "Adjust alpha for sample-size trade-offs.",
    ],
    objectives_ar=[
        "اختيار ألفا مناسب لأهمية القرار التجارى.",
        "ربط ألفا بالخطأ من النوع الأول ومستوى الثقة.",
        "تعديل ألفا لمقايضات حجم العينة.",
    ],
    sources=["05 Hypothesis Testing/Setting the Alpha & Confidence Level.mp4",
             "Slides PDF: page 70"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Alpha is the threshold for rejecting H0. Setting it BEFORE looking at the data is the rule that keeps statistics honest. Confidence level = 1 − alpha.")
    ap(d, "ألفا هو حد رفض H0. ضبطه قبل النظر للبيانات هو القاعدة التى تبقى الإحصاء أميناً. مستوى الثقة = 1 − ألفا.")
L.section("intro", "1 · The Pre-Commitment", "1 · الالتزام المسبق", s1)

def s2(d):
    eq(d, "Confidence level  =  1 − α")
    kvtable(d, header=["Confidence / الثقة","α / ألفا","False conclusion / استنتاج خاطئ","Trade-off / المقايضة"],
        rows=[
            ("95%","0.05","1 in 20","Bigger sample.\nعينة أكبر."),
            ("90%","0.10","1 in 10","Smaller sample, save time and money.\nعينة أصغر، توفّر الوقت والمال."),
            ("99%","0.01","1 in 100","Much bigger sample needed.\nعينة أكبر بكثير."),
        ])
    callout(d, "meaning",
            "Confidence level sets the probability of correctly FAILING TO REJECT a true H0. α sets the probability of incorrectly REJECTING a true H0 (Type I error).",
            "مستوى الثقة يحدد احتمال 'عدم الرفض' الصحيح لـ H0 صحيحة. α يحدد احتمال 'الرفض' الخاطئ لـ H0 صحيحة (الخطأ من النوع الأول).")
L.section("definitions", "2 · The Trade-off Table",
          "2 · جدول المقايضة", s2)

def s3(d):
    callout(d, "example",
            "Warranty A/B test:\n• α = 0.05 → 95% confidence. We accept a 5% chance of falsely declaring a difference if there isn't one.\n• That means 1 false positive every 20 tests, ON AVERAGE.\nFor a 'try it for the lulz' marketing test, that's fine. For a million-dollar product launch decision, you'd want α = 0.01 (1 in 100).",
            "اختبار A/B للضمان:\n• ألفا = 0.05 ← ثقة 95%. نقبل احتمال 5% للإعلان الخاطئ عن وجود فرق إن لم يكن موجوداً.\n• إيجابى كاذب واحد كل 20 اختباراً فى المتوسط.\nلاختبار تسويقى عابر هذا مقبول. لقرار إطلاق منتج بمليون دولار، تريد ألفا = 0.01 (1 من 100).")
L.section("example", "3 · Worked Example - Picking Alpha by Stakes",
          "3 · مثال محلول - اختيار ألفا حسب الأهمية", s3)

def s4(d):
    callout(d, "use",
            "Marketing A/B: α = 0.05 (industry default). Fast iterations, low cost per test.",
            "اختبارات A/B تسويقية: ألفا = 0.05 (افتراضى صناعى). تكرارات سريعة، تكلفة منخفضة.")
    callout(d, "use",
            "Drug approval: α = 0.01 or stricter. Cost of false positive = patients on ineffective/harmful drug.",
            "موافقة دواء: ألفا = 0.01 أو أصرم. تكلفة الإيجابى الكاذب = مرضى على دواء غير فعال/ضار.")
    callout(d, "use",
            "Pre-launch product feature: α = 0.01. Cost of false positive = rolling out a feature that doesn't help.",
            "خاصية منتج قبل الإطلاق: ألفا = 0.01. تكلفة الإيجابى الكاذب = إطلاق خاصية لا تنفع.")
L.section("practice", "4 · Choosing Alpha in Real Projects",
          "4 · اختيار ألفا فى مشاريع حقيقية", s4)

def s5(d):
    callout(d, "trap",
            "'Let's just use 0.05 because everyone does'. A 5% false-positive rate compounds over many tests. If you run 20 A/B tests at α=0.05 with no real effect, you'll get about 1 'significant' result anyway. Adjust for multiple comparisons (e.g., Bonferroni: divide α by number of tests).",
            "'لنستخدم 0.05 لأن الكل يفعل'. معدل إيجابى كاذب 5% يتراكم عبر اختبارات كثيرة. إذا شغّلت 20 اختبار A/B بـ ألفا=0.05 دون تأثير حقيقى، ستحصل على ~1 'ذو دلالة' عشوائياً. عدّل للمقارنات المتعددة (مثل Bonferroni: اقسم ألفا على عدد الاختبارات).")
L.section("traps", "5 · The Multiple-Testing Trap",
          "5 · فخ الاختبارات المتعددة", s5)

L.add_memory_aid("Alpha = 1 − Confidence. Both are pre-set; never tuned after p-value.",
                 "ألفا = 1 − الثقة. كلاهما مسبق؛ لا يُضبط بعد قيمة p.")
L.add_memory_aid("0.05 default, 0.01 for high-stakes, 0.10 for quick exploration.",
                 "0.05 افتراضى، 0.01 للحرجة، 0.10 للاستكشاف السريع.")
L.summary(
    en="Alpha is the line in the sand: how unlikely must the data be (under H0) before we reject H0? Confidence level = 1 − alpha. Pick alpha based on the cost of a false positive: 0.05 for everyday decisions, 0.01 for high-stakes, 0.10 for quick exploration. Always pre-commit and beware multiple-testing inflation.",
    ar="ألفا هو الخط الفاصل: كم يجب أن تكون البيانات غير محتملة (تحت H0) قبل أن نرفض H0؟ مستوى الثقة = 1 − ألفا. اختر ألفا حسب تكلفة الإيجابى الكاذب: 0.05 للقرارات اليومية، 0.01 للحرجة، 0.10 للاستكشاف. ثبّت دائماً مسبقاً وانتبه لتضخّم الاختبارات المتعددة."
)
L.render_memory()

L.quiz_q(qid="M05-L04-Q1", qtype="Calc",
         q_en="Alpha = 0.10. Confidence level is:",
         q_ar="ألفا = 0.10. مستوى الثقة:",
         opts_en=["10%","90%","99%","50%"], opts_ar=["10%","90%","99%","50%"], ans="B",
         exp_en="1 − 0.10 = 0.90 = 90%.", exp_ar="1 − 0.10 = 0.90 = 90%.")

L.quiz_q(qid="M05-L04-Q2", qtype="MCQ",
         q_en="High-stakes decisions (e.g., drug safety) typically use:",
         q_ar="قرارات حرجة (مثل سلامة الدواء) تستخدم عادة:",
         opts_en=["α=0.10","α=0.05","α=0.01 or stricter","No alpha"],
         opts_ar=["ألفا=0.10","ألفا=0.05","ألفا=0.01 أو أصرم","لا ألفا"], ans="C",
         exp_en="Lower alpha = fewer false positives.",
         exp_ar="ألفا أقل = إيجابيات كاذبة أقل.")

L.quiz_q(qid="M05-L04-Q3", qtype="TrueFalse",
         q_en="True or False: Running 20 independent tests at α=0.05 produces about 1 false positive on average.",
         q_ar="صح أم خطأ: تشغيل 20 اختباراً مستقلاً بـ ألفا=0.05 ينتج ~1 إيجابى كاذب فى المتوسط.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="20 × 0.05 = 1.", exp_ar="20 × 0.05 = 1.")

L.quiz_q(qid="M05-L04-Q4", qtype="MCQ",
         q_en="Lowering alpha from 0.05 to 0.01:",
         q_ar="خفض ألفا من 0.05 إلى 0.01:",
         opts_en=["Increases Type I error","Decreases Type I error",
                  "Has no effect","Increases sample mean"],
         opts_ar=["يزيد الخطأ من النوع الأول","يقلله","لا أثر","يزيد متوسط العينة"],
         ans="B", exp_en="Lower α = fewer false positives.", exp_ar="ألفا أقل = إيجابيات كاذبة أقل.")

L.quiz_q(qid="M05-L04-Q5", qtype="MCQ",
         q_en="The trade-off of a much smaller alpha is:",
         q_ar="مقايضة ألفا أصغر بكثير:",
         opts_en=["Smaller sample needed","Same sample",
                  "Larger sample needed to detect a real effect","No trade-off"],
         opts_ar=["عينة أصغر","نفس العينة","عينة أكبر لكشف تأثير حقيقى","لا مقايضة"],
         ans="C", exp_en="Smaller α with same power → larger n.",
         exp_ar="ألفا أصغر مع نفس القوة ← n أكبر.")
L.render_quiz(); L.save()

# ============================================================
# L05-05 — Selecting the Hypothesis Test
# ============================================================
L = Lesson(
    lid="M05-L05", module=5,
    title_en="Selecting the Hypothesis Test",
    title_ar="اختيار الاختبار",
    objectives_en=[
        "Distinguish two-tailed from one-tailed tests.",
        "Know when each is appropriate.",
        "Visualise the rejection regions for each.",
    ],
    objectives_ar=[
        "التمييز بين اختبار ذيلَين وذيل واحد.",
        "معرفة متى يناسب كل منهما.",
        "تصوّر مناطق الرفض لكل منهما.",
    ],
    sources=["05 Hypothesis Testing/Selecting the Hypothesis Test.mp4",
             "Slides PDF: pages 71-72"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Once H0 and Ha are written and alpha is set, you must pick the test SHAPE. Are we looking for any difference (two-tailed), or for a specific direction (one-tailed)?")
    ap(d, "بعد كتابة H0 و Ha وضبط ألفا، يجب اختيار شكل الاختبار. هل نبحث عن أى فرق (ذيلَين)، أم اتجاه محدد (ذيل واحد)؟")
L.section("intro", "1 · Two-Tail or One-Tail?",
          "1 · ذيلَين أم ذيل واحد؟", s1)

def s2(d):
    kvtable(d, header=["Test / الاختبار","Ha shape / شكل Ha","Rejection region / منطقة الرفض","Use when / استخدمه عندما"],
        rows=[
            ("Two-tailed\nذيلَين","≠","Both tails (α/2 each).\nكلا الذيلَين (ألفا/2 لكل).",
             "Looking for ANY difference.\nنبحث عن أى فرق."),
            ("Right-tailed\nذيل أيمن",">","Right tail only.\nالذيل الأيمن فقط.",
             "Expect increase.\nنتوقع زيادة."),
            ("Left-tailed\nذيل أيسر","<","Left tail only.\nالذيل الأيسر فقط.",
             "Expect decrease.\nنتوقع انخفاضاً."),
        ])
    callout(d, "key",
            "Two-tailed = safer / more conservative. One-tailed = more statistical power, but only valid if you committed to a direction BEFORE seeing data.",
            "ذيلَين = أكثر أماناً / تحفّظاً. ذيل واحد = قوة إحصائية أعلى، صحيح فقط إذا التزمت بالاتجاه قبل البيانات.")
L.section("definitions", "2 · The Three Shapes",
          "2 · الأشكال الثلاثة", s2)

def s3(d):
    callout(d, "example",
            "Warranty case revisited:\n• Two-tail Ha: μ_5y ≠ μ_7y → check both tails. α=0.05 → 0.025 in each tail.\n• One-tail Ha: μ_5y > μ_no (the 5-year warranty BEATS no warranty) → α=0.05 in the right tail only.\nOne-tail is more sensitive but only works if we genuinely expected 'better' before the test.",
            "حالة الضمان:\n• ذيلَين Ha: μ_5سنة ≠ μ_7سنة ← افحص كلا الذيلَين. ألفا=0.05 ← 0.025 فى كل ذيل.\n• ذيل واحد Ha: μ_5سنة > μ_لا (الضمان 5 سنوات يتفوق على عدم الضمان) ← ألفا=0.05 فى الذيل الأيمن فقط.\nالذيل الواحد أكثر حساسية لكنه صحيح فقط إذا توقّعنا 'أفضل' قبل الاختبار.")
L.section("example", "3 · Worked Example - The Warranty",
          "3 · مثال محلول - الضمان", s3)

def s4(d):
    callout(d, "trap",
            "Switching to one-tailed AFTER seeing the data goes in the right direction. This doubles your false-positive rate. Reviewers and auditors will catch it.",
            "التحويل لذيل واحد بعد رؤية الاتجاه المرغوب. يضاعف معدل الإيجابيات الكاذبة. المراجعون يكتشفون ذلك.")
    callout(d, "trap",
            "Using one-tailed because 'I really hope it's positive'. Hope is not a statistical argument. Pre-register based on plausible business hypothesis.",
            "استخدام ذيل واحد لأن 'أتمنى أن يكون موجباً'. الأمل ليس حجة إحصائية. سجّل مسبقاً بناءً على فرضية أعمال معقولة.")
L.section("traps", "4 · Common Traps", "4 · فخاخ شائعة", s4)

def s5(d):
    callout(d, "use",
            "Most online A/B tests use TWO-TAILED tests because you genuinely don't know which variant wins. Save one-tailed for: comparisons against a fixed baseline you must beat, regulatory thresholds, or directional product hypotheses.",
            "معظم اختبارات A/B عبر الإنترنت تستخدم ذيلَين لأنك لا تعرف أى نسخة ستفوز. استخدم ذيلاً واحداً لـ: مقارنات بأساس ثابت يجب تجاوزه، حدود تنظيمية، أو فرضيات اتجاهية للمنتج.")
L.section("practice", "5 · Real Decision Patterns",
          "5 · أنماط قرار حقيقية", s5)

L.add_memory_aid("Two-tail = any difference. One-tail = specific direction. Pre-commit always.",
                 "ذيلَين = أى فرق. ذيل واحد = اتجاه محدد. التزم مسبقاً دائماً.")
L.summary(
    en="Two-tailed tests check for any difference; one-tailed tests check for a specific direction. The choice MUST be made before looking at the data — afterwards is data-snooping. Two-tailed is the safe default for exploratory work; one-tailed is more powerful when you have a strong directional hypothesis (and the cost of being wrong in the other direction is acceptable).",
    ar="اختبار ذيلَين يفحص أى فرق؛ ذيل واحد يفحص اتجاهاً محدداً. الاختيار يجب قبل البيانات — بعدها تجسس. ذيلَين الافتراضى الآمن للاستكشاف؛ ذيل واحد أقوى عندما توجد فرضية اتجاهية قوية (وتكلفة الخطأ فى الاتجاه الآخر مقبولة)."
)
L.render_memory()

L.quiz_q(qid="M05-L05-Q1", qtype="MCQ",
         q_en="A two-tailed test rejects H0 if the test statistic falls in:",
         q_ar="اختبار ذيلَين يرفض H0 إذا وقع إحصاء الاختبار فى:",
         opts_en=["Right tail only","Left tail only","Either tail","Neither"],
         opts_ar=["الأيمن فقط","الأيسر فقط","أى منهما","لا منهما"], ans="C",
         exp_en="Two tails: extreme either direction.", exp_ar="ذيلَين: متطرف فى أى اتجاه.")

L.quiz_q(qid="M05-L05-Q2", qtype="MCQ",
         q_en="An online store wants to test if a new banner increases (specifically increases) click-through. Best test:",
         q_ar="متجر يريد اختبار هل لافتة جديدة تزيد (تحديداً تزيد) النقرات. الأفضل:",
         opts_en=["Two-tailed","Right-tailed (one-tailed)","Left-tailed","No test"],
         opts_ar=["ذيلَين","ذيل أيمن (واحد)","ذيل أيسر","لا اختبار"], ans="B",
         exp_en="Directional hope of increase → one-tailed right.",
         exp_ar="اتجاه متوقع للزيادة ← ذيل أيمن.")

L.quiz_q(qid="M05-L05-Q3", qtype="TrueFalse",
         q_en="True or False: A one-tailed test has higher statistical power than two-tailed at the same alpha.",
         q_ar="صح أم خطأ: اختبار ذيل واحد له قوة أعلى من ذيلَين عند نفس ألفا.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="All alpha in one tail = easier to reach significance there.",
         exp_ar="كل ألفا فى ذيل واحد = أسهل بلوغ الدلالة هناك.")

L.quiz_q(qid="M05-L05-Q4", qtype="MCQ",
         q_en="The biggest risk of one-tailed tests is:",
         q_ar="أكبر مخاطر اختبار ذيل واحد:",
         opts_en=["More math","Choosing direction after seeing data (data-snooping)",
                  "Worse names","Excel can't do them"],
         opts_ar=["رياضيات أكثر","اختيار الاتجاه بعد رؤية البيانات (تجسس)",
                  "أسماء أسوأ","Excel لا يستطيع"], ans="B",
         exp_en="Post-hoc direction inflates false positives.",
         exp_ar="الاتجاه اللاحق يضخّم الإيجابيات الكاذبة.")

L.quiz_q(qid="M05-L05-Q5", qtype="MCQ",
         q_en="For 'Is the new design's conversion DIFFERENT from the old?', the test is:",
         q_ar="لـ 'هل تحويل التصميم الجديد مختلف عن القديم؟'، الاختبار:",
         opts_en=["One-tail right","One-tail left","Two-tail","Bayesian"],
         opts_ar=["ذيل أيمن","ذيل أيسر","ذيلَين","بيزى"], ans="C",
         exp_en="'Different' (no direction) → two-tailed.",
         exp_ar="'مختلف' (لا اتجاه) ← ذيلَين.")
L.render_quiz(); L.save()

# ============================================================
# L05-06 — Running a Two-Tailed Test (workshop)
# ============================================================
L = Lesson(
    lid="M05-L06", module=5,
    title_en="Running a Two-Tailed Hypothesis Test",
    title_ar="تشغيل اختبار فرضيات ذيلَين",
    objectives_en=[
        "Walk end-to-end through a two-tailed test.",
        "Use Excel ToolPak (Two-Sample t-test) to compute the result.",
        "Interpret the output and write a defensible conclusion.",
    ],
    objectives_ar=[
        "المرور من البداية للنهاية باختبار ذيلَين.",
        "استخدام Excel ToolPak لحساب النتيجة.",
        "تفسير المخرج وكتابة استنتاج قابل للدفاع.",
    ],
    sources=["05 Hypothesis Testing/Running a Two-Tailed Hypothesis Test.mp4",
             "Slides PDF: page 71-72",
             "Workbook: 04 - Hypothesis Testing.xlsx (Two-Tail sheet)"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "We'll use the vacuum-store warranty case from the course slides. The store ran two arms: 750 visitors offered a 5-year warranty and 750 offered a 7-year warranty. We want to know if mean revenue per visitor differs.")
    ap(d, "سنستخدم حالة متجر المكانس. شغّل المتجر مجموعتَين: 750 زائراً عُرض عليهم ضمان 5 سنوات و 750 ضمان 7 سنوات. نريد أن نعرف هل متوسط الإيراد لكل زائر يختلف.")
L.section("intro", "1 · The Setup", "1 · الإعداد", s1)

def s2(d):
    h3(d, "Step 1: Hypotheses", "خطوة 1: الفرضيات")
    eq(d, "H0:  μ_5y  =  μ_7y      Ha:  μ_5y  ≠  μ_7y")
    h3(d, "Step 2: Alpha", "خطوة 2: ألفا")
    ep(d, "α = 0.05 (95% confidence).")
    ap(d, "ألفا = 0.05 (ثقة 95%).")
    h3(d, "Step 3: Test choice", "خطوة 3: اختيار الاختبار")
    ep(d, "Two independent samples with means → two-sample t-test (two-tailed).")
    ap(d, "عينتان مستقلتان بمتوسطَين ← اختبار t لعينتَين (ذيلَين).")
    eq(d, "t  =  (x̄₁ − x̄₂)  /  √(s₁²/n₁ + s₂²/n₂)",
       note_en="Two-sample t-statistic (Welch's)", note_ar="إحصاء t لعينتَين (Welch)")
    L.data["formulas"].append({"id":"F-M05-L06-01","name_en":"Two-sample t-statistic","name_ar":"إحصاء t لعينتَين",
        "expr":"t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)",
        "where_en":"x̄ᵢ = sample means, sᵢ = sample std devs, nᵢ = sample sizes",
        "where_ar":"x̄ᵢ متوسطات، sᵢ انحرافات، nᵢ أحجام",
        "when_en":"Comparing means of two independent groups.",
        "when_ar":"مقارنة متوسطَى مجموعتَين مستقلتَين."})

    h3(d, "Step 4: Compute (Excel)", "خطوة 4: الحساب (Excel)")
    en_(d, "Data → Data Analysis → t-Test: Two-Sample Assuming Unequal Variances.")
    en_(d, "Variable 1 Range = 5-year warranty data. Variable 2 = 7-year. Hypothesized Mean Diff = 0. α = 0.05.")
    en_(d, "Excel returns: t Stat, df, p one-tail, p two-tail.")
    an_(d, "البيانات ← تحليل البيانات ← t-Test: Two-Sample Assuming Unequal Variances.")
    an_(d, "نطاق المتغير 1 = ضمان 5 سنوات. المتغير 2 = 7 سنوات. الفرق المفترض = 0. ألفا = 0.05.")
    an_(d, "Excel يعيد: t Stat، df، p ذيل واحد، p ذيلَين.")

    h3(d, "Step 5: Decide", "خطوة 5: القرار")
    ep(d, "If p (two-tail) < 0.05 → reject H0; warranties drive different revenue.")
    ap(d, "إذا p (ذيلَين) < 0.05 ← ارفض H0؛ الضمانات تحرّك إيرادات مختلفة.")
    ep(d, "From the slides example: the difference between the means was ~13%, but the test FAILED to reject H0 because the variance within each group was huge. With n=750 and the observed spread, we could not be confident the gap wasn't noise.")
    ap(d, "من مثال الشرائح: الفرق بين المتوسطَين ~13%، لكن الاختبار فشل فى رفض H0 لأن التباين داخل كل مجموعة ضخم. مع n=750 والانتشار المرصود، لا يمكن الثقة فى أن الفجوة ليست ضوضاء.")
L.section("example", "2 · The Full Five-Step Walk-Through",
          "2 · المرور الكامل عبر الخمس خطوات", s2)

def s3(d):
    callout(d, "key",
            "Lesson learned: a 13% gap in raw means is NOT automatically a real effect. Variance and sample size matter. Always run the test before concluding.",
            "الدرس: فجوة 13% فى المتوسطات الخام ليست تأثيراً حقيقياً تلقائياً. التباين وحجم العينة يهمّان. شغّل الاختبار قبل الاستنتاج.")
L.section("interpret", "3 · The Big Lesson",
          "3 · الدرس الكبير", s3)

def s4(d):
    callout(d, "use",
            "Every two-sample A/B test follows this exact pattern in Excel or Python (scipy.stats.ttest_ind). Variant A vs Variant B, two columns of data, one t-test, one p-value, one decision.",
            "كل اختبار A/B لعينتَين يتبع هذا النمط فى Excel أو Python (scipy.stats.ttest_ind). نسخة A مقابل B، عمودان، اختبار t واحد، قيمة p واحدة، قرار واحد.")
L.section("practice", "4 · Pattern for Any A/B Test",
          "4 · نمط لأى اختبار A/B", s4)

L.add_memory_aid("Two-sample two-tailed t-test = the everyday A/B test.",
                 "اختبار t لعينتَين ذيلَين = اختبار A/B اليومى.")
L.add_memory_aid("Big raw difference ≠ statistical significance. Run the test.",
                 "فرق خام كبير ≠ دلالة إحصائية. شغّل الاختبار.")
L.summary(
    en="A two-tailed test compares two groups for any difference. Five steps: state H0/Ha, set alpha, choose the two-sample t-test, compute in Excel ToolPak, read the two-tail p-value, decide. The warranty example shows that a raw 13% mean gap can fail to reject H0 when within-group variance is high — proof that hypothesis testing protects you from spurious conclusions.",
    ar="اختبار ذيلَين يقارن مجموعتَين بحثاً عن أى فرق. خمس خطوات: صِغ H0/Ha، حدد ألفا، اختر اختبار t لعينتَين، احسب فى Excel، اقرأ p ذيلَين، قرر. مثال الضمان يُظهر أن فجوة 13% خام قد تفشل فى رفض H0 حين يكون التباين الداخلى مرتفعاً — دليل على أن اختبار الفرضيات يحميك من الاستنتاجات الزائفة."
)
L.render_memory()

L.quiz_q(qid="M05-L06-Q1", qtype="MCQ",
         q_en="For comparing two A/B variants' means in Excel, use:",
         q_ar="لمقارنة متوسطَى نسختَى A/B فى Excel، استخدم:",
         opts_en=["F-test","Z-test for one sample","t-Test: Two-Sample Assuming Unequal Variances","Chi-square"],
         opts_ar=["F-test","Z-test لعينة واحدة","t-Test: Two-Sample Assuming Unequal Variances","Chi-square"],
         ans="C", exp_en="Two-sample t-test is the workhorse for comparing two means.",
         exp_ar="t لعينتَين هو الحصان لمقارنة متوسطَين.")

L.quiz_q(qid="M05-L06-Q2", qtype="TrueFalse",
         q_en="True or False: A 13% raw gap between means always means statistical significance.",
         q_ar="صح أم خطأ: فجوة 13% خام بين متوسطَين تعنى دلالة إحصائية دائماً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Depends on variance and n. Run the test.",
         exp_ar="تعتمد على التباين و n. شغّل الاختبار.")

L.quiz_q(qid="M05-L06-Q3", qtype="MCQ",
         q_en="The Excel output shows p (two-tail) = 0.08. At α = 0.05:",
         q_ar="مخرج Excel: p ذيلَين = 0.08. عند ألفا = 0.05:",
         opts_en=["Reject H0","Fail to reject H0","Inconclusive","Bug"],
         opts_ar=["ارفض H0","لا ترفض H0","غير حاسم","خطأ"], ans="B",
         exp_en="p > α → fail to reject.", exp_ar="p > ألفا ← لا ترفض.")

L.quiz_q(qid="M05-L06-Q4", qtype="Calc",
         q_en="If x̄₁=84, x̄₂=80, s₁=15, s₂=15, n₁=n₂=200. t ≈?",
         q_ar="إذا x̄₁=84، x̄₂=80، s₁=15، s₂=15، n₁=n₂=200. t ≈؟",
         opts_en=["1.0","2.0","2.67","4.0"], opts_ar=["1.0","2.0","2.67","4.0"], ans="C",
         exp_en="(84-80)/√(225/200+225/200) = 4/√2.25 = 4/1.5 ≈ 2.67.",
         exp_ar="(84-80)/√(225/200+225/200) = 4/1.5 ≈ 2.67.",
         formula_used="t = (x̄₁ − x̄₂)/√(s₁²/n₁ + s₂²/n₂)")

L.quiz_q(qid="M05-L06-Q5", qtype="MCQ",
         q_en="If you fail to reject H0 in a two-tailed test, you can say:",
         q_ar="إذا فشلت فى رفض H0 فى ذيلَين، يمكنك القول:",
         opts_en=["The variants are equal","H0 is true",
                  "No evidence of a difference in this test","Ha is wrong"],
         opts_ar=["النسختان متساويتان","H0 صحيحة","لا دليل على فرق فى هذا الاختبار","Ha خاطئة"],
         ans="C", exp_en="Defensible language.", exp_ar="لغة قابلة للدفاع.")
L.render_quiz(); L.save()

print("L05-04, L05-05, L05-06 done.")
