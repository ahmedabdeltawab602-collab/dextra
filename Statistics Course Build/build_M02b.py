import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 02 - The Statistical Landscape"

# ============================================================
# L02-03 — Descriptive vs Inferential Statistics
# ============================================================
L = Lesson(
    lid="M02-L03", module=2,
    title_en="Descriptive vs Inferential Statistics",
    title_ar="الإحصاء الوصفى مقابل الاستدلالى",
    objectives_en=[
        "Define descriptive and inferential statistics in plain words.",
        "Pick the right family for any business question.",
        "Avoid the classic mistake of confusing the two.",
    ],
    objectives_ar=[
        "تعريف الإحصاء الوصفى والاستدلالى بكلمات بسيطة.",
        "اختيار العائلة الصحيحة لأى سؤال أعمال.",
        "تجنّب الخطأ الكلاسيكى فى الخلط بينهما.",
    ],
    sources=["02 The Statistical Landscape/Descriptive vs Inferential Statistics.mp4",
             "Slides PDF: pages 9-11",
             "Glossary: Descriptive Statistics, Inferential Statistics"],
    out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "All of statistics splits into two families. Knowing which one you are doing — at any moment — keeps your reasoning clean.")
    ap(d, "كل الإحصاء ينقسم إلى عائلتين. معرفة أيهما تستخدم — فى أى لحظة — يحافظ على وضوح تفكيرك.")
L.section("intro", "1 · One Field, Two Families", "1 · علم واحد، عائلتان", s1)

def s2(d):
    kvtable(d,
        header=["Aspect / الجانب","Descriptive / وصفى","Inferential / استدلالى"],
        rows=[
            ("Goal\nالهدف",
             "Summarise what you have.\nتلخيص ما عندك.",
             "Generalise to what you don't have.\nالتعميم لما ليس عندك."),
            ("Scope\nالنطاق",
             "The sample itself.\nالعينة فقط.",
             "The population behind the sample.\nالمجتمع وراء العينة."),
            ("Tools\nالأدوات",
             "Mean, median, std, charts.\nالمتوسط، الوسيط، الانحراف، الرسوم.",
             "p-values, confidence intervals, tests.\nقيم p، فترات الثقة، الاختبارات."),
            ("Uncertainty\nعدم اليقين",
             "None - you compute exact numbers.\nصفر - أرقام مضبوطة.",
             "Always present - quantified by probability.\nحاضر دائماً - يُقاس بالاحتمال."),
            ("Business use\nاستخدام الأعمال",
             "KPIs, dashboards, reports.\nمؤشرات الأداء، اللوحات، التقارير.",
             "A/B testing, forecasting, surveys.\nاختبار A/B، التنبؤ، الاستطلاعات."),
        ])
L.section("definitions", "2 · Side-by-Side Comparison", "2 · مقارنة جنباً إلى جنب", s2)

def s3(d):
    ep(d, "Real business questions for a tree-planting program:")
    ap(d, "أسئلة أعمال حقيقية لبرنامج زراعة أشجار:")
    h3(d, "Descriptive questions", "أسئلة وصفية")
    eb(d, "How does this year's mortality compare to last year's?")
    eb(d, "What ratios of tree species did we plant?")
    eb(d, "What was our overall survival rate?")
    ab(d, "كيف يقارن معدل النفوق هذا العام بالعام الماضى؟")
    ab(d, "ما نسب أنواع الأشجار التى زرعناها؟")
    ab(d, "ما إجمالى معدل البقاء؟")
    h3(d, "Inferential questions", "أسئلة استدلالية")
    eb(d, "What types of trees should we plant next year for highest survival?")
    eb(d, "Why was this year's survival lower? Was it really the soil?")
    eb(d, "Is the relationship between slope and survival statistically real?")
    ab(d, "أى أنواع أشجار يجب أن نزرع العام القادم لأعلى بقاء؟")
    ab(d, "لماذا انخفض البقاء هذا العام؟ هل التربة فعلاً السبب؟")
    ab(d, "هل العلاقة بين الميل ومعدل البقاء حقيقية إحصائياً؟")
L.section("example", "3 · Same Project, Different Questions",
          "3 · نفس المشروع، أسئلة مختلفة", s3)

def s4(d):
    callout(d, "use",
            "In e-commerce: descriptive stats power your daily dashboard (today's revenue, average order value, conversion rate). Inferential stats power your weekly experiments (did the new checkout button increase conversion, after accounting for randomness?).",
            "فى التجارة الإلكترونية: الإحصاء الوصفى يدعم لوحاتك اليومية (إيرادات اليوم، متوسط قيمة الطلب، معدل التحويل). الاستدلالى يدعم تجاربك الأسبوعية (هل زر الخروج الجديد رفع التحويل فعلاً، بعد استبعاد العشوائية؟).")
    callout(d, "use",
            "In banking: descriptive stats answer 'How much did our credit-card customers spend in March?'. Inferential stats answer 'Does our new fraud-detection model lower fraud rates compared to the old one?'",
            "فى البنوك: الوصفى يجيب 'كم أنفق عملاء بطاقات الائتمان فى مارس؟'. الاستدلالى يجيب 'هل نموذج اكتشاف الاحتيال الجديد يقلل معدلات الاحتيال مقارنة بالقديم؟'")
L.section("practice", "4 · Where Each Lives in Business",
          "4 · أين تعيش كل عائلة فى الأعمال", s4)

def s5(d):
    callout(d, "trap",
            "Reporting a sample mean as if it were the population truth. 'Our customers spend $42 on average' (from 200 sampled customers) is not the same as 'every customer spends $42'. The sample mean is a point estimate with its own uncertainty.",
            "تقديم متوسط العينة وكأنه حقيقة المجتمع. 'عملاؤنا ينفقون 42 دولاراً وسطياً' (من عينة 200 عميل) ليس مثل 'كل العملاء ينفقون 42'. متوسط العينة تقدير نقطى له شك معه.")
    callout(d, "trap",
            "Running an A/B test then describing the difference without a p-value. A 3% lift in clicks might be real, or it might be noise — only an inferential test tells you which.",
            "تشغيل اختبار A/B ثم وصف الفرق دون قيمة p. ارتفاع 3% فى النقرات قد يكون حقيقياً أو ضوضاء — الاختبار الاستدلالى وحده يخبرك الفرق.")
L.section("traps", "5 · Common Traps", "5 · الفخاخ الشائعة", s5)

L.add_memory_aid("Descriptive describes. Inferential infers. Description has no uncertainty; inference always does.",
                 "الوصفى يصف. الاستدلالى يستدل. الوصف لا يقين فيه، الاستدلال دائماً فيه عدم يقين.")
L.summary(
    en="Descriptive statistics tell you what is — exact numbers from the data in your hands. Inferential statistics tell you what probably is — quantified guesses about a larger population. Most business questions need both: describe first, infer later.",
    ar="الإحصاء الوصفى يخبرك بما هو موجود — أرقام مضبوطة من البيانات بين يديك. الاستدلالى يخبرك بما هو الأرجح — تخمينات مدروسة عن مجتمع أكبر. معظم أسئلة الأعمال تحتاج الاثنين: صف أولاً، استدل لاحقاً."
)
L.render_memory()

L.quiz_q(qid="M02-L03-Q1", qtype="MCQ",
         q_en="Calculating the average rating of 12 product reviews you have is:",
         q_ar="حساب متوسط تقييم 12 مراجعة منتج بين يديك هو:",
         opts_en=["Inferential","Descriptive","Hypothesis testing","Forecasting"],
         opts_ar=["استدلالى","وصفى","اختبار فرضيات","تنبؤ"],
         ans="B", exp_en="Computing a number directly from your data is descriptive.",
         exp_ar="حساب رقم مباشر من بياناتك وصفى.")

L.quiz_q(qid="M02-L03-Q2", qtype="MCQ",
         q_en="Estimating the true satisfaction of ALL customers from a 500-person survey is:",
         q_ar="تقدير الرضا الحقيقى لكل العملاء من استطلاع 500 شخص هو:",
         opts_en=["Descriptive","Inferential","A KPI calculation","None of these"],
         opts_ar=["وصفى","استدلالى","حساب مؤشر أداء","لا شىء"],
         ans="B", exp_en="Going beyond your sample to a population is inference.",
         exp_ar="تجاوز العينة إلى المجتمع هو استدلال.")

L.quiz_q(qid="M02-L03-Q3", qtype="MCQ",
         q_en="A dashboard tile showing 'Yesterday's revenue: $48,200' is:",
         q_ar="بلاطة لوحة بيانات تعرض 'إيرادات الأمس: 48,200 دولار' هى:",
         opts_en=["Inferential","Descriptive","Predictive","Bayesian"],
         opts_ar=["استدلالى","وصفى","تنبؤى","بيزى"],
         ans="B", exp_en="Reporting a known total from data is descriptive.",
         exp_ar="عرض إجمالى معروف من البيانات وصفى.")

L.quiz_q(qid="M02-L03-Q4", qtype="MCQ",
         q_en="A/B testing whether a new homepage increases conversion is:",
         q_ar="اختبار A/B لمعرفة هل تزيد الصفحة الرئيسية الجديدة التحويل هو:",
         opts_en=["Descriptive","Inferential","Neither","Both descriptive and inferential"],
         opts_ar=["وصفى","استدلالى","لا واحد","الاثنان"],
         ans="D", exp_en="You describe each variant's conversion (descriptive) then test if the difference is real (inferential).",
         exp_ar="تصف معدل التحويل لكل نسخة (وصفى) ثم تختبر هل الفرق حقيقى (استدلالى).")

L.quiz_q(qid="M02-L03-Q5", qtype="TrueFalse",
         q_en="True or False: Descriptive statistics involve uncertainty about a larger population.",
         q_ar="صح أم خطأ: الإحصاء الوصفى يتضمن عدم يقين تجاه مجتمع أكبر.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Descriptive stats compute exact values from data you already have — no uncertainty.",
         exp_ar="الوصفى يحسب قيماً مضبوطة من بياناتك — لا عدم يقين.")
L.render_quiz(); L.save()

# ============================================================
# L02-04 — Approaches to Inferring Probability
# ============================================================
L = Lesson(
    lid="M02-L04", module=2,
    title_en="Approaches to Inferring Probability",
    title_ar="مناهج الاستدلال الاحتمالى",
    objectives_en=[
        "Define the three approaches to inference: Classical, Frequentist, Bayesian.",
        "Pick the right one for a given problem.",
        "Recognise how each interprets the word 'probability'.",
    ],
    objectives_ar=[
        "تعريف المناهج الثلاثة للاستدلال: الكلاسيكى، التكرارى، البيزى.",
        "اختيار المنهج المناسب لكل مشكلة.",
        "إدراك كيف يفسّر كل منهج كلمة 'احتمال'.",
    ],
    sources=["02 The Statistical Landscape/Approaches to Inferring Probability.mp4",
             "Slides PDF: pages 12-13",
             "Glossary: Bayesian"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "There are three big philosophies of probability. They mostly agree on the math, but they disagree on what probability MEANS.")
    ap(d, "هناك ثلاث فلسفات كبرى للاحتمال. تتفق غالباً على الرياضيات لكنها تختلف على معنى 'الاحتمال'.")
L.section("intro", "1 · Three Lenses on Probability", "1 · ثلاث عدسات للاحتمال", s1)

def s2(d):
    kvtable(d,
        header=["Approach / المنهج","Assumption / الافتراض","Formula idea / فكرة الصيغة"],
        rows=[
            ("Classical\nالكلاسيكى",
             "Outcomes have known equal odds.\nالنتائج لها احتمالات متساوية معروفة.",
             "P = Favorable / Possible outcomes\nP = المرغوب / كل الاحتمالات الممكنة"),
            ("Frequentist\nالتكرارى",
             "Probability = long-run frequency of observed events.\nالاحتمال = التكرار طويل المدى للأحداث المرصودة.",
             "P = Favorable observed / Total observations\nP = المرصود المرغوب / إجمالى المشاهدات"),
            ("Bayesian\nالبيزى",
             "Combine prior belief + new evidence.\nدمج المعرفة المسبقة + الأدلة الجديدة.",
             "P(A|B) = P(B|A) · P(A) / P(B)\nقاعدة بايز"),
        ])
    eq(d, "Bayes:  P(A | B)  =  P(B | A) · P(A) / P(B)",
       note_en="Bayes' theorem - update beliefs with evidence",
       note_ar="نظرية بايز - تحديث المعرفة بالأدلة")
    L.data["formulas"].append({
        "id":"F-M02-L04-01","name_en":"Classical probability","name_ar":"الاحتمال الكلاسيكى",
        "expr":"P = Favorable / Possible",
        "where_en":"All outcomes equally likely.","where_ar":"كل النتائج متساوية الاحتمال.",
        "when_en":"Symmetric setups: coins, dice, lottery.","when_ar":"حالات متناظرة: عملة، نرد، يانصيب."
    })
    L.data["formulas"].append({
        "id":"F-M02-L04-02","name_en":"Frequentist probability","name_ar":"الاحتمال التكرارى",
        "expr":"P = Favorable observed / Total observations",
        "where_en":"Estimated from observed data.","where_ar":"يُقدّر من البيانات المرصودة.",
        "when_en":"Repeated experiments, large samples.","when_ar":"تجارب متكررة، عينات كبيرة."
    })
    L.data["formulas"].append({
        "id":"F-M02-L04-03","name_en":"Bayes' theorem","name_ar":"نظرية بايز",
        "expr":"P(A | B) = P(B | A) · P(A) / P(B)",
        "where_en":"Combines prior P(A) with likelihood P(B|A).","where_ar":"يجمع المسبق P(A) مع الاحتمال P(B|A).",
        "when_en":"Have prior knowledge + new evidence.","when_ar":"عندما تتوفر معرفة مسبقة + دليل جديد."
    })
L.section("formulas", "2 · The Three Approaches Side-by-Side",
          "2 · المناهج الثلاثة جنباً إلى جنب", s2)

def s3(d):
    callout(d, "example",
            "Q: Who wins the World Rock-Paper-Scissors final?\n• Classical: 'No real skill — 50/50.'\n• Frequentist: 'In 1,510 historical matches lefties beat righties 62% of the time. So 62%.'\n• Bayesian: 'Prior says 62% for lefty. But this lefty also won 9 of his last 10 — update to ~78%.'",
            "س: من يفوز ببطولة العالم لحجر-ورق-مقص؟\n• الكلاسيكى: 'لا توجد مهارة — 50/50.'\n• التكرارى: 'فى 1,510 مباراة سابقة، الأعسر يفوز 62% من الوقت. إذن 62%.'\n• البيزى: 'المعرفة المسبقة 62% للأعسر. لكن هذا الأعسر فاز فى 9 من آخر 10 — التحديث إلى ~78%.'")
L.section("example", "3 · Worked Example - Rock, Paper, Scissors",
          "3 · مثال محلول - حجر ورق مقص", s3)

def s4(d):
    h3(d, "When to use each", "متى تستخدم كلاً")
    eb(d, "Classical: lotteries, casino games, fair-coin teaching examples.")
    eb(d, "Frequentist: A/B tests, clinical trials, anything with many repeated observations.")
    eb(d, "Bayesian: spam filters, recommendation systems, fraud detection, anything with prior knowledge.")
    ab(d, "الكلاسيكى: اليانصيب، ألعاب الكازينو، أمثلة تعليمية للعملة العادلة.")
    ab(d, "التكرارى: اختبارات A/B، التجارب السريرية، أى شىء فيه ملاحظات متكررة كثيرة.")
    ab(d, "البيزى: فلاتر السبام، أنظمة التوصية، اكتشاف الاحتيال، أى شىء فيه معرفة مسبقة.")
L.section("practice", "4 · Picking the Right Approach in Practice",
          "4 · اختيار المنهج فى الواقع", s4)

def s5(d):
    callout(d, "trap",
            "Using the classical approach when conditions aren't symmetric. Saying 'either it happens or it doesn't, so 50/50' for a startup's success ignores the actual base rates — most startups fail.",
            "استخدام المنهج الكلاسيكى حين لا تتوفر التناظرات. قول 'إما أن يحدث أو لا، إذن 50/50' لنجاح شركة ناشئة يتجاهل المعدلات الحقيقية — معظم الشركات الناشئة تفشل.")
    callout(d, "trap",
            "In Bayesian: choosing a wildly wrong prior. Garbage prior + good data ≠ good answer.",
            "فى البيزى: اختيار معرفة مسبقة خاطئة جداً. معرفة مسبقة سيئة + بيانات جيدة ≠ إجابة جيدة.")
L.section("traps", "5 · Common Traps", "5 · الفخاخ الشائعة", s5)

L.add_memory_aid("Classical = symmetry. Frequentist = repetition. Bayesian = update beliefs with evidence.",
                 "الكلاسيكى = التناظر. التكرارى = التكرار. البيزى = حدّث معرفتك بالأدلة.")
L.summary(
    en="The three approaches differ in what 'probability' means. Classical assumes symmetric outcomes; Frequentist relies on long-run observations; Bayesian updates prior belief with new data. This course mostly uses the Frequentist lens (Modules 04 and 05), the dominant approach in classical hypothesis testing and business A/B testing.",
    ar="المناهج الثلاثة تختلف فى معنى الاحتمال. الكلاسيكى يفترض نتائج متناظرة، التكرارى يعتمد على الملاحظات طويلة المدى، البيزى يحدث المعرفة المسبقة بالبيانات الجديدة. الدورة تعتمد غالباً على العدسة التكرارية (الموديولات 04 و 05) لأنها السائدة فى اختبار الفرضيات الكلاسيكى واختبارات A/B فى الأعمال."
)
L.render_memory()

L.quiz_q(qid="M02-L04-Q1", qtype="MCQ",
         q_en="Saying 'a fair coin lands heads 50% of the time' is which approach?",
         q_ar="القول 'عملة عادلة تظهر صورتها 50% من الوقت' من أى منهج؟",
         opts_en=["Frequentist","Bayesian","Classical","None"], opts_ar=["تكرارى","بيزى","كلاسيكى","لا شىء"],
         ans="C", exp_en="Equal possible outcomes — classic classical setup.",
         exp_ar="نتائج متساوية الاحتمال — وضع كلاسيكى بحت.")

L.quiz_q(qid="M02-L04-Q2", qtype="MCQ",
         q_en="A casino tracks 100,000 spins and finds red showed 47.9% of the time. Concluding P(red) ≈ 0.479 is:",
         q_ar="كازينو يسجل 100,000 دورة فتظهر الحمراء 47.9% من الوقت. استنتاج P(حمراء) ≈ 0.479 هو:",
         opts_en=["Classical","Frequentist","Bayesian","Wrong"], opts_ar=["كلاسيكى","تكرارى","بيزى","خاطئ"],
         ans="B", exp_en="Estimating probability from observed long-run frequency = frequentist.",
         exp_ar="تقدير الاحتمال من التكرار المرصود طويل المدى = تكرارى.")

L.quiz_q(qid="M02-L04-Q3", qtype="MCQ",
         q_en="A spam filter starts believing P(spam)=0.4, sees 'win money now' in the email, and updates to P(spam|words)=0.95. This is:",
         q_ar="فلتر سبام يبدأ بالاعتقاد P(سبام)=0.4، يرى 'اربح المال الآن' فى الرسالة، فيحدّث إلى 0.95. هذا:",
         opts_en=["Classical","Frequentist","Bayesian","Heuristic"], opts_ar=["كلاسيكى","تكرارى","بيزى","استدلالى"],
         ans="C", exp_en="Updating prior with new evidence is the heart of Bayes.",
         exp_ar="تحديث المعرفة المسبقة بالأدلة الجديدة هو جوهر بايز.")

L.quiz_q(qid="M02-L04-Q4", qtype="Calc",
         q_en="A bag has 3 red and 7 blue balls. Probability of drawing red?",
         q_ar="كيس به 3 كرات حمراء و 7 زرقاء. احتمال سحب الحمراء؟",
         opts_en=["0.30","0.50","0.70","0.43"], opts_ar=["0.30","0.50","0.70","0.43"],
         ans="A", exp_en="Classical: favorable=3, possible=10 → 3/10 = 0.30.",
         exp_ar="كلاسيكى: المرغوب=3، الكل=10 → 3/10 = 0.30.",
         formula_used="P = Favorable / Possible")

L.quiz_q(qid="M02-L04-Q5", qtype="MCQ",
         q_en="Which approach is dominant in business A/B testing?",
         q_ar="أى منهج هو السائد فى اختبارات A/B فى الأعمال؟",
         opts_en=["Classical","Frequentist","Bayesian","Machine learning"],
         opts_ar=["كلاسيكى","تكرارى","بيزى","تعلم آلى"],
         ans="B", exp_en="Frequentist p-value testing dominates online A/B testing platforms.",
         exp_ar="اختبار قيم p التكرارى يسيطر على منصات اختبار A/B.")
L.render_quiz(); L.save()
print("L02-03, L02-04 done.")
