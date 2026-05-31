import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 03 - Descriptive Statistics"
os.makedirs(OUT, exist_ok=True)

# ───────── Cover ─────────
d = setup(); fc(d)
titlep(d, module_no=3,
       module_en="Descriptive Statistics",
       module_ar="الإحصاء الوصفى",
       subtitle_en="Mean · Median · Std Dev · Normal · Correlation · Regression",
       subtitle_ar="المتوسط · الوسيط · الانحراف · الطبيعى · الارتباط · الانحدار")
end_m(d, os.path.join(OUT, "M03 - Cover.docx"))

# ============================================================
# L03-01 — Chapter Introduction
# ============================================================
L = Lesson(
    lid="M03-L01", module=3,
    title_en="Chapter Introduction - Descriptive Statistics",
    title_ar="مقدمة الفصل - الإحصاء الوصفى",
    objectives_en=[
        "Understand what descriptive statistics does — and what it doesn't.",
        "See the three families: central tendency, dispersion, shape.",
        "Get a roadmap of the 12 lessons ahead.",
    ],
    objectives_ar=[
        "فهم ماذا يفعل الإحصاء الوصفى — وماذا لا يفعل.",
        "رؤية العائلات الثلاثة: النزعة المركزية، التشتت، الشكل.",
        "الحصول على خريطة طريق للدروس الـ 12 القادمة.",
    ],
    sources=["03 Descriptive Statistics/Chapter Introduction.mp4",
             "Slides PDF: pages 17-19",
             "Translation: ترجمة الشريحة.docx pages 17-19",
             "Excel: 02 - Descriptive Statistics.xlsx"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Descriptive statistics is all about organising and summarising data — answering: 'What does my data look like?'")
    ap(d, "الإحصاء الوصفى يدور حول تنظيم البيانات وتلخيصها — يجيب: 'كيف تبدو بياناتى؟'")
    callout(d, "key",
            "Descriptive stats does NOT make claims about a larger population. It only describes the data you already have. The moment you say 'so all customers spend...' you've crossed into inferential statistics.",
            "الإحصاء الوصفى لا يطلق ادعاءات على مجتمع أكبر. هو فقط يصف البيانات التى لديك. لحظة قولك 'إذن كل العملاء ينفقون...' تكون قد عبرت إلى الاستدلالى.")
L.section("intro", "1 · What Descriptive Statistics Does",
          "1 · ماذا يفعل الإحصاء الوصفى", s1)

def s2(d):
    ep(d, "We summarise data along three dimensions:")
    ap(d, "نلخّص البيانات على ثلاثة أبعاد:")
    kvtable(d, header=["Family / العائلة","Question / السؤال","Tools / الأدوات"],
        rows=[
            ("Central Tendency\nالنزعة المركزية",
             "Where is the centre?\nأين المركز؟",
             "Mean, Median, Mode\nالمتوسط، الوسيط، المنوال"),
            ("Dispersion / Spread\nالتشتت",
             "How spread out is it?\nما مدى انتشار البيانات؟",
             "Range, Variance, Std Dev, IQR\nالمدى، التباين، الانحراف، الربيعى"),
            ("Shape / Distribution\nالشكل",
             "What's the silhouette?\nما خطّ التوزيع؟",
             "Skewness, Kurtosis\nالالتواء، التفرطح"),
        ])
L.section("definitions", "2 · The Three Big Families", "2 · العائلات الثلاثة", s2)

def s3(d):
    ep(d, "This module walks through them in 12 carefully sequenced lessons:")
    ap(d, "هذا الموديول يعبر بهم عبر 12 درساً متتابعاً:")
    en_(d, "Central tendency: Mean / Median / Mode (L02).")
    en_(d, "Dispersion: Min, Max, Range, Quartiles (L03), then Variance & Std Dev (L04).")
    en_(d, "Shape: Normal distribution (L05), Kurtosis (L06), Skewness (L07).")
    en_(d, "Excel hands-on: Univariate analysis workshop (L08).")
    en_(d, "Bivariate analysis (L09): Correlation (L10), Linear Regression (L11), Excel workshop (L12).")
    an_(d, "النزعة المركزية: المتوسط/الوسيط/المنوال (الدرس 02).")
    an_(d, "التشتت: الحد الأدنى والأقصى والمدى والربيعيات (03)، ثم التباين والانحراف (04).")
    an_(d, "الشكل: التوزيع الطبيعى (05)، التفرطح (06)، الالتواء (07).")
    an_(d, "Excel عملياً: ورشة التحليل أحادى المتغير (08).")
    an_(d, "التحليل ثنائى المتغير (09): الارتباط (10)، الانحدار الخطى (11)، ورشة Excel (12).")
L.section("practice", "3 · Roadmap of the 12 Lessons", "3 · خريطة الدروس الـ 12", s3)

L.add_memory_aid("Three questions: Where is the centre? How spread? What shape?",
                 "ثلاثة أسئلة: أين المركز؟ ما التشتت؟ ما الشكل؟")
L.summary(
    en="Descriptive statistics organises and summarises the data you have. It answers three questions — centre, spread, shape — and gives you the vocabulary to talk about any dataset before you make decisions on it.",
    ar="الإحصاء الوصفى ينظم البيانات التى لديك ويلخّصها. يجيب على ثلاثة أسئلة — المركز، التشتت، الشكل — ويمنحك المفردات لتتحدث عن أى بيانات قبل أن تتخذ قرارات عليها."
)
L.render_memory()

L.quiz_q(qid="M03-L01-Q1", qtype="MCQ",
         q_en="Descriptive statistics primarily answers:",
         q_ar="الإحصاء الوصفى يجيب أساساً على:",
         opts_en=["What will happen tomorrow?","What does my data look like?","Is the difference real?","What is the prior probability?"],
         opts_ar=["ماذا سيحدث غداً؟","كيف تبدو بياناتى؟","هل الفرق حقيقى؟","ما الاحتمال المسبق؟"],
         ans="B", exp_en="Pure description of the data on hand.",
         exp_ar="مجرد وصف للبيانات التى بين يديك.")

L.quiz_q(qid="M03-L01-Q2", qtype="MCQ",
         q_en="The three families of descriptive statistics are:",
         q_ar="العائلات الثلاثة للإحصاء الوصفى هى:",
         opts_en=["Mean, mode, p-value","Centre, spread, shape","Classical, frequentist, Bayesian","Sample, population, statistic"],
         opts_ar=["متوسط، منوال، قيمة p","المركز، التشتت، الشكل","كلاسيكى، تكرارى، بيزى","عينة، مجتمع، إحصاء"],
         ans="B", exp_en="Centre + Spread + Shape.", exp_ar="مركز + تشتت + شكل.")

L.quiz_q(qid="M03-L01-Q3", qtype="MCQ",
         q_en="A measure of central tendency is:",
         q_ar="مقياس نزعة مركزية هو:",
         opts_en=["Standard deviation","Skewness","Median","Range"],
         opts_ar=["الانحراف المعيارى","الالتواء","الوسيط","المدى"],
         ans="C", exp_en="Median is a centre measure (50th percentile).",
         exp_ar="الوسيط مقياس مركز (الربيع الـ 50).")

L.quiz_q(qid="M03-L01-Q4", qtype="TrueFalse",
         q_en="True or False: Descriptive statistics tells us about uncertainty in our estimates of a population.",
         q_ar="صح أم خطأ: الإحصاء الوصفى يخبرنا عن عدم اليقين فى تقديرات المجتمع.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="That's the job of INFERENTIAL statistics. Descriptive only describes the sample.",
         exp_ar="هذه وظيفة الإحصاء الاستدلالى. الوصفى فقط يصف العينة.")

L.quiz_q(qid="M03-L01-Q5", qtype="MCQ",
         q_en="In a retail company, monthly KPIs (revenue, returns, items per order) shown on a dashboard are an example of:",
         q_ar="فى متجر، مؤشرات الأداء الشهرية (إيرادات، إرجاع، أصناف لكل طلب) على لوحة بيانات هى مثال على:",
         opts_en=["Inferential statistics","Descriptive statistics","Machine learning","Hypothesis testing"],
         opts_ar=["إحصاء استدلالى","إحصاء وصفى","تعلم آلى","اختبار فرضيات"],
         ans="B", exp_en="Summary metrics from data on hand = descriptive.",
         exp_ar="مقاييس ملخصة من البيانات الحالية = وصفى.")
L.render_quiz(); L.save()

# ============================================================
# L03-02 — Measures of Central Tendency
# ============================================================
L = Lesson(
    lid="M03-L02", module=3,
    title_en="Measures of Central Tendency",
    title_ar="مقاييس النزعة المركزية",
    objectives_en=[
        "Calculate Mean, Median, and Mode by hand and understand each.",
        "Choose the right measure for the data shape.",
        "See how outliers distort the Mean but spare the Median.",
    ],
    objectives_ar=[
        "حساب المتوسط والوسيط والمنوال يدوياً وفهم كل منها.",
        "اختيار المقياس المناسب حسب شكل البيانات.",
        "رؤية كيف تشوّه القيم المتطرفة المتوسط دون الوسيط.",
    ],
    sources=["03 Descriptive Statistics/Measures of Central Tendency.mp4",
             "Slides PDF: page 20",
             "Glossary: Mean, Median, Mode",
             "OpenStax: 2.5 Measures of the Center of the Data"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "A measure of central tendency answers: 'What is a typical value in this data?' Three classic measures, each with its own personality:")
    ap(d, "مقياس النزعة المركزية يجيب: 'ما القيمة النموذجية فى هذه البيانات؟' ثلاثة مقاييس كلاسيكية، لكل منها شخصية:")
L.section("intro", "1 · The Question of 'A Typical Value'",
          "1 · سؤال 'القيمة النموذجية'", s1)

def s2(d):
    kvtable(d, header=["Measure / المقياس","Definition / التعريف","Strength / النقطة القوية","Weakness / النقطة الضعيفة"],
        rows=[
            ("Mean (Average)\nالمتوسط",
             "Sum of all values ÷ count.\nمجموع القيم ÷ العدد.",
             "Uses every data point.\nيستخدم كل قيمة.",
             "Pulled by outliers.\nتؤثر فيه القيم المتطرفة."),
            ("Median\nالوسيط",
             "Middle value when sorted.\nالقيمة الوسطى بعد الترتيب.",
             "Robust against outliers.\nمقاوم للقيم المتطرفة.",
             "Ignores most of the data.\nيتجاهل معظم البيانات."),
            ("Mode\nالمنوال",
             "Most frequent value.\nالقيمة الأكثر تكراراً.",
             "Works on categorical data.\nيعمل على البيانات الفئوية.",
             "May not exist or be unique.\nقد لا يوجد أو يتعدد."),
        ])
L.section("definitions", "2 · The Three Measures Side-by-Side",
          "2 · المقاييس الثلاثة جنباً إلى جنب", s2)

def s3(d):
    eq(d, "μ  =  (Σ xᵢ) / N", note_en="Population mean", note_ar="متوسط المجتمع")
    eq(d, "x̄  =  (Σ xᵢ) / n", note_en="Sample mean", note_ar="متوسط العينة")
    L.data["formulas"].append({"id":"F-M03-L02-01","name_en":"Population mean","name_ar":"متوسط المجتمع",
        "expr":"μ = (Σ xᵢ) / N","where_en":"N = total population size","where_ar":"N = حجم المجتمع",
        "when_en":"Have data for entire population.","when_ar":"تملك بيانات المجتمع كاملاً."})
    L.data["formulas"].append({"id":"F-M03-L02-02","name_en":"Sample mean","name_ar":"متوسط العينة",
        "expr":"x̄ = (Σ xᵢ) / n","where_en":"n = sample size","where_ar":"n = حجم العينة",
        "when_en":"Working with a sample (most real cases).","when_ar":"عند العمل بعينة (معظم الحالات)."})
    ep(d, "Median rule: sort the data. If n is odd, take the middle value. If n is even, average the two middle values.")
    ap(d, "قاعدة الوسيط: رتّب البيانات. إذا كان n فردياً، خذ القيمة الوسطى. إذا كان n زوجياً، خذ متوسط القيمتَين الوسطيتَين.")
    ep(d, "Mode: the value that appears most often. Datasets can be unimodal (one mode), bimodal (two), or have no mode.")
    ap(d, "المنوال: القيمة الأكثر تكراراً. قد تكون البيانات أحادية المنوال، أو ثنائية المنوال، أو دون منوال.")
L.section("formulas", "3 · Formulas", "3 · الصيغ", s3)

def s4(d):
    callout(d, "example",
            "Customer-spend data (10 customers, EGP):\n50, 60, 70, 70, 80, 90, 100, 110, 120, 2,500\nCompute Mean, Median, Mode.",
            "بيانات إنفاق 10 عملاء (ج.م):\n50, 60, 70, 70, 80, 90, 100, 110, 120, 2,500\nاحسب المتوسط، الوسيط، المنوال.")
    h3(d, "Mean", "المتوسط")
    eq(d, "x̄ = (50+60+70+70+80+90+100+110+120+2500) / 10  =  3,250 / 10  =  325")
    ep(d, "Mean = 325 EGP. But notice — only one customer spent more than 200 EGP. The mean is misleading because of the 2,500 outlier.")
    ap(d, "المتوسط = 325 ج.م. لكن لاحظ — عميل واحد فقط أنفق أكثر من 200 ج.م. المتوسط مضلّل بسبب القيمة المتطرفة 2,500.")
    h3(d, "Median", "الوسيط")
    ep(d, "Sorted data; 10 values, so average positions 5 and 6: (80 + 90) / 2 = 85.")
    ap(d, "البيانات مرتبة؛ 10 قيم، فمتوسط الموقعَين 5 و 6: (80 + 90) / 2 = 85.")
    eq(d, "Median = (80 + 90) / 2 = 85")
    h3(d, "Mode", "المنوال")
    ep(d, "70 appears twice; all others once. Mode = 70.")
    ap(d, "القيمة 70 تظهر مرتين؛ الباقى مرة واحدة. المنوال = 70.")
    callout(d, "key",
            "Mean 325 vs Median 85 — the gap of 240 is the outlier's signature. In this skewed reality, the Median (85) much better represents the typical customer.",
            "متوسط 325 مقابل وسيط 85 — الفجوة 240 هى بصمة القيمة المتطرفة. فى هذا التشوّه، الوسيط (85) يمثل العميل النموذجى أفضل بكثير.")
L.section("example", "4 · Worked Example - Customer Spend",
          "4 · مثال محلول - إنفاق العملاء", s4)

def s5(d):
    kvtable(d, header=["Distribution shape / شكل التوزيع","Mean vs Median vs Mode / المقارنة"],
        rows=[
            ("Normal (symmetric)\nطبيعى متماثل",
             "Mean = Median = Mode\nالمتوسط = الوسيط = المنوال"),
            ("Right-skewed (positive)\nمنحرف لليمين (موجب)",
             "Mode < Median < Mean (mean is pulled right)\nالمنوال < الوسيط < المتوسط"),
            ("Left-skewed (negative)\nمنحرف لليسار (سالب)",
             "Mean < Median < Mode (mean is pulled left)\nالمتوسط < الوسيط < المنوال"),
        ])
L.section("interpret", "5 · Reading Shape from the Three Measures",
          "5 · قراءة الشكل من المقاييس الثلاثة", s5)

def s6(d):
    callout(d, "trap",
            "Reporting only the Mean when the data is skewed. Income, time-on-site, transaction value — most business metrics are right-skewed; always also show the Median.",
            "تقديم المتوسط فقط حين تكون البيانات منحرفة. الدخل، وقت البقاء، قيمة المعاملة — معظم مقاييس الأعمال منحرفة لليمين؛ اعرض الوسيط دائماً.")
    callout(d, "trap",
            "Forgetting that Mode can be used for categorical data. The 'most popular product colour' is a Mode, not a Mean.",
            "نسيان أن المنوال يصلح للبيانات الفئوية. 'أكثر لون منتج شعبية' منوال لا متوسط.")
L.section("traps", "6 · Common Traps", "6 · الفخاخ الشائعة", s6)

def s7(d):
    callout(d, "use",
            "E-commerce report: 'Average order value: EGP 850.' Always add 'Median order value: EGP 420' — this tells the business what the typical customer buys, not just what wealthy outliers buy.",
            "تقرير تجارة إلكترونية: 'متوسط قيمة الطلب: 850 ج.م.' أضف دائماً 'وسيط قيمة الطلب: 420 ج.م' — يخبر العمل بما يشتريه العميل النموذجى لا الأثرياء.")
    callout(d, "use",
            "Salary survey: News headlines say 'Average salary in the tech sector is $145k.' The Median is $95k. Mean is inflated by a few highly-paid executives. Both numbers tell different — and equally valid — stories.",
            "استطلاع رواتب: عناوين الأخبار تقول 'متوسط راتب التكنولوجيا 145 ألف دولار.' الوسيط 95 ألف. المتوسط مضخّم بسبب مدراء قليلى العدد عالى الأجر. الرقمَان يحكيان قصصاً مختلفة — وكلاهما صحيح.")
L.section("practice", "7 · Real Business Uses", "7 · استخدامات أعمال حقيقية", s7)

L.add_memory_aid("Mean = arithmetic average. Median = middle. Mode = most common.",
                 "المتوسط = حسابى. الوسيط = الوسط. المنوال = الأكثر شيوعاً.")
L.add_memory_aid("Outliers distort the Mean and pull it toward themselves. The Median doesn't budge.",
                 "القيم المتطرفة تشوّه المتوسط وتجذبه نحوها. الوسيط لا يتحرك.")
L.add_memory_aid("In skewed business data (revenue, time, salary): always show Mean AND Median.",
                 "فى بيانات الأعمال المنحرفة (إيرادات، وقت، راتب): اعرض المتوسط والوسيط معاً دائماً.")
L.summary(
    en="Mean, Median, and Mode each describe 'the centre' from a different angle. Mean uses every value but is sensitive to outliers. Median is robust. Mode tells you the most common value (or category). Smart analysts report at least two of them whenever the data is skewed.",
    ar="المتوسط والوسيط والمنوال يصفون 'المركز' من زاوية مختلفة. المتوسط يستخدم كل القيم لكنه حسّاس للقيم المتطرفة. الوسيط مقاوم. المنوال يخبرك بالأكثر شيوعاً (أو الفئة الأكثر تكراراً). المحلل الذكى يقدّم اثنَين منهم على الأقل حين تكون البيانات منحرفة."
)
L.render_memory()

L.quiz_q(qid="M03-L02-Q1", qtype="Calc",
         q_en="A small shop's daily sales (units) for a week: 12, 14, 13, 15, 14, 16, 14. What is the Mode?",
         q_ar="مبيعات محل صغير يومياً (وحدات) لأسبوع: 12, 14, 13, 15, 14, 16, 14. ما المنوال؟",
         opts_en=["12","13","14","15"], opts_ar=["12","13","14","15"], ans="C",
         exp_en="14 appears 3 times — most often.", exp_ar="القيمة 14 تظهر 3 مرات — الأكثر تكراراً.")

L.quiz_q(qid="M03-L02-Q2", qtype="Calc",
         q_en="Data: 4, 8, 6, 10, 12. What is the Mean?",
         q_ar="بيانات: 4, 8, 6, 10, 12. ما المتوسط؟",
         opts_en=["6","8","10","12"], opts_ar=["6","8","10","12"], ans="B",
         exp_en="(4+8+6+10+12)/5 = 40/5 = 8.", exp_ar="(4+8+6+10+12)/5 = 40/5 = 8.",
         formula_used="x̄ = Σxᵢ / n")

L.quiz_q(qid="M03-L02-Q3", qtype="Calc",
         q_en="Salaries (k EGP): 8, 9, 10, 11, 80. Which measure best represents a typical salary?",
         q_ar="رواتب (ألف ج.م): 8, 9, 10, 11, 80. أى مقياس يمثل الراتب النموذجى أفضل؟",
         opts_en=["Mean (23.6)","Median (10)","Mode (none)","Range (72)"],
         opts_ar=["المتوسط (23.6)","الوسيط (10)","المنوال (لا يوجد)","المدى (72)"],
         ans="B", exp_en="The 80k outlier pulls the mean far from where typical salaries lie. Median = 10 is far more honest.",
         exp_ar="القيمة المتطرفة 80 ألف تجذب المتوسط بعيداً عن الرواتب النموذجية. الوسيط = 10 أكثر صدقاً.")

L.quiz_q(qid="M03-L02-Q4", qtype="MCQ",
         q_en="In a right-skewed (positive) distribution:",
         q_ar="فى توزيع منحرف لليمين (موجب):",
         opts_en=["Mean > Median > Mode","Mean = Median = Mode","Mode > Median > Mean","Median > Mean > Mode"],
         opts_ar=["متوسط > وسيط > منوال","متوسط = وسيط = منوال","منوال > وسيط > متوسط","وسيط > متوسط > منوال"],
         ans="A", exp_en="The long right tail drags the mean rightward, past the median.",
         exp_ar="الذيل الأيمن الطويل يسحب المتوسط لليمين، بعيداً عن الوسيط.")

L.quiz_q(qid="M03-L02-Q5", qtype="MCQ",
         q_en="A favourite colour survey (Red, Blue, Blue, Green, Red, Blue) - the relevant measure is:",
         q_ar="استبيان اللون المفضّل (أحمر، أزرق، أزرق، أخضر، أحمر، أزرق) - المقياس المناسب:",
         opts_en=["Mean","Median","Mode","Variance"],
         opts_ar=["المتوسط","الوسيط","المنوال","التباين"],
         ans="C", exp_en="Categorical data → Mode. Blue appears 3 times.",
         exp_ar="بيانات فئوية ← المنوال. الأزرق يظهر 3 مرات.")
L.render_quiz(); L.save()

print("L03-01, L03-02 done.")
