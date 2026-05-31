import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 03 - Descriptive Statistics"

# ============================================================
# L03-09 — Univariate vs Bivariate
# ============================================================
L = Lesson(
    lid="M03-L09", module=3,
    title_en="Univariate vs Bivariate Analysis",
    title_ar="التحليل أحادى ومثنوى المتغير",
    objectives_en=[
        "Distinguish univariate from bivariate analysis.",
        "Choose the right visualisation for each.",
        "Recognise when a question requires looking at two variables together.",
    ],
    objectives_ar=[
        "التمييز بين التحليل أحادى المتغير ومثنوى المتغير.",
        "اختيار التصور المناسب لكل منهما.",
        "إدراك متى يتطلب السؤال النظر فى متغيرَين معاً.",
    ],
    sources=["03 Descriptive Statistics/Univariate vs Bivariate.mp4",
             "Slides PDF: page 31"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "So far, every measure we calculated (mean, median, std dev) looked at ONE column at a time — univariate analysis. The next leap is bivariate: comparing two columns to discover relationships.")
    ap(d, "حتى الآن، كل المقاييس (المتوسط، الوسيط، الانحراف) نظرت إلى عمود واحد فى المرة — تحليل أحادى المتغير. القفزة التالية: التحليل ثنائى المتغير — مقارنة عمودَين لاكتشاف العلاقات.")
L.section("intro", "1 · From One Variable to Two", "1 · من متغير واحد إلى اثنَين", s1)

def s2(d):
    kvtable(d, header=["Aspect / الجانب","Univariate / أحادى","Bivariate / ثنائى"],
        rows=[
            ("Columns examined\nالأعمدة","One.\nواحد.","Two together.\nاثنان معاً."),
            ("Goal\nالهدف","Describe one variable.\nوصف متغير.",
             "Measure relationship between two.\nقياس العلاقة بين اثنَين."),
            ("Tools\nالأدوات","Mean, median, std, histogram.\nمتوسط، وسيط، انحراف، رسم.",
             "Correlation, scatter plot, regression.\nارتباط، تشتت، انحدار."),
            ("Sample question\nسؤال نموذجى",
             "What's the average order value?\nما متوسط قيمة الطلب؟",
             "Does ad spend predict sales?\nهل ينبئ إنفاق الإعلانات بالمبيعات؟"),
        ])
L.section("definitions", "2 · Side-by-Side", "2 · جنباً إلى جنب", s2)

def s3(d):
    ep(d, "Many business questions only become interesting when you ask about two variables together:")
    ap(d, "كثير من أسئلة الأعمال تصبح مثيرة حين تسأل عن متغيرَين معاً:")
    eb(d, "Does the day of the week affect sales? (day × sales)")
    eb(d, "Does price discount drive volume? (discount % × units sold)")
    eb(d, "Does customer age predict purchase amount? (age × spend)")
    eb(d, "Does session duration correlate with conversion? (duration × converted)")
    ab(d, "هل يؤثر يوم الأسبوع على المبيعات؟ (يوم × مبيعات)")
    ab(d, "هل يؤدى خصم السعر إلى زيادة الكميات؟ (خصم% × وحدات)")
    ab(d, "هل ينبئ عمر العميل بمبلغ الشراء؟ (عمر × إنفاق)")
    ab(d, "هل ترتبط مدة الجلسة بالتحويل؟ (مدة × تحويل)")
L.section("practice", "3 · Why Bivariate Matters in Business",
          "3 · لماذا يهم الثنائى فى الأعمال", s3)

def s4(d):
    callout(d, "tip",
            "The scatter plot is the visual workhorse of bivariate analysis. One axis per variable, one dot per observation. Patterns jump out: linear, curved, scattered, clustered.",
            "مخطط التشتت هو حصان عمل التحليل الثنائى. محور لكل متغير، نقطة لكل مشاهدة. الأنماط تظهر: خطية، منحنية، عشوائية، متجمعة.")
L.section("concept", "4 · The Visual Workhorse", "4 · حصان العمل البصرى", s4)

L.add_memory_aid("Univariate = one column. Bivariate = two columns + relationship.",
                 "أحادى = عمود واحد. ثنائى = عمودان + علاقة.")
L.summary(
    en="Univariate analysis describes one variable; bivariate analysis explores how two variables move together. The pivot from one to the other is the gateway to predicting, correlating, and modelling — which is what the next lessons (correlation, regression) build on.",
    ar="الأحادى يصف متغيراً واحداً؛ الثنائى يستكشف كيف يتحرك متغيران معاً. هذه القفزة هى بوابة التنبؤ والارتباط والنمذجة — وهو ما يبنى عليه الدرس التالى (الارتباط، الانحدار)."
)
L.render_memory()

L.quiz_q(qid="M03-L09-Q1", qtype="MCQ",
         q_en="Calculating the standard deviation of one column is:",
         q_ar="حساب الانحراف المعيارى لعمود واحد:",
         opts_en=["Bivariate","Univariate","Correlation","Regression"],
         opts_ar=["ثنائى","أحادى","ارتباط","انحدار"], ans="B",
         exp_en="One variable at a time = univariate.", exp_ar="متغير واحد = أحادى.")

L.quiz_q(qid="M03-L09-Q2", qtype="MCQ",
         q_en="A scatter plot of age vs spend is:",
         q_ar="مخطط تشتت العمر مقابل الإنفاق:",
         opts_en=["Univariate","Bivariate","Histogram","Box plot"],
         opts_ar=["أحادى","ثنائى","رسم تكرار","صندوقى"], ans="B",
         exp_en="Two variables on two axes = bivariate.", exp_ar="متغيران على محورَين = ثنائى.")

L.quiz_q(qid="M03-L09-Q3", qtype="MCQ",
         q_en="Which question is bivariate?",
         q_ar="أى سؤال ثنائى؟",
         opts_en=["What is the average order value?","Does discount level affect order value?",
                  "What is the median age?","What is the mode response?"],
         opts_ar=["ما متوسط قيمة الطلب؟","هل يؤثر مستوى الخصم على قيمة الطلب؟",
                  "ما الوسيط العمرى؟","ما المنوال؟"], ans="B",
         exp_en="Two variables: discount and order value.", exp_ar="متغيران: خصم وقيمة طلب.")

L.quiz_q(qid="M03-L09-Q4", qtype="TrueFalse",
         q_en="True or False: A histogram is a bivariate visualisation.",
         q_ar="صح أم خطأ: الرسم التكرارى تصور ثنائى.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Histogram shows one variable's distribution = univariate.",
         exp_ar="الرسم التكرارى يظهر توزيع متغير واحد = أحادى.")

L.quiz_q(qid="M03-L09-Q5", qtype="MCQ",
         q_en="Marketing wants to know which channel drives most signups. They should start with:",
         q_ar="التسويق يريد معرفة أى قناة تأتى بأكبر تسجيلات. يجب البدء بـ:",
         opts_en=["Univariate count of signups",
                  "Bivariate: signups grouped by channel",
                  "Bivariate: signups vs revenue",
                  "Skip statistics"],
         opts_ar=["عدّ أحادى للتسجيلات","ثنائى: تسجيلات مجمّعة بالقناة","ثنائى: تسجيلات مقابل إيرادات","تخطى الإحصاء"],
         ans="B", exp_en="Channel × signups → bivariate breakdown.",
         exp_ar="قناة × تسجيلات ← تحليل ثنائى.")
L.render_quiz(); L.save()

# ============================================================
# L03-10 — Correlation
# ============================================================
L = Lesson(
    lid="M03-L10", module=3,
    title_en="Correlation and the Correlation Coefficient",
    title_ar="الارتباط ومعامل الارتباط",
    objectives_en=[
        "Define correlation and the correlation coefficient r.",
        "Read r values: strength and direction.",
        "Avoid the most famous statistical fallacy: correlation ≠ causation.",
    ],
    objectives_ar=[
        "تعريف الارتباط ومعامل الارتباط r.",
        "قراءة قيم r: القوة والاتجاه.",
        "تجنّب أشهر مغالطة إحصائية: الارتباط ليس سببية.",
    ],
    sources=["03 Descriptive Statistics/Correlation.mp4",
             "Slides PDF: pages 32-33",
             "Glossary: Correlation, Correlation coefficient"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Correlation measures how much two variables move together. The correlation coefficient r is a single number, between −1 and +1, that summarises the strength AND direction of a LINEAR relationship.")
    ap(d, "الارتباط يقيس مدى تحرك متغيرَين معاً. معامل الارتباط r رقم واحد بين -1 و +1 يلخّص قوة واتجاه العلاقة الخطية.")
L.section("intro", "1 · One Number That Says A Lot",
          "1 · رقم واحد يقول الكثير", s1)

def s2(d):
    kvtable(d, header=["r value / قيمة r","Strength / القوة","Direction / الاتجاه"],
        rows=[
            ("r ≈ +1","Strong / قوية","Positive — they rise together.\nموجبة - يصعدان معاً."),
            ("r ≈ +0.7","Moderate-to-strong / متوسطة-قوية","Positive.\nموجبة."),
            ("r ≈ +0.3","Weak / ضعيفة","Positive.\nموجبة."),
            ("r ≈ 0","None (linear)","No linear pattern.\nلا نمط خطى."),
            ("r ≈ −0.3","Weak / ضعيفة","Negative.\nسالبة."),
            ("r ≈ −1","Strong / قوية","Negative — one rises, the other falls.\nسالبة - واحد يصعد والآخر يهبط."),
        ])
    eq(d, "r  =  Σ(xᵢ − x̄)(yᵢ − ȳ)  /  [(n−1) · sₓ · s_y]",
       note_en="Pearson correlation - Excel: =CORREL(X,Y)", note_ar="ارتباط بيرسون - Excel: =CORREL(X,Y)")
    L.data["formulas"].append({"id":"F-M03-L10-01","name_en":"Pearson correlation r","name_ar":"معامل ارتباط بيرسون",
        "expr":"r = Σ(xᵢ−x̄)(yᵢ−ȳ) / [(n−1)·sₓ·s_y]",
        "where_en":"sₓ, s_y are sample std devs","where_ar":"sₓ, s_y انحرافا العينة",
        "when_en":"Measuring linear association between two numeric variables.","when_ar":"قياس علاقة خطية بين متغيرَين رقمَيين."})
L.section("definitions", "2 · The r Scale", "2 · مقياس r", s2)

def s3(d):
    ep(d, "Three properties to remember:")
    ap(d, "ثلاث خصائص للحفظ:")
    en_(d, "Unit-free: r doesn't change if you switch from dollars to thousands of dollars.")
    en_(d, "Symmetric: corr(X,Y) = corr(Y,X).")
    en_(d, "Linear-only: r misses curved relationships.")
    an_(d, "بلا وحدة: r لا يتغير بتغيير من دولار لألف دولار.")
    an_(d, "متماثل: corr(X,Y) = corr(Y,X).")
    an_(d, "خطى فقط: r يفوّت العلاقات المنحنية.")
L.section("concept", "3 · Three Properties of r", "3 · ثلاث خصائص لـ r", s3)

def s4(d):
    callout(d, "example",
            "From the course workbook (S&P 500 Bivariate sheet): correlation of daily price-CHANGE with daily VOLUME is r = −0.244.\nInterpretation: a weak negative correlation. On days when volume is unusually high, returns tend to be slightly negative — likely capturing panic-selling days.",
            "من ملف الدورة (ورقة Bivariate): ارتباط التغير اليومى مع الحجم اليومى r = −0.244.\nالتفسير: ارتباط سالب ضعيف. فى الأيام التى الحجم مرتفع، العوائد تميل لأن تكون سالبة قليلاً — على الأرجح تلتقط أيام بيع بدافع الذعر.")
L.section("example", "4 · Worked Example - S&P 500 Volume vs Returns",
          "4 · مثال محلول - حجم S&P 500 والعوائد", s4)

def s5(d):
    callout(d, "trap",
            "Correlation ≠ Causation. Ice-cream sales correlate with drowning deaths — both rise in summer. Ice cream doesn't cause drowning; HEAT does. Always ask: 'Is there a hidden third variable?'",
            "الارتباط ليس سببية. مبيعات الآيس كريم ترتبط بحالات الغرق — كلاهما يرتفع فى الصيف. لا الآيس كريم سبب الغرق، بل الحرارة هى السبب الخفى. اسأل دائماً: 'هل هناك متغير ثالث خفى؟'")
    callout(d, "trap",
            "r = 0 does NOT mean 'no relationship' — it means 'no LINEAR relationship'. A perfect parabola has r ≈ 0 even though Y is fully determined by X. Always check the scatter plot.",
            "r = 0 لا يعنى 'لا علاقة' — يعنى 'لا علاقة خطية'. القطع المكافئ المثالى r ≈ 0 رغم أن Y محدد كاملاً بـ X. تحقق دائماً من مخطط التشتت.")
    callout(d, "trap",
            "Correlation can be inflated by outliers. One billionaire customer can make age-spend correlation look much stronger than it really is.",
            "الارتباط قد يتضخم بالقيم المتطرفة. عميل ملياردير واحد قد يجعل ارتباط العمر-الإنفاق يبدو أقوى مما هو حقاً.")
L.section("traps", "5 · The Three Big Pitfalls",
          "5 · الفخاخ الثلاثة الكبرى", s5)

def s6(d):
    callout(d, "use",
            "Marketing: r between ad-spend and revenue. r=0.8 means ad spend is a strong leading indicator — defend that marketing budget!",
            "تسويق: r بين إنفاق الإعلانات والإيرادات. r=0.8 يعنى الإنفاق مؤشر قائد قوى — دافع عن ميزانية التسويق!")
    callout(d, "use",
            "HR analytics: r between training hours and productivity. Positive r supports continued investment in training.",
            "الموارد البشرية: r بين ساعات التدريب والإنتاجية. r موجب يدعم استمرار الاستثمار فى التدريب.")
    callout(d, "use",
            "Finance portfolio: assets with low correlation between them reduce overall portfolio risk (diversification).",
            "محفظة مالية: أصول بارتباطات منخفضة بينها تقلل المخاطرة الكلية (التنويع).")
L.section("practice", "6 · Business Uses", "6 · استخدامات أعمال", s6)

L.add_memory_aid("r ranges from −1 to +1. Sign = direction. Magnitude = strength.",
                 "r من -1 إلى +1. الإشارة = الاتجاه. الحجم = القوة.")
L.add_memory_aid("Correlation ≠ Causation. Always check for a hidden third variable.",
                 "الارتباط ليس سببية. ابحث عن متغير ثالث خفى دائماً.")
L.add_memory_aid("Always plot before reporting r. A scatter plot saves you from many fallacies.",
                 "ارسم قبل أن تعرض r. مخطط التشتت ينقذك من مغالطات كثيرة.")
L.summary(
    en="Correlation coefficient r is a unit-free, scale-invariant measure of the linear relationship between two variables, bounded by [−1, +1]. It is one of the most useful descriptive statistics — and one of the most abused. Always pair r with a scatter plot, watch for outliers, and never claim causation from correlation alone.",
    ar="معامل الارتباط r مقياس بلا وحدة، ثابت المقياس، للعلاقة الخطية بين متغيرَين، محصور بين -1 و +1. هو واحد من أكثر المقاييس الوصفية فائدة — وأكثرها إساءة استخدام. اربط r دائماً بمخطط تشتت، انتبه للمتطرفات، ولا تدّعى السببية من الارتباط وحده."
)
L.render_memory()

L.quiz_q(qid="M03-L10-Q1", qtype="MCQ",
         q_en="r = −0.85 means:",
         q_ar="r = −0.85 يعنى:",
         opts_en=["Strong positive linear relationship","Strong negative linear relationship",
                  "Weak negative","No linear relationship"],
         opts_ar=["علاقة خطية موجبة قوية","علاقة خطية سالبة قوية","سالبة ضعيفة","لا علاقة خطية"], ans="B",
         exp_en="|r| close to 1 = strong; negative sign = inverse.",
         exp_ar="|r| قرب 1 = قوى؛ الإشارة السالبة = عكسى.")

L.quiz_q(qid="M03-L10-Q2", qtype="MCQ",
         q_en="The famous saying about correlation is:",
         q_ar="القول الشهير عن الارتباط:",
         opts_en=["Correlation = Causation","Correlation implies causation","Correlation does NOT imply causation","Correlation is always wrong"],
         opts_ar=["الارتباط = السببية","الارتباط يعنى السببية","الارتباط لا يعنى السببية","الارتباط دائماً خاطئ"], ans="C",
         exp_en="The classic warning every analyst learns.", exp_ar="التحذير الكلاسيكى لكل محلل.")

L.quiz_q(qid="M03-L10-Q3", qtype="MCQ",
         q_en="A scatter plot shows a clear U-shape. r ≈ 0. What's true?",
         q_ar="مخطط تشتت يظهر شكل U واضح. r ≈ 0. ما الصحيح؟",
         opts_en=["No relationship exists","A relationship exists but is non-linear",
                  "Data is wrong","r should be 1"],
         opts_ar=["لا توجد علاقة","علاقة موجودة لكنها غير خطية","البيانات خاطئة","r يجب أن يكون 1"], ans="B",
         exp_en="r captures linear only. A parabola has near-zero r despite a perfect non-linear relationship.",
         exp_ar="r يلتقط الخطى فقط. القطع المكافئ r ~ 0 رغم وجود علاقة غير خطية مثالية.")

L.quiz_q(qid="M03-L10-Q4", qtype="Calc",
         q_en="Marketing finds r = 0.92 between TV ad spend and weekly sales. The best conclusion:",
         q_ar="التسويق وجد r = 0.92 بين الإنفاق على إعلانات التلفزيون والمبيعات الأسبوعية. الاستنتاج الأفضل:",
         opts_en=["TV ads cause sales perfectly",
                  "Strong positive linear association; further testing needed for causation",
                  "Coincidence; r doesn't matter","Stop advertising"],
         opts_ar=["إعلانات التلفزيون تسبب المبيعات بشكل مثالى",
                  "ارتباط خطى موجب قوى؛ يحتاج اختبار سببية إضافى","صدفة؛ r لا يهم","أوقف الإعلانات"],
         ans="B", exp_en="Strong correlation supports investigating, but doesn't prove cause.",
         exp_ar="ارتباط قوى يدعم البحث، لكن لا يثبت السببية.")

L.quiz_q(qid="M03-L10-Q5", qtype="TrueFalse",
         q_en="True or False: Correlation has the same units as the original variables.",
         q_ar="صح أم خطأ: الارتباط له نفس وحدات المتغيرات الأصلية.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="r is dimensionless (unit-free). That's a huge advantage.",
         exp_ar="r بلا وحدة. وهذه ميزة كبيرة.")
L.render_quiz(); L.save()

# ============================================================
# L03-11 — Simple Linear Regression
# ============================================================
L = Lesson(
    lid="M03-L11", module=3,
    title_en="Simple Linear Regression",
    title_ar="الانحدار الخطى البسيط",
    objectives_en=[
        "Write and interpret the regression equation Y = mX + b.",
        "Distinguish the dependent variable Y from independent variable X.",
        "Use regression to make basic business predictions.",
    ],
    objectives_ar=[
        "كتابة وتفسير معادلة الانحدار Y = mX + b.",
        "التمييز بين المتغير التابع Y والمستقل X.",
        "استخدام الانحدار للتنبؤ فى الأعمال.",
    ],
    sources=["03 Descriptive Statistics/Simple Linear Regression.mp4",
             "Slides PDF: page 34",
             "Glossary: Regression Analysis, Coefficient",
             "OpenStax: Chapter 12"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Correlation tells you that two variables MOVE together. Simple linear regression goes further: it gives you the EQUATION of the line that best fits the data, so you can predict Y from X.")
    ap(d, "الارتباط يخبرك بأن متغيرَين يتحركان معاً. الانحدار الخطى البسيط يتقدم: يعطيك معادلة الخط الذى يناسب البيانات أفضل، فيمكنك التنبؤ بـ Y من X.")
L.section("intro", "1 · From Correlation to Prediction",
          "1 · من الارتباط إلى التنبؤ", s1)

def s2(d):
    eq(d, "Y  =  m · X  +  b",
       note_en="The regression line. Y = predicted value.", note_ar="خط الانحدار. Y = القيمة المتنبأ بها.")
    h3(d, "Vocabulary", "المفردات")
    kvtable(d, header=["Symbol / الرمز","Meaning / المعنى","Business meaning / المعنى التجارى"],
        rows=[
            ("Y", "Dependent variable (what we predict).\nمتغير تابع (ما نتنبأ به).",
             "Sales, revenue, churn rate.\nمبيعات، إيرادات، معدل مغادرة."),
            ("X", "Independent variable (the predictor).\nمتغير مستقل (المنبئ).",
             "Ad spend, time on site, customer age.\nإنفاق إعلانات، وقت موقع، عمر العميل."),
            ("m", "Slope of the line.\nميل الخط.",
             "How much Y changes per +1 of X.\nكم يتغير Y لكل +1 من X."),
            ("b", "Y-intercept.\nنقطة تقاطع Y.",
             "Y when X = 0.\nقيمة Y حين X = 0."),
        ])
    eq(d, "m  =  r · (s_y / sₓ)        b  =  ȳ − m · x̄",
       note_en="Least-squares formulas for slope and intercept",
       note_ar="صيغ المربعات الصغرى للميل والتقاطع")
    for f in [
        ("F-M03-L11-01","Regression line","خط الانحدار","Y = m·X + b",
         "Predicted Y from X","Y المتنبأ به من X",
         "Predicting a numeric outcome from one numeric predictor.","التنبؤ بنتيجة رقمية من منبئ رقمى."),
        ("F-M03-L11-02","Slope","الميل","m = r · (s_y / sₓ)",
         "r is correlation","r هو الارتباط",
         "Computing slope from correlation and std devs.","حساب الميل من الارتباط والانحرافات."),
        ("F-M03-L11-03","Intercept","نقطة التقاطع","b = ȳ − m · x̄",
         "Passes through (x̄, ȳ)","يمرّ بـ (x̄, ȳ)",
         "Completes the line equation.","يكمل معادلة الخط."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
L.section("formulas", "2 · The Equation Y = mX + b",
          "2 · المعادلة Y = mX + b", s2)

def s3(d):
    callout(d, "example",
            "Umbrella sales vs daily rainfall. After fitting:\nY = 12·X + 30\nWhere Y = umbrellas sold, X = inches of rain.\n• 0 inches → 30 umbrellas (baseline).\n• 1 inch → 12·1 + 30 = 42.\n• 5 inches → 12·5 + 30 = 90.",
            "مبيعات المظلات مقابل الأمطار اليومية. بعد التركيب:\nY = 12·X + 30\nحيث Y = مظلات مباعة، X = بوصات مطر.\n• 0 بوصة ← 30 مظلة (الأساس).\n• 1 بوصة ← 12·1 + 30 = 42.\n• 5 بوصات ← 12·5 + 30 = 90.")
    ep(d, "Interpretation: each extra inch of rain raises predicted sales by 12 umbrellas. Even on dry days, baseline demand is 30.")
    ap(d, "التفسير: كل بوصة إضافية ترفع المبيعات المتوقعة 12 مظلة. حتى فى الأيام الجافة، الطلب الأساسى 30.")
L.section("example", "3 · Worked Example - Umbrellas vs Rain",
          "3 · مثال محلول - المظلات والمطر", s3)

def s4(d):
    callout(d, "use",
            "Marketing: Sales = 8 · (Ad_Spend in $1000s) + 1,200. Reading: every extra $1,000 in ads adds 8 sales; without any ads, baseline = 1,200.",
            "تسويق: المبيعات = 8 · (إنفاق_الإعلانات بآلاف الدولارات) + 1,200. كل ألف دولار إضافى يضيف 8 مبيعات؛ بدون إعلانات الأساس 1,200.")
    callout(d, "use",
            "Finance: Predict revenue from headcount. Useful for budgeting and capacity planning.",
            "المالية: توقع الإيرادات من عدد الموظفين. مفيد للموازنة وتخطيط القدرات.")
    callout(d, "use",
            "Real estate: Price = 1,800 · sq_meters + 50,000. Quick price comp for new listings.",
            "العقارات: السعر = 1,800 · م² + 50,000. تقدير سعر سريع لعقارات جديدة.")
L.section("practice", "4 · Business Uses", "4 · استخدامات أعمال", s4)

def s5(d):
    callout(d, "trap",
            "Extrapolation: don't predict outside the observed X range. The umbrella formula breaks down at 100 inches of rain (and at negative inches).",
            "الاستقراء خارج النطاق: لا تتنبأ خارج نطاق X المرصود. صيغة المظلات تنهار عند 100 بوصة (وعند البوصات السالبة).")
    callout(d, "trap",
            "Linear assumption: regression assumes a LINE fits. If the true relationship is curved, a straight line is misleading.",
            "افتراض الخطية: الانحدار يفترض أن الخط يلائم. إذا كانت العلاقة منحنية، الخط مضلّل.")
    callout(d, "trap",
            "Outliers can tilt the entire line. One billion-dollar deal can change the slope of an entire revenue-vs-headcount regression.",
            "المتطرفات تميل الخط كله. صفقة بمليار دولار قد تغير ميل انحدار الإيرادات والموظفين.")
L.section("traps", "5 · Common Traps", "5 · الفخاخ الشائعة", s5)

L.add_memory_aid("Y = mX + b. m = slope. b = intercept. Use it to predict.",
                 "Y = mX + b. m = الميل. b = التقاطع. استخدمها للتنبؤ.")
L.add_memory_aid("Don't extrapolate beyond the data you fitted.",
                 "لا تستقرئ خارج البيانات التى استخدمتها للتركيب.")
L.summary(
    en="Simple linear regression fits a straight line Y = mX + b that best summarises a linear relationship between two numeric variables. The slope m quantifies how Y responds to X; the intercept b gives the baseline. Regression is the workhorse of business prediction — sales, demand, churn — but it works only when a line really fits the data, and only inside the observed X range.",
    ar="الانحدار الخطى البسيط يلائم خطاً Y = mX + b يلخّص علاقة خطية بين متغيرَين رقمَيين. الميل m يقيس كم يستجيب Y لـ X؛ التقاطع b يعطى الأساس. الانحدار حصان عمل التنبؤ فى الأعمال — مبيعات، طلب، مغادرة — لكنه يعمل فقط حين يلائم الخط حقاً، وداخل نطاق X المرصود."
)
L.render_memory()

L.quiz_q(qid="M03-L11-Q1", qtype="MCQ",
         q_en="In Y = mX + b, which is the slope?",
         q_ar="فى Y = mX + b، أيهما الميل؟",
         opts_en=["Y","X","m","b"], opts_ar=["Y","X","m","b"], ans="C",
         exp_en="m is slope; b is intercept.", exp_ar="m الميل، b التقاطع.")

L.quiz_q(qid="M03-L11-Q2", qtype="Calc",
         q_en="Y = 5X + 10. Predict Y when X = 4.",
         q_ar="Y = 5X + 10. توقع Y حين X = 4.",
         opts_en=["20","30","50","14"], opts_ar=["20","30","50","14"], ans="B",
         exp_en="5·4 + 10 = 30.", exp_ar="5·4 + 10 = 30.",
         formula_used="Y = mX + b")

L.quiz_q(qid="M03-L11-Q3", qtype="MCQ",
         q_en="A regression of sales (Y) on ads (X) gives m = 8 (in units of sales per $1000 ad spend). Interpretation:",
         q_ar="انحدار مبيعات (Y) على إعلانات (X) يعطى m = 8 (وحدة مبيعات لكل ألف دولار). التفسير:",
         opts_en=["Every $1000 in ads adds 8 sales (on average)",
                  "Every sale generates $8000 in ads",
                  "Slope means nothing","Sales fall 8 per ad dollar"],
         opts_ar=["كل ألف دولار يضيف 8 مبيعات (وسطياً)",
                  "كل بيع يولّد 8000 دولار إعلانات","الميل لا يعنى شيئاً","المبيعات تنخفض 8 لكل دولار"],
         ans="A", exp_en="Slope = average change in Y per +1 of X.",
         exp_ar="الميل = متوسط التغير فى Y لكل +1 من X.")

L.quiz_q(qid="M03-L11-Q4", qtype="MCQ",
         q_en="Why is extrapolation outside the data range dangerous?",
         q_ar="لماذا يعتبر الاستقراء خارج نطاق البيانات خطراً؟",
         opts_en=["It always works","The linear relationship may not hold outside the observed range",
                  "It's against the law","Excel can't do it"],
         opts_ar=["يعمل دائماً","قد لا تستمر العلاقة الخطية خارج النطاق","ضد القانون","Excel لا يستطيع"],
         ans="B", exp_en="The line was fitted within a range; outside, the world may behave differently.",
         exp_ar="الخط رُكِّب داخل نطاق؛ خارجه قد يتغير العالم.")

L.quiz_q(qid="M03-L11-Q5", qtype="TrueFalse",
         q_en="True or False: r and the regression slope m always have the same sign.",
         q_ar="صح أم خطأ: r و الميل m لهما نفس الإشارة دائماً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="m = r · (s_y/sₓ). Since std devs are positive, sign(m) = sign(r).",
         exp_ar="m = r · (s_y/sₓ). الانحرافات موجبة، فإشارة m = إشارة r.")
L.render_quiz(); L.save()

# ============================================================
# L03-12 — Excel Workshop: Bivariate Analysis
# ============================================================
L = Lesson(
    lid="M03-L12", module=3,
    title_en="Excel Workshop - Bivariate Analysis",
    title_ar="ورشة Excel - التحليل ثنائى المتغير",
    objectives_en=[
        "Compute correlation in Excel.",
        "Build a scatter plot.",
        "Run simple linear regression and read the output.",
    ],
    objectives_ar=[
        "حساب الارتباط فى Excel.",
        "بناء مخطط تشتت.",
        "تشغيل انحدار خطى بسيط وقراءة المخرجات.",
    ],
    sources=["03 Descriptive Statistics/Excel - Bivariate Analysis.mp4",
             "Workbook: 02 - Descriptive Statistics.xlsx (Bivariate sheet)",
             "Solution: 02 - Complete - Descriptive Statistics.xlsx"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "This workshop converts the theory of correlation and regression (L10-L11) into a working Excel flow on the course workbook.")
    ap(d, "هذه الورشة تحوّل نظرية الارتباط والانحدار (L10-L11) إلى سير عمل تطبيقى فى Excel على ملف الدورة.")
L.section("intro", "1 · From Theory to Spreadsheet",
          "1 · من النظرية إلى الجدول", s1)

def s2(d):
    h3(d, "Step 1 - Compute correlation", "خطوة 1 - حساب الارتباط")
    ep(d, "In an empty cell, type: =CORREL(C2:C1260, D2:D1260) where C=Volume, D=Change.")
    ap(d, "فى خلية فارغة اكتب: =CORREL(C2:C1260, D2:D1260) حيث C=الحجم، D=التغير.")
    ep(d, "Result: r ≈ −0.244. Weak negative correlation between volume and daily return.")
    ap(d, "النتيجة: r ≈ −0.244. ارتباط سالب ضعيف بين الحجم والعائد اليومى.")

    h3(d, "Step 2 - Build a scatter plot", "خطوة 2 - بناء مخطط تشتت")
    en_(d, "Select both columns (Volume and Change).")
    en_(d, "Insert tab → Scatter (the icon with dots) → Scatter (first option).")
    en_(d, "Right-click on the dots → Add Trendline → Linear → tick 'Display Equation' and 'Display R²'.")
    an_(d, "حدّد العمودَين (الحجم والتغير).")
    an_(d, "إدراج ← مخطط تشتت ← Scatter (الخيار الأول).")
    an_(d, "اضغط بالزر الأيمن على النقاط ← Add Trendline ← خطى ← فعّل 'Display Equation' و 'Display R²'.")

    h3(d, "Step 3 - Full regression (ToolPak)", "خطوة 3 - انحدار كامل (ToolPak)")
    en_(d, "Data → Data Analysis → Regression.")
    en_(d, "Y range = Change. X range = Volume.")
    en_(d, "Tick 'Labels'. Output range → empty cell → OK.")
    en_(d, "Read: Multiple R (= |r|), R², Intercept (b), X-variable coefficient (m), p-values.")
    an_(d, "Data ← تحليل البيانات ← Regression.")
    an_(d, "نطاق Y = Change. نطاق X = Volume.")
    an_(d, "Labels ← نطاق الإخراج ← موافق.")
    an_(d, "اقرأ: Multiple R، R²، Intercept (b)، معامل X (m)، قيم p.")
L.section("concept", "2 · Step-by-Step Workflow", "2 · سير العمل خطوة بخطوة", s2)

def s3(d):
    callout(d, "key",
            "R² (R-squared) = the fraction of Y's variance explained by X. R² = 0.06 means volume explains only 6% of return variance — most of the daily move comes from other factors. Even a 'significant' regression with small R² is barely useful for prediction.",
            "R² = نسبة تباين Y المُفسَّر بـ X. R² = 0.06 يعنى الحجم يفسّر 6% فقط من تباين العائد — معظم الحركة اليومية من عوامل أخرى. حتى الانحدار 'الذى ذو دلالة' بـ R² صغير يكاد لا يفيد للتنبؤ.")
L.section("interpret", "3 · Understanding R²", "3 · فهم R²", s3)

def s4(d):
    callout(d, "use",
            "ANY business question of the form 'Does X drive Y?' can use this workflow: ad spend vs revenue, session time vs conversion, employee training hours vs sales. Always: CORREL → scatter + trendline → ToolPak regression → R².",
            "أى سؤال أعمال على شكل 'هل X يحرك Y؟' يستخدم هذا السير: إعلانات مقابل إيرادات، وقت جلسة مقابل تحويل، ساعات تدريب مقابل مبيعات. دائماً: CORREL ← مخطط + خط ← ToolPak ← R².")
L.section("practice", "4 · Bring This Anywhere", "4 · طبّقها فى أى مكان", s4)

def s5(d):
    h3(d, "Excel quick reference", "مرجع Excel سريع")
    eb(d, "Correlation: =CORREL(X_range, Y_range)")
    eb(d, "Slope: =SLOPE(Y_range, X_range)  ← note Y first!")
    eb(d, "Intercept: =INTERCEPT(Y_range, X_range)")
    eb(d, "Predict Y at X=v: =FORECAST.LINEAR(v, Y_range, X_range)")
    eb(d, "R² of regression line: =RSQ(Y_range, X_range)")
L.section("formulas", "5 · Excel Function Reference",
          "5 · مرجع دوال Excel", s5)

L.add_memory_aid("CORREL → scatter + trendline → Regression in ToolPak. Three clicks tell the whole story.",
                 "CORREL ← مخطط تشتت + خط ← Regression فى ToolPak. ثلاث نقرات تروى القصة.")
L.summary(
    en="Excel gives you a complete bivariate workflow in minutes: =CORREL for the coefficient, a scatter plot with trendline for the shape, and Data → Regression for the full output (R², coefficients, p-values). Practise on the course workbook with Volume vs Change (r ≈ −0.244), then apply this to your own business data.",
    ar="Excel يعطيك سير عمل ثنائى المتغير كاملاً فى دقائق: =CORREL للمعامل، مخطط تشتت مع خط للشكل، و Data → Regression للمخرجات الكاملة (R²، معاملات، قيم p). تدرّب على ملف الدورة بـ الحجم مقابل التغير (r ≈ −0.244)، ثم طبّق هذا على بياناتك."
)
L.render_memory()

L.quiz_q(qid="M03-L12-Q1", qtype="MCQ",
         q_en="To get correlation in Excel:",
         q_ar="للحصول على الارتباط فى Excel:",
         opts_en=["=CORREL(X,Y)","=AVERAGE(X,Y)","=SUM(X*Y)","=DEVSQ(X)"],
         opts_ar=["=CORREL(X,Y)","=AVERAGE(X,Y)","=SUM(X*Y)","=DEVSQ(X)"], ans="A",
         exp_en="CORREL returns Pearson r.", exp_ar="CORREL يعيد r لبيرسون.")

L.quiz_q(qid="M03-L12-Q2", qtype="MCQ",
         q_en="In =SLOPE(A,B), which range is Y?",
         q_ar="فى =SLOPE(A,B)، أيهما Y؟",
         opts_en=["A (first)","B (second)","Either","Neither"],
         opts_ar=["A (الأول)","B (الثانى)","أى","لا أحد"], ans="A",
         exp_en="In SLOPE and INTERCEPT, Y comes first, then X.",
         exp_ar="فى SLOPE و INTERCEPT، Y أولاً ثم X.")

L.quiz_q(qid="M03-L12-Q3", qtype="MCQ",
         q_en="R² = 0.85 means:",
         q_ar="R² = 0.85 يعنى:",
         opts_en=["X causes 85% of Y","X explains 85% of Y's variance",
                  "85% chance the relationship is real","R = 0.85"],
         opts_ar=["X يسبب 85% من Y","X يفسّر 85% من تباين Y",
                  "85% احتمال العلاقة حقيقية","R = 0.85"], ans="B",
         exp_en="R² is variance explained, not causation, not the same as r.",
         exp_ar="R² تباين مُفسَّر، ليس سببية، ليس مثل r.")

L.quiz_q(qid="M03-L12-Q4", qtype="Calc",
         q_en="If r = 0.6, R² = ?",
         q_ar="إذا كان r = 0.6، فما R²؟",
         opts_en=["0.6","0.36","0.4","0.8"], opts_ar=["0.6","0.36","0.4","0.8"], ans="B",
         exp_en="R² = r² = 0.6² = 0.36.", exp_ar="R² = r² = 0.6² = 0.36.")

L.quiz_q(qid="M03-L12-Q5", qtype="TrueFalse",
         q_en="True or False: A scatter plot is essential before trusting a correlation number.",
         q_ar="صح أم خطأ: مخطط التشتت ضرورى قبل الوثوق برقم الارتباط.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="Always visualise — outliers and curves are invisible in r alone.",
         exp_ar="ارسم دائماً — المتطرفات والمنحنيات غير مرئية فى r وحده.")
L.render_quiz(); L.save()

print("L03-09, L03-10, L03-11, L03-12 done.")

# Build Module 03 index
module_index = {
    "id":"M03","module":3,"type":"module",
    "title_en":"Descriptive Statistics",
    "title_ar":"الإحصاء الوصفى",
    "lessons":[
        {"id":"M03-L01","title_en":"Chapter Introduction - Descriptive Statistics","title_ar":"مقدمة الفصل"},
        {"id":"M03-L02","title_en":"Measures of Central Tendency","title_ar":"مقاييس النزعة المركزية"},
        {"id":"M03-L03","title_en":"Measures of Dispersion - Range and Quartiles","title_ar":"المدى والربيعيات"},
        {"id":"M03-L04","title_en":"Variance and Standard Deviation","title_ar":"التباين والانحراف المعيارى"},
        {"id":"M03-L05","title_en":"The Normal Distribution","title_ar":"التوزيع الطبيعى"},
        {"id":"M03-L06","title_en":"Kurtosis","title_ar":"التفرطح"},
        {"id":"M03-L07","title_en":"Asymmetric Distributions and Skewness","title_ar":"الالتواء"},
        {"id":"M03-L08","title_en":"Excel Workshop - Univariate Analysis","title_ar":"ورشة Excel - أحادى المتغير"},
        {"id":"M03-L09","title_en":"Univariate vs Bivariate Analysis","title_ar":"أحادى مقابل ثنائى"},
        {"id":"M03-L10","title_en":"Correlation and the Correlation Coefficient","title_ar":"الارتباط ومعامله"},
        {"id":"M03-L11","title_en":"Simple Linear Regression","title_ar":"الانحدار الخطى البسيط"},
        {"id":"M03-L12","title_en":"Excel Workshop - Bivariate Analysis","title_ar":"ورشة Excel - ثنائى المتغير"},
    ],
    "sources":["03 Descriptive Statistics/*","Slides PDF pages 17-35",
               "Workbook: 02 - Descriptive Statistics.xlsx"]
}
with open(os.path.join(OUT,"M03 - Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(module_index, f, ensure_ascii=False, indent=2)
print("Module 03 index written.")
