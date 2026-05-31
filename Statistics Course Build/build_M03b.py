import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 03 - Descriptive Statistics"

# ============================================================
# L03-03 — Measures of Dispersion: Range & Quartiles
# ============================================================
L = Lesson(
    lid="M03-L03", module=3,
    title_en="Measures of Dispersion - Range and Quartiles",
    title_ar="مقاييس التشتت - المدى والربيعيات",
    objectives_en=[
        "Calculate Min, Max, Range, and Quartiles.",
        "Compute and interpret the Interquartile Range (IQR).",
        "Spot outliers with the 1.5 × IQR rule.",
    ],
    objectives_ar=[
        "حساب الحد الأدنى والأقصى والمدى والربيعيات.",
        "حساب وتفسير المدى الربيعى (IQR).",
        "اكتشاف القيم المتطرفة بقاعدة 1.5 × IQR.",
    ],
    sources=["03 Descriptive Statistics/Measures of Dispersion.mp4",
             "Slides PDF: pages 21-23",
             "Glossary: Range, Standard Deviation"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Knowing the centre alone isn't enough. Two datasets can share the same mean and tell completely different stories — one tightly clustered, one wildly spread out. Dispersion measures the spread.")
    ap(d, "معرفة المركز وحدها ليست كافية. مجموعتا بيانات قد تشتركان فى نفس المتوسط وتحكيان قصتَين مختلفتَين تماماً — واحدة متجمعة بإحكام، والأخرى منتشرة بشدة. مقاييس التشتت تقيس الانتشار.")
    callout(d, "key",
            "The mean tells you where customers are on average; dispersion tells you how reliably they cluster around that average. For a stable business, you want low dispersion. For high-risk investments, you accept more.",
            "المتوسط يخبرك أين العملاء فى المعدل، التشتت يخبرك بمدى تجمعهم حول هذا المعدل. لعمل مستقر، تريد تشتتاً منخفضاً. للاستثمارات عالية المخاطرة، تقبل بالأعلى.")
L.section("intro", "1 · Why Spread Matters", "1 · لماذا الانتشار مهم", s1)

def s2(d):
    kvtable(d, header=["Measure / المقياس","Definition / التعريف","Sensitive to outliers? / حسّاس للقيم المتطرفة؟"],
        rows=[
            ("Minimum / Maximum\nالحد الأدنى والأقصى",
             "Smallest / largest value.\nأصغر / أكبر قيمة.",
             "Extremely sensitive.\nحسّاس جداً."),
            ("Range\nالمدى",
             "Max − Min.\nالأقصى − الأدنى.",
             "Very sensitive.\nحسّاس جداً."),
            ("Quartiles (Q1, Q2, Q3)\nالربيعيات",
             "Cut data into 4 equal groups.\nيقسم البيانات لـ 4 مجموعات.",
             "Robust.\nمقاوم."),
            ("IQR (Interquartile Range)\nالمدى الربيعى",
             "Q3 − Q1; the middle 50%.\nالنطاق للنصف الأوسط 50%.",
             "Robust.\nمقاوم."),
        ])
L.section("definitions", "2 · The Spread Family", "2 · عائلة التشتت", s2)

def s3(d):
    eq(d, "Range = Max − Min", note_en="Simple but outlier-sensitive", note_ar="بسيط لكنه يتأثر بالمتطرفات")
    eq(d, "Q1 = 25th percentile     Q2 = Median (50th)     Q3 = 75th percentile",
       note_en="Quartiles split sorted data into 4 quarters", note_ar="تقسّم البيانات المرتبة لأربعة أرباع")
    eq(d, "IQR = Q3 − Q1", note_en="Range of the middle 50% — robust spread", note_ar="نطاق النصف الأوسط 50% — تشتت مقاوم")
    eq(d, "Outlier rule: x < Q1 − 1.5·IQR    or    x > Q3 + 1.5·IQR",
       note_en="Tukey's 1.5 × IQR fence", note_ar="حد توكى 1.5 × IQR")
    for f in [
        ("F-M03-L03-01","Range","المدى","Range = Max − Min","Spread of full data","انتشار البيانات كاملاً",
         "Quick first look; small data.","نظرة أولى سريعة؛ بيانات صغيرة."),
        ("F-M03-L03-02","IQR","المدى الربيعى","IQR = Q3 − Q1","Robust spread (middle 50%)","تشتت مقاوم للنصف الأوسط",
         "Skewed data or outliers present.","بيانات منحرفة أو فيها متطرفات."),
        ("F-M03-L03-03","Outlier fence","حد القيم المتطرفة","Lower: Q1 − 1.5·IQR  ·  Upper: Q3 + 1.5·IQR",
         "1.5 × IQR rule","قاعدة 1.5 × IQR","Identifying outliers in skewed data.","لاكتشاف المتطرفات فى بيانات منحرفة."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
L.section("formulas", "3 · Formulas", "3 · الصيغ", s3)

def s4(d):
    callout(d, "example",
            "Customer ages (n=14):  0, 1, 1, 1, 4, 5, 6, 6, 7, 10, 10, 27, 42, 70.\nCompute Range, Q1, Q2, Q3, IQR. Are 27, 42, 70 outliers?",
            "أعمار العملاء (n=14):  0, 1, 1, 1, 4, 5, 6, 6, 7, 10, 10, 27, 42, 70.\nاحسب المدى، Q1، Q2، Q3، IQR. هل 27، 42، 70 متطرفات؟")
    h3(d, "Step 1: Sort & basics", "خطوة 1: ترتيب وأساسيات")
    ep(d, "Already sorted. Min = 0, Max = 70.")
    ap(d, "مرتبة بالفعل. الحد الأدنى = 0، الأقصى = 70.")
    eq(d, "Range = 70 − 0 = 70")
    h3(d, "Step 2: Quartiles", "خطوة 2: الربيعيات")
    ep(d, "n=14. Median (Q2) = average of positions 7 and 8 = (6 + 6) / 2 = 6.")
    ap(d, "n=14. الوسيط (Q2) = متوسط الموقعَين 7 و 8 = (6 + 6) / 2 = 6.")
    ep(d, "Q1 = median of lower half (positions 1-7) = position 4 = 1. Q3 = median of upper half (positions 8-14) = position 11 = 10.")
    ap(d, "Q1 = وسيط النصف الأدنى (1-7) = الموقع 4 = 1. Q3 = وسيط النصف الأعلى (8-14) = الموقع 11 = 10.")
    eq(d, "Q1 = 1     Q2 = 6     Q3 = 10")
    h3(d, "Step 3: IQR", "خطوة 3: IQR")
    eq(d, "IQR = 10 − 1 = 9")
    h3(d, "Step 4: Outliers", "خطوة 4: المتطرفات")
    eq(d, "Lower fence: 1 − 1.5·9 = 1 − 13.5 = −12.5")
    eq(d, "Upper fence: 10 + 1.5·9 = 10 + 13.5 = 23.5")
    ep(d, "Any value above 23.5 is an outlier. So 27, 42, and 70 ALL qualify as outliers.")
    ap(d, "أى قيمة فوق 23.5 متطرفة. إذن 27 و 42 و 70 كلها متطرفات.")
    callout(d, "key",
            "Range = 70 made the data look wildly spread. IQR = 9 reveals the typical customer age varies by only 9 years. The IQR is the 'honest' spread measure when outliers exist.",
            "المدى = 70 جعل البيانات تبدو منتشرة جداً. IQR = 9 يكشف أن عمر العميل النموذجى يتغيّر بـ 9 سنوات فقط. IQR هو مقياس التشتت 'الصادق' عند وجود متطرفات.")
L.section("example", "4 · Worked Example - Customer Ages",
          "4 · مثال محلول - أعمار العملاء", s4)

def s5(d):
    callout(d, "use",
            "Real-estate analytics: Range of home prices in a city is huge (luxury skews it). But IQR of home prices tells you the 'typical' market — far more useful for pricing your average listing.",
            "تحليلات العقارات: مدى أسعار المنازل ضخم بسبب الفاخرة. لكن IQR يخبرك بـ 'السوق النموذجى' — أكثر فائدة لتسعير عقار عادى.")
    callout(d, "use",
            "Customer service tickets: Mean handle time = 8 minutes, but IQR = 4 (Q1=3, Q3=7). You know most tickets close in 3-7 minutes, and anything above 7 + 1.5*4 = 13 minutes deserves a closer look.",
            "تذاكر خدمة العملاء: متوسط زمن الحل = 8 دقائق، لكن IQR = 4 (Q1=3، Q3=7). تعلم أن معظم التذاكر تُحلّ فى 3-7 دقائق، وأى تذكرة فوق 13 دقيقة تستحق نظرة أعمق.")
L.section("practice", "5 · Business Uses", "5 · استخدامات أعمال", s5)

def s6(d):
    callout(d, "trap",
            "Using Range alone for skewed data. A range of 70 in customer ages says nothing useful when 12 of 14 customers are under 10.",
            "استخدام المدى وحده مع بيانات منحرفة. مدى 70 سنة فى أعمار العملاء لا يقول شيئاً حين 12 من 14 عميلاً تحت 10 سنوات.")
    callout(d, "trap",
            "Removing every '1.5×IQR outlier' without thinking. Sometimes outliers are real (your VIP customers) — investigate before deleting.",
            "حذف كل قيمة متطرفة بدون تفكير. أحياناً تكون المتطرفات حقيقية (عملاء VIP) — حقّق قبل الحذف.")
L.section("traps", "6 · Common Traps", "6 · الفخاخ الشائعة", s6)

L.add_memory_aid("Range = Max − Min (sensitive). IQR = Q3 − Q1 (robust).",
                 "المدى = الأقصى − الأدنى (حسّاس). IQR = Q3 − Q1 (مقاوم).")
L.add_memory_aid("Outlier fence: 1.5 × IQR beyond Q1 or Q3.",
                 "حد المتطرفات: 1.5 × IQR خارج Q1 أو Q3.")
L.summary(
    en="Range and IQR both measure spread, but they're not interchangeable. Range is simple and sensitive — one outlier blows it up. IQR is robust — it ignores extremes and shows the spread of the middle 50%. In real-world business data (often skewed), IQR is your trustworthy companion, and the 1.5×IQR rule flags suspect values.",
    ar="المدى و IQR يقيسان التشتت لكنهما ليسا متبادلَين. المدى بسيط وحسّاس — قيمة متطرفة واحدة تفجّره. IQR مقاوم — يتجاهل الأطراف ويظهر تشتت النصف الأوسط 50%. فى بيانات الأعمال (غالباً منحرفة)، IQR هو رفيقك الموثوق، وقاعدة 1.5×IQR ترصد القيم المشبوهة."
)
L.render_memory()

L.quiz_q(qid="M03-L03-Q1", qtype="Calc",
         q_en="Data: 5, 7, 8, 11, 14, 17, 22. What is the Range?",
         q_ar="بيانات: 5, 7, 8, 11, 14, 17, 22. ما المدى؟",
         opts_en=["5","17","22","-17"], opts_ar=["5","17","22","-17"], ans="B",
         exp_en="Max 22 − Min 5 = 17.", exp_ar="الأقصى 22 − الأدنى 5 = 17.",
         formula_used="Range = Max − Min")

L.quiz_q(qid="M03-L03-Q2", qtype="MCQ",
         q_en="The IQR represents the spread of:",
         q_ar="IQR يمثل تشتت:",
         opts_en=["The whole dataset","The middle 50% of data","The bottom 25%","The top 25%"],
         opts_ar=["كل البيانات","النصف الأوسط 50% من البيانات","الأدنى 25%","الأعلى 25%"],
         ans="B", exp_en="IQR = Q3 − Q1 captures the middle half.", exp_ar="IQR يلتقط النصف الأوسط.")

L.quiz_q(qid="M03-L03-Q3", qtype="Calc",
         q_en="If Q1 = 20 and Q3 = 50, what is the IQR and which values are outliers?",
         q_ar="إذا كان Q1 = 20 و Q3 = 50، فما IQR وأى القيم متطرفة؟",
         opts_en=["IQR=30; outliers below -25 or above 95",
                  "IQR=70; outliers below 0 or above 100",
                  "IQR=30; outliers above 50 only",
                  "IQR=10; outliers any value"],
         opts_ar=["IQR=30؛ متطرفات أقل من -25 أو أكبر من 95",
                  "IQR=70؛ متطرفات أقل من 0 أو أكبر من 100",
                  "IQR=30؛ متطرفات فوق 50 فقط",
                  "IQR=10؛ متطرفات أى قيمة"],
         ans="A", exp_en="IQR=30. Lower fence=20−45=−25. Upper=50+45=95.",
         exp_ar="IQR=30. حد سفلى=20−45=−25. علوى=50+45=95.")

L.quiz_q(qid="M03-L03-Q4", qtype="TrueFalse",
         q_en="True or False: Range is more resistant to outliers than IQR.",
         q_ar="صح أم خطأ: المدى أكثر مقاومة للمتطرفات من IQR.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Range USES the extremes; IQR ignores them.", exp_ar="المدى يستخدم الأطراف، IQR يتجاهلها.")

L.quiz_q(qid="M03-L03-Q5", qtype="MCQ",
         q_en="Two stores have the same average daily revenue. Store A's range is $5,000; Store B's range is $25,000. What does this say?",
         q_ar="متجران لهما نفس متوسط الإيرادات اليومية. مدى A = 5,000 دولار؛ مدى B = 25,000. ماذا يعنى ذلك؟",
         opts_en=["A has more variability","B has more variability — less predictable",
                  "They're identical","Range doesn't tell us anything"],
         opts_ar=["A أكثر تباين","B أكثر تباين — أقل قدرة على التنبؤ",
                  "متطابقان","المدى لا يخبرنا بشىء"],
         ans="B", exp_en="Wider range = wilder swings = harder to plan staffing, inventory, cash flow.",
         exp_ar="مدى أوسع = تقلبات أعنف = أصعب فى التخطيط للموارد والمخزون والسيولة.")
L.render_quiz(); L.save()

# ============================================================
# L03-04 — Variance & Standard Deviation
# ============================================================
L = Lesson(
    lid="M03-L04", module=3,
    title_en="Variance and Standard Deviation",
    title_ar="التباين والانحراف المعيارى",
    objectives_en=[
        "Compute Variance (σ², s²) and Standard Deviation (σ, s) by hand.",
        "Understand WHY we square deviations.",
        "Interpret std dev in the units of the original data.",
    ],
    objectives_ar=[
        "حساب التباين (σ², s²) والانحراف المعيارى (σ, s) يدوياً.",
        "فهم لماذا نربّع الانحرافات.",
        "تفسير الانحراف المعيارى بوحدات البيانات الأصلية.",
    ],
    sources=["03 Descriptive Statistics/Measures of Dispersion.mp4",
             "Slides PDF: pages 24-25",
             "Glossary: Standard Deviation",
             "OpenStax: 2.7 Measures of the Spread of the Data"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Variance and Standard Deviation are the two most important measures of dispersion. They capture, on average, how far each data point sits from the centre.")
    ap(d, "التباين والانحراف المعيارى هما أهم مقياسَى تشتت. يلتقطان، فى المعدل، كم تبعد كل نقطة عن المركز.")
    callout(d, "key",
            "Variance is the AVERAGE of SQUARED deviations from the mean. Standard deviation is the square root of variance — restoring the original units.",
            "التباين هو متوسط الانحرافات المربّعة عن المتوسط. الانحراف المعيارى هو الجذر التربيعى للتباين — يُعيد الوحدات الأصلية.")
L.section("intro", "1 · The Key Idea", "1 · الفكرة الجوهرية", s1)

def s2(d):
    eq(d, "σ²  =  Σ (xᵢ − μ)² / N",
       note_en="Population variance", note_ar="تباين المجتمع")
    eq(d, "s²  =  Σ (xᵢ − x̄)² / (n − 1)",
       note_en="Sample variance — n−1 is Bessel's correction", note_ar="تباين العينة — n−1 هو تصحيح بِسِل")
    eq(d, "σ  =  √σ²       s  =  √s²",
       note_en="Standard deviation = square root of variance", note_ar="الانحراف المعيارى = جذر التباين")
    for f in [
        ("F-M03-L04-01","Population variance","تباين المجتمع","σ² = Σ(xᵢ − μ)² / N",
         "Mean of squared deviations","متوسط الانحرافات المربّعة",
         "Whole population available.","المجتمع كله متوفر."),
        ("F-M03-L04-02","Sample variance","تباين العينة","s² = Σ(xᵢ − x̄)² / (n − 1)",
         "Bessel's correction n−1","تصحيح بِسِل n−1",
         "Working with a sample (most cases).","عند العمل بعينة (معظم الحالات)."),
        ("F-M03-L04-03","Standard deviation","الانحراف المعيارى","σ = √σ²    or    s = √s²",
         "Returns to original units","يُعيد الوحدات الأصلية",
         "Always - it's the headline measure.","دائماً - هو المقياس الرئيسى."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
L.section("formulas", "2 · The Formulas", "2 · الصيغ", s2)

def s3(d):
    ep(d, "Why square the deviations? If we just averaged signed deviations, positives and negatives cancel — giving zero. Squaring forces every deviation to be positive, then we take the square root to restore the original scale.")
    ap(d, "لماذا نربّع الانحرافات؟ لو فقط حسبنا متوسط الانحرافات بإشاراتها، فسالبها يلغى موجبها — والنتيجة صفر. التربيع يجبر كل انحراف على أن يكون موجباً، ثم نأخذ الجذر لاستعادة المقياس الأصلى.")
    callout(d, "meaning",
            "Why n−1 in sample variance? When using the sample mean, we lose one 'degree of freedom'. Dividing by n−1 corrects the underestimation. This is called Bessel's correction.",
            "لماذا n−1 فى تباين العينة؟ لأن استخدام متوسط العينة يُفقدنا 'درجة حرية' واحدة. القسمة على n−1 تصحّح التقدير المنخفض. هذا يُسمى تصحيح بِسِل.")
L.section("concept", "3 · Why Squaring, and Why n−1",
          "3 · لماذا التربيع، ولماذا n−1", s3)

def s4(d):
    callout(d, "example",
            "Sample of 3 daily sales (units): 3, 6, 15. Compute Mean, Variance (sample), Standard Deviation.",
            "عينة من 3 أيام مبيعات (وحدات): 3، 6، 15. احسب المتوسط، تباين العينة، الانحراف المعيارى.")
    h3(d, "Step 1: Mean", "خطوة 1: المتوسط")
    eq(d, "x̄ = (3 + 6 + 15) / 3 = 24 / 3 = 8")
    h3(d, "Step 2: Deviations from mean", "خطوة 2: الانحرافات عن المتوسط")
    eq(d, "(3 − 8) = −5    (6 − 8) = −2    (15 − 8) = +7")
    h3(d, "Step 3: Squared deviations", "خطوة 3: تربيع الانحرافات")
    eq(d, "(−5)² = 25    (−2)² = 4    (7)² = 49")
    eq(d, "Σ = 25 + 4 + 49 = 78")
    h3(d, "Step 4: Sample variance", "خطوة 4: تباين العينة")
    eq(d, "s²  =  78 / (3 − 1)  =  78 / 2  =  39")
    h3(d, "Step 5: Standard deviation", "خطوة 5: الانحراف المعيارى")
    eq(d, "s  =  √39  ≈  6.24 units")
    callout(d, "key",
            "Interpretation: on average, daily sales deviate ±6.24 units from the mean of 8. That's a wide band — your data is volatile.",
            "التفسير: فى المعدل، تنحرف المبيعات اليومية ±6.24 وحدة عن المتوسط 8. هذا نطاق واسع — البيانات متقلبة.")
L.section("example", "4 · Worked Example - Daily Sales",
          "4 · مثال محلول - مبيعات يومية", s4)

def s5(d):
    callout(d, "meaning",
            "Standard deviation is in the SAME UNITS as the data. If data is in EGP, std dev is in EGP. Variance is in EGP² — not as intuitive.",
            "الانحراف المعيارى بنفس وحدات البيانات. إذا كانت ج.م، فهو ج.م. التباين بـ ج.م² — أقل بداهة.")
    callout(d, "use",
            "Two products with the same monthly sales mean (1,000 units):\n• Product A: std dev = 50 → very predictable (810–1,190 most months)\n• Product B: std dev = 400 → wildly volatile (200–1,800 most months)\nProduct A is easier to stock and forecast.",
            "منتجان لهما نفس متوسط المبيعات الشهرى (1,000 وحدة):\n• المنتج A: الانحراف = 50 ← متوقَّع (810–1,190 غالباً)\n• المنتج B: الانحراف = 400 ← متقلب (200–1,800 غالباً)\nA أسهل فى التخزين والتنبؤ.")
L.section("interpret", "5 · Reading Std Dev in Business",
          "5 · قراءة الانحراف فى الأعمال", s5)

def s6(d):
    callout(d, "use",
            "Finance: standard deviation of stock returns = volatility = risk. The S&P 500 has historically had an annual std dev of ~15%. A stock with 50% std dev is 3.3× riskier.",
            "المالية: الانحراف المعيارى لعوائد السهم = التقلب = المخاطرة. مؤشر S&P 500 تاريخياً ~15% انحراف سنوى. سهم بانحراف 50% أعلى مخاطرة 3.3 مرات.")
    callout(d, "use",
            "Manufacturing quality control: lower std dev of part dimensions = better process. Six Sigma manufacturing aims for std dev so small that defects occur fewer than 3.4 times per million parts.",
            "ضبط جودة التصنيع: انحراف أقل لأبعاد القطع = عملية أفضل. تصنيع Six Sigma يستهدف انحرافاً صغيراً بحيث تحدث العيوب أقل من 3.4 لكل مليون قطعة.")
L.section("practice", "6 · Business Uses", "6 · استخدامات أعمال", s6)

def s7(d):
    callout(d, "trap",
            "Confusing variance and standard deviation. Variance is squared; std dev is in real units. Always report std dev in business reports.",
            "الخلط بين التباين والانحراف. التباين مربّع، الانحراف بوحدات حقيقية. اذكر الانحراف فى تقارير الأعمال دائماً.")
    callout(d, "trap",
            "Dividing by n instead of n−1 for a sample. Excel's STDEV.S uses n−1 (correct); STDEV.P uses n. Pick the right one!",
            "القسمة على n بدلاً من n−1 لعينة. Excel يستخدم STDEV.S بـ n−1 (الصحيح)؛ STDEV.P بـ n. اختر الصحيح!")
L.section("traps", "7 · Common Traps", "7 · الفخاخ الشائعة", s7)

L.add_memory_aid("Variance = average squared deviation. Std Dev = √variance (in original units).",
                 "التباين = متوسط الانحرافات المربّعة. الانحراف = √التباين (بالوحدات الأصلية).")
L.add_memory_aid("Sample → divide by n−1. Population → divide by N.",
                 "عينة ← اقسم على n−1. مجتمع ← اقسم على N.")
L.add_memory_aid("In finance and ops, lower std dev = lower risk = easier to plan.",
                 "فى المالية والعمليات، انحراف أقل = مخاطرة أقل = تخطيط أسهل.")
L.summary(
    en="Variance and Standard Deviation tell you 'on average, how far do data points sit from the mean'. Squaring eliminates negative-positive cancellation; the square root then returns to the original units. For samples, divide by n−1 (Bessel's correction). Std dev is the single most useful number for comparing volatility, risk, and predictability across two datasets.",
    ar="التباين والانحراف يخبرانك 'فى المعدل، كم تبتعد النقاط عن المتوسط'. التربيع يلغى تعارض الإشارات، والجذر يُعيد الوحدات الأصلية. للعينات، اقسم على n−1 (تصحيح بِسِل). الانحراف المعيارى هو أكثر رقم مفيد لمقارنة التقلب والمخاطرة والقدرة على التنبؤ بين بيانات."
)
L.render_memory()

L.quiz_q(qid="M03-L04-Q1", qtype="Calc",
         q_en="For sample {2, 4, 6}, mean = 4. What is sample variance s²?",
         q_ar="لعينة {2, 4, 6}، المتوسط = 4. ما تباين العينة s²؟",
         opts_en=["4","2.67","8","6"], opts_ar=["4","2.67","8","6"], ans="A",
         exp_en="Deviations: −2, 0, 2 → squared 4, 0, 4 → sum 8 → /(n−1)=/2 → 4.",
         exp_ar="الانحرافات: −2, 0, 2 ← مربّعة 4, 0, 4 ← مجموع 8 ← /(n−1)=/2 ← 4.",
         formula_used="s² = Σ(xᵢ−x̄)² / (n−1)")

L.quiz_q(qid="M03-L04-Q2", qtype="Calc",
         q_en="If sample variance = 25, what is sample standard deviation?",
         q_ar="إذا كان تباين العينة = 25، فما الانحراف المعيارى؟",
         opts_en=["12.5","5","25","625"], opts_ar=["12.5","5","25","625"], ans="B",
         exp_en="s = √s² = √25 = 5.", exp_ar="s = √s² = √25 = 5.")

L.quiz_q(qid="M03-L04-Q3", qtype="MCQ",
         q_en="Why do we square the deviations when computing variance?",
         q_ar="لماذا نربّع الانحرافات عند حساب التباين؟",
         opts_en=["To make calculations harder",
                  "Positive and negative deviations would cancel out otherwise",
                  "To convert units",
                  "Because Excel requires it"],
         opts_ar=["لجعل الحسابات أصعب",
                  "وإلا تلغى الموجبة والسالبة بعضها",
                  "لتحويل الوحدات",
                  "لأن Excel يتطلبها"],
         ans="B", exp_en="Squaring eliminates sign cancellation; the deviations would otherwise average to zero.",
         exp_ar="التربيع يلغى التعارض؛ وإلا فمتوسط الانحرافات المُوقّعة يساوى صفر.")

L.quiz_q(qid="M03-L04-Q4", qtype="MCQ",
         q_en="A stock with std dev 30% is — relative to one with std dev 10%:",
         q_ar="سهم بانحراف 30% — مقارنة بـ 10%:",
         opts_en=["Less risky","Equally risky","About 3× more volatile","Cheaper"],
         opts_ar=["أقل مخاطرة","نفس المخاطرة","أكثر تقلباً بـ 3 مرات","أرخص"],
         ans="C", exp_en="Higher std dev = higher volatility = higher risk.",
         exp_ar="انحراف أعلى = تقلب أعلى = مخاطرة أعلى.")

L.quiz_q(qid="M03-L04-Q5", qtype="TrueFalse",
         q_en="True or False: Variance and standard deviation are in the same units.",
         q_ar="صح أم خطأ: التباين والانحراف بنفس الوحدات.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Variance is squared units (e.g., EGP²); std dev is in original units (EGP).",
         exp_ar="التباين بوحدات مربّعة (ج.م²)؛ الانحراف بالوحدات الأصلية (ج.م).")
L.render_quiz(); L.save()

# ============================================================
# L03-05 — The Normal Distribution
# ============================================================
L = Lesson(
    lid="M03-L05", module=3,
    title_en="The Normal Distribution",
    title_ar="التوزيع الطبيعى",
    objectives_en=[
        "Recognise the four properties of a Normal distribution.",
        "Apply the Empirical Rule (68-95-99.7).",
        "Use the standard normal Z-score to compare observations.",
    ],
    objectives_ar=[
        "إدراك الخصائص الأربع للتوزيع الطبيعى.",
        "تطبيق القاعدة التجريبية (68-95-99.7).",
        "استخدام درجة Z للتوزيع الطبيعى القياسى لمقارنة المشاهدات.",
    ],
    sources=["03 Descriptive Statistics/Normal Distribution.mp4",
             "Slides PDF: pages 26-27",
             "Glossary: Normal Distribution",
             "OpenStax: Chapter 6"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The Normal distribution — the famous bell curve — is the most important distribution in statistics. It appears in everything from heights to test scores to measurement errors. Many of the techniques later in this course assume your data (or your means) are roughly Normal.")
    ap(d, "التوزيع الطبيعى — المنحنى الجرسى الشهير — هو أهم توزيع فى الإحصاء. يظهر فى كل شىء من الأطوال إلى درجات الامتحانات إلى أخطاء القياس. كثير من تقنيات الدورة لاحقاً تفترض أن بياناتك (أو متوسطاتك) طبيعية تقريباً.")
L.section("intro", "1 · Why the Normal Distribution Matters",
          "1 · لماذا يهمّ التوزيع الطبيعى", s1)

def s2(d):
    ep(d, "Four defining properties:")
    ap(d, "أربع خصائص مميِّزة:")
    en_(d, "Symmetric around the mean.")
    en_(d, "Mean = Median = Mode (all at the centre).")
    en_(d, "The tails approach but never touch the x-axis (asymptotic).")
    en_(d, "It follows the Empirical Rule: 68-95-99.7.")
    an_(d, "متماثل حول المتوسط.")
    an_(d, "المتوسط = الوسيط = المنوال (كلها فى المركز).")
    an_(d, "الذيلان يقتربان من المحور السينى ولا يلمسانه أبداً (مقاربان).")
    an_(d, "يتبع القاعدة التجريبية: 68-95-99.7.")
L.section("definitions", "2 · The Four Properties", "2 · الخصائص الأربع", s2)

def s3(d):
    ep(d, "The Empirical Rule (also called the 68-95-99.7 rule) is the single most useful fact about Normal distributions:")
    ap(d, "القاعدة التجريبية (تُسمى أيضاً قاعدة 68-95-99.7) هى أهم حقيقة عن التوزيعات الطبيعية:")
    kvtable(d, header=["Range / النطاق","% of data / النسبة","Meaning / المعنى"],
        rows=[
            ("μ ± 1σ", "~68%", "Most observations cluster here.\nمعظم المشاهدات هنا."),
            ("μ ± 2σ", "~95%", "Almost all observations.\nتقريباً كل المشاهدات."),
            ("μ ± 3σ", "~99.7%", "Anything outside is rare or an outlier.\nأى شىء خارج نادر أو متطرف."),
        ])
    eq(d, "Z  =  (x − μ) / σ",
       note_en="Z-score: how many std devs is x from the mean", note_ar="درجة Z: كم انحرافاً معيارياً يبعد x عن المتوسط")
    for f in [
        ("F-M03-L05-01","Empirical rule","القاعدة التجريبية","μ ± 1σ ≈ 68%  ·  μ ± 2σ ≈ 95%  ·  μ ± 3σ ≈ 99.7%",
         "Coverage under a Normal curve","التغطية تحت المنحنى الطبيعى",
         "Data is approximately Normal.","حين تكون البيانات طبيعية تقريباً."),
        ("F-M03-L05-02","Z-score","درجة Z","Z = (x − μ) / σ",
         "Standardised distance from mean","المسافة المعيارية من المتوسط",
         "Comparing values across different distributions.","مقارنة قيم من توزيعات مختلفة."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
L.section("formulas", "3 · The Empirical Rule & Z-score",
          "3 · القاعدة التجريبية ودرجة Z", s3)

def s4(d):
    callout(d, "example",
            "A retail bank's monthly customer transaction count is approximately Normal with mean 18 and std dev 4. What % of customers transact between 14 and 22 times?",
            "عدد المعاملات الشهرية لعملاء بنك تجزئة طبيعى تقريباً، بمتوسط 18 وانحراف 4. ما النسبة التى تتعامل بين 14 و 22 مرة؟")
    ep(d, "Range 14 to 22 = μ−σ to μ+σ. By the Empirical Rule, ≈ 68% of customers fall in this range.")
    ap(d, "النطاق 14 إلى 22 = μ−σ إلى μ+σ. بالقاعدة التجريبية، ≈ 68% من العملاء يقعون فى هذا النطاق.")
    callout(d, "example",
            "Customer X transacts 30 times. Is that unusual?\nZ = (30 − 18) / 4 = 3.\nValues with |Z| ≥ 3 occur < 0.3% of the time. Customer X is exceptional — definitely VIP-target material.",
            "عميل X يتعامل 30 مرة. هل ذلك غير عادى؟\nZ = (30 − 18) / 4 = 3.\nالقيم بـ |Z| ≥ 3 تحدث أقل من 0.3% من الوقت. العميل X استثنائى — هدف VIP بلا شك.")
L.section("example", "4 · Worked Example - Banking Transactions",
          "4 · مثال محلول - معاملات بنكية", s4)

def s5(d):
    callout(d, "use",
            "Marketing: a customer survey shows an NPS std dev of 15 around a mean of 40. Using the Empirical Rule, 95% of customers score between 10 and 70. Anything below 10 should trigger an outreach call.",
            "تسويق: استبيان NPS بانحراف 15 ومتوسط 40. بالقاعدة التجريبية، 95% من العملاء بين 10 و 70. أى شىء تحت 10 يستحق مكالمة متابعة.")
    callout(d, "use",
            "Operations: a call centre's daily call volume is Normal with mean 1,200 and std 200. Plan staffing for at least 1,200 + 2(200) = 1,600 calls — covers 95% of days.",
            "عمليات: مكالمات مركز الاتصالات اليومية طبيعية، متوسط 1,200 وانحراف 200. خطّط للموظفين 1,200 + 2(200) = 1,600 — يغطى 95% من الأيام.")
L.section("practice", "5 · Business Uses", "5 · استخدامات أعمال", s5)

def s6(d):
    callout(d, "trap",
            "Assuming all data is Normal. Income, time-on-page, transaction value — most business data is right-skewed. Always check the histogram before invoking the Empirical Rule.",
            "افتراض أن كل البيانات طبيعية. الدخل والوقت وقيمة المعاملة — معظم بيانات الأعمال منحرفة لليمين. تحقق من الرسم قبل تطبيق القاعدة التجريبية.")
    callout(d, "trap",
            "Confusing the data distribution with the sampling distribution. The CLT (next module) makes sample MEANS Normal even when the underlying data isn't.",
            "الخلط بين توزيع البيانات وتوزيع المعاينة. نظرية النهاية المركزية (الموديول القادم) تجعل متوسطات العينات طبيعية حتى لو لم تكن البيانات الأصلية كذلك.")
L.section("traps", "6 · Common Traps", "6 · الفخاخ الشائعة", s6)

L.add_memory_aid("Bell curve. Symmetric. Mean = Median = Mode. 68-95-99.7 rule.",
                 "منحنى جرسى. متماثل. المتوسط = الوسيط = المنوال. قاعدة 68-95-99.7.")
L.add_memory_aid("Z = (x − μ) / σ. |Z|>2 = unusual. |Z|>3 = exceptional.",
                 "Z = (x − μ) / σ. |Z|>2 = غير عادى. |Z|>3 = استثنائى.")
L.summary(
    en="The Normal distribution is symmetric, bell-shaped, and follows the 68-95-99.7 rule. It's the backbone of much of inferential statistics. The Z-score lets you compare values from any Normal distribution on a common scale. Always check that your data is actually Normal-shaped before applying these rules.",
    ar="التوزيع الطبيعى متماثل، جرسى الشكل، يتبع قاعدة 68-95-99.7. هو العمود الفقرى للإحصاء الاستدلالى. درجة Z تتيح مقارنة قيم من أى توزيع طبيعى على مقياس مشترك. تحقق دائماً أن بياناتك طبيعية فعلاً قبل تطبيق هذه القواعد."
)
L.render_memory()

L.quiz_q(qid="M03-L05-Q1", qtype="MCQ",
         q_en="Which is NOT a property of the Normal distribution?",
         q_ar="أى مما يلى ليس من خصائص التوزيع الطبيعى؟",
         opts_en=["Symmetric","Mean = Median = Mode","Always positive (no negative values)","Bell-shaped"],
         opts_ar=["متماثل","المتوسط = الوسيط = المنوال","موجب دائماً (لا قيم سالبة)","جرسى الشكل"],
         ans="C", exp_en="The Normal distribution extends from −∞ to +∞; it can have negative values.",
         exp_ar="الطبيعى يمتد من -∞ إلى +∞؛ يمكن أن يحوى قيماً سالبة.")

L.quiz_q(qid="M03-L05-Q2", qtype="Calc",
         q_en="IQ scores: μ=100, σ=15. What % of people score between 85 and 115?",
         q_ar="درجات الذكاء: μ=100، σ=15. ما النسبة بين 85 و 115؟",
         opts_en=["50%","68%","95%","99.7%"], opts_ar=["50%","68%","95%","99.7%"], ans="B",
         exp_en="85 = μ−σ, 115 = μ+σ → 68%.", exp_ar="85 = μ−σ، 115 = μ+σ ← 68%.")

L.quiz_q(qid="M03-L05-Q3", qtype="Calc",
         q_en="A test has μ=70, σ=10. Student scored 90. What is Z?",
         q_ar="اختبار μ=70، σ=10. الطالب حصل على 90. ما Z؟",
         opts_en=["0.5","1","2","3"], opts_ar=["0.5","1","2","3"], ans="C",
         exp_en="Z = (90−70)/10 = 2 → top ~2.5% of test-takers.",
         exp_ar="Z = (90−70)/10 = 2 ← أعلى ~2.5% من المختبَرين.",
         formula_used="Z = (x − μ) / σ")

L.quiz_q(qid="M03-L05-Q4", qtype="TrueFalse",
         q_en="True or False: All business data is Normally distributed.",
         q_ar="صح أم خطأ: كل بيانات الأعمال موزعة طبيعياً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Income, customer spend, and many metrics are skewed. Check first!",
         exp_ar="الدخل وإنفاق العملاء وكثير من المقاييس منحرفة. تحقق أولاً!")

L.quiz_q(qid="M03-L05-Q5", qtype="MCQ",
         q_en="Approximately what % of data lies between μ−2σ and μ+2σ?",
         q_ar="تقريباً ما النسبة بين μ−2σ و μ+2σ؟",
         opts_en=["50%","68%","95%","99.7%"], opts_ar=["50%","68%","95%","99.7%"], ans="C",
         exp_en="The 95% wing of 68-95-99.7.", exp_ar="جناح الـ 95% فى 68-95-99.7.")
L.render_quiz(); L.save()

print("L03-03, L03-04, L03-05 done.")
