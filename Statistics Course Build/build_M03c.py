import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 03 - Descriptive Statistics"

# ============================================================
# L03-06 — Kurtosis
# ============================================================
L = Lesson(
    lid="M03-L06", module=3,
    title_en="Kurtosis",
    title_ar="التفرطح",
    objectives_en=[
        "Understand what kurtosis measures (tail-heaviness).",
        "Distinguish Mesokurtic, Leptokurtic, Platykurtic distributions.",
        "Connect high kurtosis to tail-risk in finance and operations.",
    ],
    objectives_ar=[
        "فهم ما يقيسه التفرطح (ثقل الذيول).",
        "التمييز بين Mesokurtic و Leptokurtic و Platykurtic.",
        "ربط التفرطح المرتفع بمخاطر الذيول فى المالية والعمليات.",
    ],
    sources=["03 Descriptive Statistics/Kurtosis.mp4",
             "Slides PDF: page 28",
             "Glossary: Kurtosis"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Kurtosis measures the 'tailedness' of a distribution — how much weight sits in the extreme tails relative to the centre. High kurtosis means the data has fatter tails and more extreme outliers; low kurtosis means flatter, more uniform-looking tails.")
    ap(d, "التفرطح يقيس 'ثقل الذيول' فى التوزيع — كم من الكتلة تجلس فى الأطراف مقارنة بالمركز. التفرطح المرتفع يعنى ذيولاً أسمن وقيماً متطرفة أكثر؛ المنخفض يعنى ذيولاً مسطّحة أكثر تجانساً.")
    callout(d, "key",
            "Kurtosis is NOT about pointiness or peak height — it is about how much probability lives in the tails. Two distributions can have identical peaks but completely different kurtosis.",
            "التفرطح ليس عن حدّة القمة أو ارتفاعها — بل عن كم احتمال يعيش فى الذيول. توزيعان قد يتشاركان نفس القمة لكن بتفرطح مختلف.")
L.section("intro", "1 · What Kurtosis Measures", "1 · ما يقيسه التفرطح", s1)

def s2(d):
    kvtable(d, header=["Type / النوع","Excess kurtosis / التفرطح الزائد","Tail behaviour / سلوك الذيول","Example / مثال"],
        rows=[
            ("Mesokurtic\nمتوسط التفرطح","≈ 0","Normal tails\nذيول طبيعية","Heights, IQ scores\nالأطوال، درجات الذكاء"),
            ("Leptokurtic\nمفرطح (تفرطح زائد)",">> 0","Fat tails - frequent extremes\nذيول سمينة - تطرف متكرر",
             "S&P 500 returns, crypto\nعوائد المؤشرات، العملات الرقمية"),
            ("Platykurtic\nمسطح","< 0","Thin tails - rare extremes\nذيول رفيعة - تطرف نادر",
             "Bond prices, utility bills\nأسعار السندات، فواتير المرافق"),
        ])
    eq(d, "Excess kurtosis  =  Kurtosis − 3",
       note_en="Excess kurtosis: 0 means Normal-like tails", note_ar="التفرطح الزائد: 0 يعنى ذيولاً طبيعية")
    L.data["formulas"].append({"id":"F-M03-L06-01","name_en":"Excess kurtosis","name_ar":"التفرطح الزائد",
        "expr":"Excess kurtosis = Kurtosis − 3",
        "where_en":"Normal distribution has kurtosis = 3, so excess = 0","where_ar":"الطبيعى تفرطحه = 3، فالزائد = 0",
        "when_en":"Comparing tail risk across distributions.","when_ar":"مقارنة مخاطر الذيول بين توزيعات."})
L.section("definitions", "2 · The Three Types", "2 · الأنواع الثلاثة", s2)

def s3(d):
    callout(d, "example",
            "S&P 500 daily returns: ~10% annual std dev, BUT kurtosis is high. The 1987 'Black Monday' was a 22 std-dev event — under a Normal model, this should happen once every 10^50 years. Reality: it happened in 30 years of data. Fat tails are deadly.",
            "عوائد S&P 500 اليومية: ~10% انحراف سنوى، لكن التفرطح مرتفع. 'الإثنين الأسود' 1987 كان حدثاً بـ 22 انحراف. تحت نموذج طبيعى، يحدث مرة كل 10^50 سنة. الواقع: حدث فى 30 سنة. الذيول السمينة قاتلة.")
L.section("example", "3 · Worked Example - Fat Tails Kill",
          "3 · مثال محلول - الذيول السمينة قاتلة", s3)

def s4(d):
    callout(d, "use",
            "Risk management: a portfolio that looks 'safe' under Normal-distribution assumptions can blow up if returns are leptokurtic. Modern risk models add 'tail risk' adjustments.",
            "إدارة المخاطر: محفظة تبدو 'آمنة' تحت افتراضات الطبيعى قد تنفجر إذا كانت العوائد leptokurtic. النماذج الحديثة تضيف تعديلات لـ 'مخاطر الذيول'.")
    callout(d, "use",
            "Operations: e-commerce checkout times. If kurtosis is high, a few extremely slow checkouts (database glitches) dominate the experience. Median + 99th percentile is more informative than just the mean.",
            "العمليات: أوقات الدفع فى التجارة الإلكترونية. إذا كان التفرطح عالياً، قلة من المعاملات شديدة البطء تهيمن على التجربة. الوسيط + الربيع الـ 99 أكثر فائدة من المتوسط فقط.")
L.section("practice", "4 · Business Uses", "4 · استخدامات أعمال", s4)

def s5(d):
    callout(d, "trap",
            "Treating kurtosis as 'peak height'. The peak can be high or low independently of kurtosis. Focus on the tails.",
            "اعتبار التفرطح 'ارتفاع القمة'. القمة قد تكون مرتفعة أو منخفضة بغض النظر عن التفرطح. ركز على الذيول.")
L.section("traps", "5 · Common Trap", "5 · فخّ شائع", s5)

L.add_memory_aid("Kurtosis = tail fatness. Leptokurtic = fat tails (risk!). Platykurtic = thin tails (stable).",
                 "التفرطح = سمنة الذيول. Leptokurtic = ذيول سمينة (مخاطرة!). Platykurtic = ذيول رفيعة (مستقر).")
L.summary(
    en="Kurtosis is about tails, not peaks. Leptokurtic distributions hide rare but catastrophic events; platykurtic distributions are predictable and uniform. In finance and operations, ignoring kurtosis means underestimating tail-risk — the very disasters statistics is supposed to warn you about.",
    ar="التفرطح عن الذيول لا القمم. التوزيعات leptokurtic تخفى أحداثاً نادرة لكن كارثية؛ platykurtic متوقعة ومنتظمة. فى المالية والعمليات، تجاهل التفرطح يعنى الاستهانة بمخاطر الذيول — الكوارث التى من المفترض أن ينذرنا عنها الإحصاء."
)
L.render_memory()

L.quiz_q(qid="M03-L06-Q1", qtype="MCQ",
         q_en="Leptokurtic distributions have:",
         q_ar="التوزيعات leptokurtic لها:",
         opts_en=["Thin tails (rare extremes)","Fat tails (frequent extremes)","Equal tails","No tails"],
         opts_ar=["ذيول رفيعة (تطرف نادر)","ذيول سمينة (تطرف متكرر)","ذيول متساوية","لا ذيول"],
         ans="B", exp_en="Leptokurtic = high kurtosis = fat tails.", exp_ar="Leptokurtic = تفرطح مرتفع = ذيول سمينة.")

L.quiz_q(qid="M03-L06-Q2", qtype="MCQ",
         q_en="Normal distribution kurtosis is:",
         q_ar="تفرطح التوزيع الطبيعى:",
         opts_en=["0","1","3","100"], opts_ar=["0","1","3","100"], ans="C",
         exp_en="Kurtosis of Normal = 3 (excess kurtosis = 0).",
         exp_ar="تفرطح الطبيعى = 3 (الزائد = 0).")

L.quiz_q(qid="M03-L06-Q3", qtype="MCQ",
         q_en="A platykurtic distribution is:",
         q_ar="التوزيع platykurtic:",
         opts_en=["High kurtosis, fat tails","Low kurtosis, thin tails",
                  "Mean = Median","Always Normal"],
         opts_ar=["تفرطح مرتفع، ذيول سمينة","تفرطح منخفض، ذيول رفيعة","المتوسط = الوسيط","طبيعى دائماً"],
         ans="B", exp_en="Platykurtic = flat tails, fewer extremes.",
         exp_ar="Platykurtic = ذيول مسطحة، تطرفات أقل.")

L.quiz_q(qid="M03-L06-Q4", qtype="TrueFalse",
         q_en="True or False: Kurtosis describes the height of the peak.",
         q_ar="صح أم خطأ: التفرطح يصف ارتفاع القمة.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Kurtosis is about tail-weight, not peak height.",
         exp_ar="التفرطح عن ثقل الذيول، لا ارتفاع القمة.")

L.quiz_q(qid="M03-L06-Q5", qtype="MCQ",
         q_en="In risk management, a high-kurtosis return distribution implies:",
         q_ar="فى إدارة المخاطر، توزيع عوائد عالى التفرطح يعنى:",
         opts_en=["Smooth predictable returns","Higher chance of extreme events than a Normal model predicts",
                  "No risk","Always positive returns"],
         opts_ar=["عوائد ناعمة متوقعة","فرصة أعلى لأحداث متطرفة أكثر مما يتوقعه نموذج طبيعى",
                  "لا مخاطر","عوائد موجبة دائماً"],
         ans="B", exp_en="Fat tails = real-world tail risk that Normal models miss.",
         exp_ar="ذيول سمينة = مخاطر ذيول حقيقية تفوّتها نماذج الطبيعى.")
L.render_quiz(); L.save()

# ============================================================
# L03-07 — Asymmetric Distributions & Skewness
# ============================================================
L = Lesson(
    lid="M03-L07", module=3,
    title_en="Asymmetric Distributions and Skewness",
    title_ar="التوزيعات غير المتماثلة والالتواء",
    objectives_en=[
        "Define skewness and recognise positive vs negative skew.",
        "Predict the relative position of Mean, Median, and Mode from skew direction.",
        "Apply log-transforms when skew is too strong.",
    ],
    objectives_ar=[
        "تعريف الالتواء وتمييز الالتواء الموجب والسالب.",
        "توقع الترتيب النسبى للمتوسط والوسيط والمنوال من اتجاه الالتواء.",
        "تطبيق تحويلات لوغاريتمية حين يكون الالتواء حاداً.",
    ],
    sources=["03 Descriptive Statistics/Asymmetric (and Other) Distributions.mp4",
             "Slides PDF: pages 29-30",
             "Glossary: Asymmetric, Skewness"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Most real-world business distributions are NOT symmetric. Customer spend, time on site, transaction amounts, loan sizes — almost all are skewed to the right. Skewness measures and labels the asymmetry.")
    ap(d, "معظم توزيعات الأعمال الحقيقية ليست متماثلة. إنفاق العملاء، وقت الموقع، المعاملات، أحجام القروض — تقريباً كلها منحرفة لليمين. الالتواء يقيس ويسمّى عدم التماثل.")
L.section("intro", "1 · Symmetry is the Exception", "1 · التماثل هو الاستثناء", s1)

def s2(d):
    kvtable(d, header=["Skew direction / اتجاه الالتواء","Tail / الذيل","Mean vs Median / المتوسط مقابل الوسيط","Example / مثال"],
        rows=[
            ("Symmetric (no skew)\nمتماثل","Both equal\nمتساويان","Mean = Median\nالمتوسط = الوسيط",
             "Heights\nالأطوال"),
            ("Right-skewed (positive)\nمنحرف لليمين (موجب)","Long right tail\nذيل أيمن طويل",
             "Mean > Median\nالمتوسط > الوسيط","Income, transactions\nالدخل، المعاملات"),
            ("Left-skewed (negative)\nمنحرف لليسار (سالب)","Long left tail\nذيل أيسر طويل",
             "Mean < Median\nالمتوسط < الوسيط","Age at death, easy-exam scores\nالعمر عند الوفاة، اختبارات سهلة"),
        ])
L.section("definitions", "2 · Three Cases", "2 · ثلاث حالات", s2)

def s3(d):
    eq(d, "Pearson's first skewness  =  (Mean − Mode) / σ",
       note_en="Quick descriptive measure", note_ar="مقياس وصفى سريع")
    eq(d, "Pearson's second  =  3 · (Mean − Median) / σ",
       note_en="Often easier - uses Median", note_ar="غالباً أسهل - يستخدم الوسيط")
    eq(d, "Moment skewness  =  [Σ(xᵢ − x̄)³ / n] / σ³",
       note_en="Formal definition (Excel: SKEW)", note_ar="التعريف الرسمى (Excel: SKEW)")
    L.data["formulas"].append({"id":"F-M03-L07-01","name_en":"Pearson skewness","name_ar":"التواء بيرسون",
        "expr":"Skew ≈ 3·(Mean − Median)/σ","where_en":"σ = standard deviation","where_ar":"σ = الانحراف المعيارى",
        "when_en":"Quick check on skew direction and strength.","when_ar":"فحص سريع للالتواء واتجاهه."})
L.section("formulas", "3 · How to Measure Skewness", "3 · كيف نقيس الالتواء", s3)

def s4(d):
    callout(d, "example",
            "10 customer order values (EGP): 50, 80, 90, 100, 110, 120, 130, 140, 200, 2,500.\nMean = 352. Median = 115. Mean − Median = 237.\nClearly right-skewed (positive) — pulled by the 2,500 outlier.",
            "10 قيم طلبات (ج.م): 50، 80، 90، 100، 110، 120، 130، 140، 200، 2,500.\nالمتوسط = 352. الوسيط = 115. الفرق = 237.\nمنحرف لليمين بوضوح (موجب) — مسحوب بالقيمة المتطرفة 2,500.")
    ep(d, "Decision: report MEDIAN (115) as the typical order value, not MEAN.")
    ap(d, "القرار: اعرض الوسيط (115) كقيمة الطلب النموذجية، لا المتوسط.")
L.section("example", "4 · Worked Example - Order Values",
          "4 · مثال محلول - قيم الطلبات", s4)

def s5(d):
    callout(d, "use",
            "Real-estate prices in a city: right-skewed (luxury properties pull mean up). Reports should always show Median home price, not Mean.",
            "أسعار المنازل فى مدينة: منحرفة لليمين (الفاخرة تسحب المتوسط). تقارير الإسكان يجب أن تعرض الوسيط لا المتوسط.")
    callout(d, "use",
            "Page-load time: right-skewed (most pages fast, a few very slow). The 95th percentile or Median + p99 tells the real user experience better than the mean.",
            "وقت تحميل الصفحة: منحرف لليمين (معظم الصفحات سريعة، بعضها بطىء جداً). الربيع الـ 95 أو الوسيط + p99 يحكى تجربة المستخدم الحقيقية أفضل من المتوسط.")
    callout(d, "use",
            "Log-transform trick: when data is heavily right-skewed (income, sales), take log(x). The transformed distribution is often nearly Normal, unlocking the tools that assume normality.",
            "حيلة التحويل اللوغاريتمى: حين يكون الانحراف لليمين حاداً (دخل، مبيعات)، خذ log(x). التوزيع المحوّل غالباً ما يكون شبه طبيعى، مما يفتح أدوات تفترض الطبيعى.")
L.section("practice", "5 · Business Uses", "5 · استخدامات أعمال", s5)

def s6(d):
    callout(d, "trap",
            "Reporting the Mean of right-skewed data as 'the typical customer'. Use Median; flag outliers; consider log-transform for modelling.",
            "عرض متوسط بيانات منحرفة لليمين كـ 'العميل النموذجى'. استخدم الوسيط؛ ضع علامة على المتطرفات؛ فكّر فى التحويل اللوغاريتمى للنمذجة.")
L.section("traps", "6 · Common Trap", "6 · فخّ شائع", s6)

L.add_memory_aid("Right-skew → Mean > Median (mean pulled right).",
                 "التواء يمين ← المتوسط > الوسيط (المتوسط مسحوب يميناً).")
L.add_memory_aid("Left-skew → Mean < Median (mean pulled left).",
                 "التواء يسار ← المتوسط < الوسيط.")
L.add_memory_aid("Heavy right-skew? Try log-transform for analysis.",
                 "التواء يمين حاد؟ جرّب التحويل اللوغاريتمى.")
L.summary(
    en="Skewness measures asymmetry. Right-skewed (positive) distributions have long right tails and Mean > Median — typical of income, prices, transactions. Left-skewed is rarer in business. When skew is heavy, report the Median, watch for outliers, and consider transforming the data before running tools that assume symmetry.",
    ar="الالتواء يقيس عدم التماثل. التوزيعات منحرفة اليمين (موجبة) لها ذيول يمنى طويلة و المتوسط > الوسيط — مثل الدخل والأسعار والمعاملات. الانحراف لليسار أندر فى الأعمال. حين يكون الالتواء حاداً، اعرض الوسيط، تابع المتطرفات، وفكّر فى تحويل البيانات قبل أدوات تفترض التماثل."
)
L.render_memory()

L.quiz_q(qid="M03-L07-Q1", qtype="MCQ",
         q_en="A distribution with a long right tail is:",
         q_ar="توزيع له ذيل أيمن طويل:",
         opts_en=["Symmetric","Left-skewed","Right-skewed","Kurtotic only"],
         opts_ar=["متماثل","منحرف يسار","منحرف يمين","متفرطح فقط"], ans="C",
         exp_en="Long right tail = right (positive) skew.", exp_ar="ذيل أيمن طويل = التواء يمين (موجب).")

L.quiz_q(qid="M03-L07-Q2", qtype="MCQ",
         q_en="Customer income data shows Mean = $80k, Median = $55k. Skew direction?",
         q_ar="دخل عملاء: المتوسط = 80 ألف، الوسيط = 55 ألف. اتجاه الالتواء؟",
         opts_en=["Symmetric","Right (positive)","Left (negative)","Cannot tell"],
         opts_ar=["متماثل","يمين (موجب)","يسار (سالب)","لا يمكن"], ans="B",
         exp_en="Mean > Median → right-skewed (a few high earners pull the mean up).",
         exp_ar="المتوسط > الوسيط ← منحرف يمين (قلة من ذوى الدخل المرتفع تسحب المتوسط).")

L.quiz_q(qid="M03-L07-Q3", qtype="MCQ",
         q_en="For right-skewed business data, the BEST summary measure is:",
         q_ar="لبيانات منحرفة يميناً، أفضل مقياس تلخيصى:",
         opts_en=["Mean","Median","Mode","Range"], opts_ar=["المتوسط","الوسيط","المنوال","المدى"], ans="B",
         exp_en="Median is robust to the long right tail.", exp_ar="الوسيط مقاوم للذيل الأيمن الطويل.")

L.quiz_q(qid="M03-L07-Q4", qtype="TrueFalse",
         q_en="True or False: Left-skewed means the Mean is greater than the Median.",
         q_ar="صح أم خطأ: الالتواء اليسارى يعنى أن المتوسط أكبر من الوسيط.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Left-skew: Mean < Median.", exp_ar="التواء يسار: المتوسط < الوسيط.")

L.quiz_q(qid="M03-L07-Q5", qtype="MCQ",
         q_en="A common trick for heavily right-skewed data is to apply:",
         q_ar="حيلة شائعة لبيانات منحرفة بشدة يميناً:",
         opts_en=["Squaring","Log-transform","Multiply by 0","Reverse the data"],
         opts_ar=["تربيع","تحويل لوغاريتمى","الضرب فى صفر","عكس البيانات"], ans="B",
         exp_en="log(x) compresses the long tail and often yields a near-Normal shape.",
         exp_ar="log(x) يضغط الذيل الطويل وغالباً ينتج شكلاً قريباً من الطبيعى.")
L.render_quiz(); L.save()

# ============================================================
# L03-08 — Excel Workshop: Univariate Analysis
# ============================================================
L = Lesson(
    lid="M03-L08", module=3,
    title_en="Excel Workshop - Univariate Analysis",
    title_ar="ورشة Excel - التحليل أحادى المتغير",
    objectives_en=[
        "Run a full univariate descriptive analysis in Excel.",
        "Use the Data Analysis ToolPak to get all measures in one shot.",
        "Read the output: mean, median, std dev, kurtosis, skewness.",
    ],
    objectives_ar=[
        "تشغيل تحليل وصفى أحادى المتغير كامل فى Excel.",
        "استخدام Data Analysis ToolPak للحصول على كل المقاييس مرة واحدة.",
        "قراءة المخرجات: المتوسط، الوسيط، الانحراف، التفرطح، الالتواء.",
    ],
    sources=["03 Descriptive Statistics/Excel - Univariate Analysis.mp4 (×5 parts)",
             "Workbook: 02 - Descriptive Statistics.xlsx",
             "Solution: 02 - Complete - Descriptive Statistics.xlsx"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Excel can compute every univariate descriptive statistic with one click — if you know where to find the Data Analysis ToolPak. This lesson walks through the entire flow with the course's S&P 500 daily-returns workbook.")
    ap(d, "يمكن لـ Excel حساب كل إحصاء وصفى أحادى المتغير بنقرة واحدة — إذا عرفت كيف تشغّل Data Analysis ToolPak. هذا الدرس يمر بكل الخطوات بملف العوائد اليومية لـ S&P 500 الخاص بالدورة.")
L.section("intro", "1 · The Tool You're About to Master",
          "1 · الأداة التى ستتقنها", s1)

def s2(d):
    h3(d, "Step 1 - Enable the ToolPak", "خطوة 1 - تفعيل ToolPak")
    en_(d, "File → Options → Add-ins → Manage: Excel Add-ins → Go.")
    en_(d, "Tick 'Analysis ToolPak' → OK. Now a 'Data Analysis' button appears in the Data tab.")
    an_(d, "ملف ← خيارات ← الإضافات ← إدارة: إضافات Excel ← انتقل.")
    an_(d, "حدّد 'Analysis ToolPak' ← موافق. سيظهر زر 'تحليل البيانات' فى تبويب Data.")

    h3(d, "Step 2 - Compute the 'Change' column", "خطوة 2 - حساب عمود 'التغير'")
    ep(d, "Formula: =(Price_today − Price_yesterday) / Price_yesterday")
    ap(d, "الصيغة: =(سعر_اليوم − سعر_الأمس) / سعر_الأمس")
    eq(d, "Change  =  (Bᵢ − Bᵢ₋₁) / Bᵢ₋₁")
    ep(d, "Fill down to all 1,260 rows.")
    ap(d, "اسحب لأسفل لكل 1,260 صفاً.")

    h3(d, "Step 3 - Run Descriptive Statistics", "خطوة 3 - تشغيل الإحصاء الوصفى")
    en_(d, "Data tab → Data Analysis → Descriptive Statistics → OK.")
    en_(d, "Input range = 'Change' column (with header).")
    en_(d, "Tick 'Labels in first row'.")
    en_(d, "Tick 'Summary statistics' AND 'Kth largest/smallest' if you want extremes.")
    en_(d, "Choose Output range → some empty cell → OK.")
    an_(d, "Data ← تحليل البيانات ← Descriptive Statistics ← موافق.")
    an_(d, "نطاق الإدخال = عمود 'Change' (مع العنوان).")
    an_(d, "حدّد 'Labels in first row'.")
    an_(d, "حدّد 'Summary statistics' و 'Kth largest/smallest' إن أردت الأطراف.")
    an_(d, "اختر مكان الإخراج ← موافق.")
L.section("concept", "2 · Step-by-Step Workflow", "2 · سير العمل خطوة بخطوة", s2)

def s3(d):
    ep(d, "From the course's solution file (02 - Complete - Descriptive Statistics.xlsx), the Excel output for the Change column looks like this:")
    ap(d, "من ملف الحل (02 - Complete - Descriptive Statistics.xlsx)، مخرجات Excel لعمود Change تبدو هكذا:")
    kvtable(d, header=["Measure / المقياس","Value / القيمة","Interpretation / التفسير"],
        rows=[
            ("Mean\nالمتوسط","0.00037 (≈ 0.04% daily)",
             "Average daily return.\nمتوسط العائد اليومى."),
            ("Standard Error\nالخطأ المعيارى","0.000371",
             "SE = σ / √n - precision of the mean.\nدقة المتوسط."),
            ("Median\nالوسيط","0.000842",
             "Median > Mean → slight left skew.\nالوسيط > المتوسط ← التواء يسار طفيف."),
            ("Mode\nالمنوال","0","Most frequent change: 0 (flat day).\nأكثر تكراراً: 0 (يوم ثابت)."),
            ("Standard Deviation\nالانحراف المعيارى","0.0132 (≈ 1.3% daily)",
             "Daily volatility.\nالتقلب اليومى."),
            ("Sample Variance\nتباين العينة","0.000174",
             "σ² in squared returns.\nبالعوائد المربّعة."),
            ("Kurtosis\nالتفرطح","11.83",
             "Excess far above 0 → very fat tails!\nزائد أعلى من 0 بكثير ← ذيول سمينة جداً!"),
        ])
L.section("example", "3 · Reading the Output", "3 · قراءة المخرجات", s3)

def s4(d):
    callout(d, "key",
            "Story from the numbers: average daily return is tiny (0.04%); typical daily move is about ±1.3% (one std dev); but kurtosis ≈ 12 means crashes and rallies happen far more often than a Normal model predicts. This is exactly why options markets price tail risk.",
            "القصة من الأرقام: متوسط العائد اليومى ضئيل (0.04%)؛ التحرك اليومى المعتاد ±1.3% (انحراف واحد)؛ لكن تفرطح ≈ 12 يعنى الانهيارات والصعود تحدث أكثر بكثير مما يتوقع نموذج طبيعى. لذا تسعّر أسواق الخيارات مخاطر الذيول.")
L.section("interpret", "4 · The Story the Numbers Tell",
          "4 · القصة التى ترويها الأرقام", s4)

def s5(d):
    callout(d, "use",
            "Bring this workflow to ANY business dataset: monthly revenues, daily transactions, helpdesk ticket times. The same one-click summary gives you a complete univariate picture before you spend hours on detailed dashboards.",
            "طبّق سير العمل هذا على أى بيانات: إيرادات شهرية، معاملات يومية، أوقات تذاكر الدعم. نفس الملخص بنقرة واحدة يعطيك صورة كاملة قبل ساعات على لوحات تفصيلية.")
    h3(d, "Quick Excel function reference", "مرجع سريع لدوال Excel")
    eb(d, "Mean: =AVERAGE(range)")
    eb(d, "Median: =MEDIAN(range)")
    eb(d, "Mode (one): =MODE.SNGL(range)")
    eb(d, "Sample std dev: =STDEV.S(range)")
    eb(d, "Population std dev: =STDEV.P(range)")
    eb(d, "Variance (sample): =VAR.S(range)")
    eb(d, "Skewness: =SKEW(range)")
    eb(d, "Kurtosis (excess): =KURT(range)")
    eb(d, "Quartiles: =QUARTILE.INC(range, 1 or 3)")
    eb(d, "Percentile: =PERCENTILE.INC(range, 0.95)")
L.section("practice", "5 · Bring This Anywhere", "5 · طبّقها فى أى مكان", s5)

L.add_memory_aid("ToolPak → Descriptive Statistics → one click = full univariate report.",
                 "ToolPak ← Descriptive Statistics ← نقرة واحدة = تقرير كامل أحادى المتغير.")
L.summary(
    en="Excel's Data Analysis ToolPak runs a complete univariate descriptive analysis in seconds. The output gives you mean, median, std dev, skewness, kurtosis and more — everything from this module's previous lessons in one table. Practise on the course workbook, then bring the workflow to your own data.",
    ar="Data Analysis ToolPak فى Excel ينفّذ تحليلاً وصفياً أحادى المتغير كاملاً فى ثوانٍ. المخرجات تعطيك المتوسط والوسيط والانحراف والالتواء والتفرطح وأكثر — كل ما سبق فى دروس هذا الموديول فى جدول واحد. تدرّب على ملف الدورة، ثم انقل سير العمل إلى بياناتك."
)
L.render_memory()

L.quiz_q(qid="M03-L08-Q1", qtype="MCQ",
         q_en="Excel function for sample standard deviation:",
         q_ar="دالة Excel للانحراف المعيارى للعينة:",
         opts_en=["=STDEV.P","=STDEV.S","=STD","=DEVSQ"],
         opts_ar=["=STDEV.P","=STDEV.S","=STD","=DEVSQ"], ans="B",
         exp_en="STDEV.S divides by n−1 (sample). STDEV.P uses N (population).",
         exp_ar="STDEV.S يقسم على n−1. STDEV.P على N.")

L.quiz_q(qid="M03-L08-Q2", qtype="MCQ",
         q_en="To enable Data Analysis ToolPak you go to:",
         q_ar="لتفعيل Data Analysis ToolPak تذهب إلى:",
         opts_en=["Insert > Add-ins","File > Options > Add-ins > Excel Add-ins > Go","Home > Conditional Formatting","Data > PivotTable"],
         opts_ar=["إدراج > الإضافات","ملف > خيارات > الإضافات > إضافات Excel > انتقل","الصفحة الرئيسية > التنسيق الشرطى","البيانات > Pivot"],
         ans="B", exp_en="File → Options → Add-ins is the path.",
         exp_ar="ملف ← خيارات ← الإضافات هو المسار.")

L.quiz_q(qid="M03-L08-Q3", qtype="MCQ",
         q_en="In the S&P 500 daily returns: kurtosis = 11.83. This tells you:",
         q_ar="فى العوائد اليومية لـ S&P 500: التفرطح = 11.83. يخبرك أن:",
         opts_en=["Returns are perfectly Normal","Returns have very fat tails (extreme events more common than Normal)",
                  "There is no risk","Variance is zero"],
         opts_ar=["العوائد طبيعية تماماً","العوائد لها ذيول سمينة جداً (تطرف أكثر من الطبيعى)",
                  "لا توجد مخاطرة","التباين صفر"],
         ans="B", exp_en="Excess kurtosis far above 0 = fat tails = real-world tail risk.",
         exp_ar="تفرطح زائد >> 0 = ذيول سمينة = مخاطر ذيول.")

L.quiz_q(qid="M03-L08-Q4", qtype="Calc",
         q_en="To compute the 95th percentile of values in A2:A100, use:",
         q_ar="لحساب الربيع الـ 95 لقيم A2:A100، استخدم:",
         opts_en=["=PERCENTILE.INC(A2:A100, 95)","=PERCENTILE.INC(A2:A100, 0.95)",
                  "=QUARTILE(A2:A100, 95)","=MEAN(A2:A100)"],
         opts_ar=["=PERCENTILE.INC(A2:A100, 95)","=PERCENTILE.INC(A2:A100, 0.95)",
                  "=QUARTILE(A2:A100, 95)","=MEAN(A2:A100)"], ans="B",
         exp_en="Percentile argument is a decimal between 0 and 1.",
         exp_ar="معامل الربيع رقم عشرى بين 0 و 1.")

L.quiz_q(qid="M03-L08-Q5", qtype="TrueFalse",
         q_en="True or False: STDEV.P and STDEV.S give the same result.",
         q_ar="صح أم خطأ: STDEV.P و STDEV.S يعطيان نفس النتيجة.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="P divides by N, S divides by n−1 — slightly different.",
         exp_ar="P يقسم على N، S على n−1 — مختلفان قليلاً.")
L.render_quiz(); L.save()

print("L03-06, L03-07, L03-08 done.")
