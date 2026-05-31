import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 04 - Inferential Statistics"

# ============================================================
# L04-07 — Determining Sample Size
# ============================================================
L = Lesson(
    lid="M04-L07", module=4,
    title_en="Determining Sample Size",
    title_ar="تحديد حجم العينة",
    objectives_en=[
        "Identify the four factors that drive required sample size.",
        "Use the course's sample-size table to plan studies.",
        "Recognise when you need more data (or fewer assumptions).",
    ],
    objectives_ar=[
        "تحديد العوامل الأربعة التى تحرّك حجم العينة المطلوب.",
        "استخدام جدول حجم العينة من الدورة لتخطيط الدراسات.",
        "إدراك متى تحتاج بيانات أكثر (أو افتراضات أقل).",
    ],
    sources=["04 Inferential Statistics/Determining Sample Size.mp4",
             "Slides PDF: page 54",
             "Glossary: Sample Size"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The single most asked question in any project: 'How many should I survey / test / measure?' There's no universal answer, but four factors drive it.")
    ap(d, "السؤال الأكثر تكراراً فى أى مشروع: 'كم يجب أن أستطلع/أختبر/أقيس؟' لا توجد إجابة عالمية، لكن أربعة عوامل تحرّكه.")
L.section("intro", "1 · The Universal Question", "1 · السؤال العالمى", s1)

def s2(d):
    kvtable(d, header=["Factor / العامل","Direction / الاتجاه","Why / لماذا"],
        rows=[
            ("Margin of error / هامش الخطأ",
             "Smaller error → larger n.\nخطأ أصغر ← n أكبر.",
             "More precision needs more data.\nدقة أعلى تحتاج بيانات أكثر."),
            ("Confidence level / مستوى الثقة",
             "Higher confidence → larger n.\nثقة أعلى ← n أكبر.",
             "More certainty needs more data.\nيقين أكثر يحتاج أكثر."),
            ("Population variance / تباين المجتمع",
             "Higher variance → larger n.\nتباين أعلى ← n أكبر.",
             "Noisier data needs more samples.\nبيانات أكثر ضوضاء تحتاج أكثر."),
            ("Rarity of event / ندرة الحدث",
             "Rarer event → larger n.\nأندر ← n أكبر.",
             "Need enough cases of rare event.\nنحتاج حالات كافية من الحدث النادر."),
        ])
L.section("definitions", "2 · The Four Driving Factors",
          "2 · العوامل الأربعة المحرّكة", s2)

def s3(d):
    ep(d, "Practical rule-of-thumb table from the course slides (for proportion-based studies with large populations >5,000):")
    ap(d, "جدول قاعدة عامة من شرائح الدورة (لدراسات النسب مع مجتمعات كبيرة >5,000):")
    kvtable(d, header=["Margin of error / هامش الخطأ","Pop ≥5000","2,500","1,000","500","200"],
        rows=[
            ("±10%", "96", "93", "88", "81", "65"),
            ("±7.5%", "171", "160", "146", "127", "92"),
            ("±5%", "384", "333", "278", "217", "132"),
            ("±3%", "1,067", "748", "516", "341", "169"),
        ])
    callout(d, "key",
            "The famous 'magic number 384' for ±5% margin and 95% confidence with a large population. This is why most political polls survey ~400-1,200 people.",
            "الرقم السحرى '384' لهامش ±5% وثقة 95% مع مجتمع كبير. لذا تستطلع معظم استطلاعات الرأى السياسية ~400-1,200 شخص.")
    eq(d, "n  =  (Z² · p · (1−p))  /  E²",
       note_en="Sample size for proportion (large population)",
       note_ar="حجم العينة لنسبة (مجتمع كبير)")
    L.data["formulas"].append({"id":"F-M04-L07-01","name_en":"Sample size for proportion","name_ar":"حجم العينة لنسبة",
        "expr":"n = Z²·p·(1−p) / E²",
        "where_en":"Z=critical value (1.96 for 95%), p=estimated proportion, E=margin of error",
        "where_ar":"Z=القيمة الحرجة (1.96 لـ 95%)، p=النسبة المتوقعة، E=هامش الخطأ",
        "when_en":"Planning surveys with proportions.","when_ar":"تخطيط استبيانات بنسب."})
L.section("formulas", "3 · The Sample-Size Table & Formula",
          "3 · الجدول والصيغة", s3)

def s4(d):
    callout(d, "example",
            "Marketing wants to estimate the % of customers willing to try a new product, with ±5% error and 95% confidence. Worst-case p = 0.5.\nn = 1.96² · 0.5 · 0.5 / 0.05² = 3.8416 · 0.25 / 0.0025 ≈ 385.\nMatches the table's 384.",
            "التسويق يريد تقدير نسبة العملاء المستعدين لتجربة منتج جديد، بدقة ±5% وثقة 95%. أسوأ حالة p = 0.5.\nn = 1.96² · 0.5 · 0.5 / 0.05² = 3.8416 · 0.25 / 0.0025 ≈ 385.\nيطابق الجدول 384.")
L.section("example", "4 · Worked Example - The Magic 384",
          "4 · مثال محلول - الرقم السحرى 384", s4)

def s5(d):
    callout(d, "use",
            "Customer satisfaction surveys: ±3% margin for a 5-million-customer base needs ~1,067 random responses. Email blast 5,000 and expect ~20% response rate.",
            "استبيانات رضا العملاء: هامش ±3% لـ 5 مليون عميل يحتاج ~1,067 رد عشوائى. أرسل لـ 5,000 وتوقع معدل رد ~20%.")
    callout(d, "use",
            "Quality control: if defect rate is rare (~1%), you need a larger sample to see enough defects. Switch from rule-of-thumb to formula with the actual p ~ 0.01.",
            "ضبط الجودة: لو كان معدل العيوب نادراً (~1%)، تحتاج عينة أكبر لرؤية عيوب كافية. انتقل من القاعدة إلى الصيغة بالنسبة الفعلية p ~ 0.01.")
L.section("practice", "5 · Business Planning Uses",
          "5 · استخدامات تخطيط الأعمال", s5)

def s6(d):
    callout(d, "trap",
            "Using a sample-size formula for proportions when your outcome is a mean. Different formula (uses σ): n = (Z·σ/E)².",
            "استخدام صيغة النسب حين تكون النتيجة متوسطاً. الصيغة مختلفة (تستخدم σ): n = (Z·σ/E)².")
    callout(d, "trap",
            "Assuming p = 0.5 to be safe makes the sample bigger than needed if true p is far from 0.5. Use a pilot to refine.",
            "افتراض p = 0.5 للأمان يجعل العينة أكبر من اللازم إذا كانت p بعيدة عن 0.5. استخدم تجربة استكشافية للضبط.")
L.section("traps", "6 · Common Traps", "6 · الفخاخ الشائعة", s6)

L.add_memory_aid("Four levers: margin, confidence, variance, rarity. All push n upward.",
                 "أربعة عوامل: الهامش، الثقة، التباين، الندرة. كلها تدفع n لأعلى.")
L.add_memory_aid("384 = magic number for ±5% margin, 95% confidence, large population.",
                 "384 = الرقم السحرى لـ ±5%، ثقة 95%، مجتمع كبير.")
L.summary(
    en="Sample size is the bridge between budget and statistical power. Four factors push it up: tighter margins, higher confidence, higher variance, rarer events. The course's practical table covers the common cases; for unusual outcomes, fall back to formulas like n = Z²·p·(1−p)/E² (proportions) or n = (Z·σ/E)² (means). Plan sample size BEFORE collecting data — not after.",
    ar="حجم العينة هو الجسر بين الميزانية والقوة الإحصائية. أربعة عوامل تدفعه لأعلى: هوامش أضيق، ثقة أعلى، تباين أعلى، أحداث أندر. جدول الدورة العملى يغطى الحالات الشائعة؛ للنتائج الأقل اعتيادية، ارجع إلى الصيغ مثل n = Z²·p·(1−p)/E² (للنسب) أو n = (Z·σ/E)² (للمتوسطات). خطّط للعينة قبل جمع البيانات — لا بعد."
)
L.render_memory()

L.quiz_q(qid="M04-L07-Q1", qtype="MCQ",
         q_en="Which does NOT increase required sample size?",
         q_ar="أى مما يلى لا يزيد حجم العينة المطلوب؟",
         opts_en=["Tighter margin of error","Higher confidence level",
                  "Larger sample mean","Higher population variance"],
         opts_ar=["هامش خطأ أضيق","مستوى ثقة أعلى","متوسط عينة أكبر","تباين مجتمع أعلى"],
         ans="C", exp_en="Mean's value doesn't affect required size; variance does.",
         exp_ar="قيمة المتوسط لا تؤثر؛ التباين يؤثر.")

L.quiz_q(qid="M04-L07-Q2", qtype="Calc",
         q_en="Need ±5% margin, 95% confidence, large population. Required n is approximately:",
         q_ar="هامش ±5%، ثقة 95%، مجتمع كبير. n المطلوب تقريباً:",
         opts_en=["50","384","1,067","10,000"], opts_ar=["50","384","1,067","10,000"], ans="B",
         exp_en="The classic '384' from the table.", exp_ar="الـ '384' الكلاسيكى من الجدول.")

L.quiz_q(qid="M04-L07-Q3", qtype="MCQ",
         q_en="For a survey of 5,000 customers with ±3% margin and 95% confidence, sample needed is around:",
         q_ar="استبيان لـ 5,000 عميل بهامش ±3% وثقة 95%، العينة المطلوبة حوالى:",
         opts_en=["384","880","1,067","2,000"], opts_ar=["384","880","1,067","2,000"], ans="B",
         exp_en="From the table: 880 for population of 5,000 at ±3%.",
         exp_ar="من الجدول: 880 لمجتمع 5,000 عند ±3%.")

L.quiz_q(qid="M04-L07-Q4", qtype="TrueFalse",
         q_en="True or False: A larger sample always means a more accurate estimate, regardless of how the sample was collected.",
         q_ar="صح أم خطأ: عينة أكبر تعنى تقديراً أدق دائماً، بغض النظر عن طريقة الجمع.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="A biased sample of any size gives biased estimates. Random > big.",
         exp_ar="عينة متحيزة بأى حجم تعطى تقديرات متحيزة. العشوائية > الحجم.")

L.quiz_q(qid="M04-L07-Q5", qtype="MCQ",
         q_en="Detecting a rare event (say 1% rate) accurately requires:",
         q_ar="اكتشاف حدث نادر (نسبة 1%) بدقة يتطلب:",
         opts_en=["Smaller sample","Larger sample","Same sample as 50% rate","No sample"],
         opts_ar=["عينة أصغر","عينة أكبر","نفس عينة الـ 50%","لا عينة"], ans="B",
         exp_en="Rare events need bigger n to see enough cases.",
         exp_ar="الأحداث النادرة تحتاج n أكبر لرؤية حالات كافية.")
L.render_quiz(); L.save()

# ============================================================
# L04-08 — Law of Large Numbers
# ============================================================
L = Lesson(
    lid="M04-L08", module=4,
    title_en="The Law of Large Numbers",
    title_ar="قانون الأعداد الكبيرة",
    objectives_en=[
        "State the Law of Large Numbers (LLN).",
        "Watch a sample mean converge to the true mean as n grows.",
        "Use LLN to justify trust in large-sample averages.",
    ],
    objectives_ar=[
        "صياغة قانون الأعداد الكبيرة (LLN).",
        "مشاهدة كيف يقترب متوسط العينة من المتوسط الحقيقى كلما زاد n.",
        "استخدام LLN لتبرير الثقة فى متوسطات العينات الكبيرة.",
    ],
    sources=["04 Inferential Statistics/Law of Large Numbers.mp4",
             "04 Inferential Statistics/Excel - Law of Large Numbers.mp4",
             "Slides PDF: pages 55-56",
             "Workbook: 03 - CLT & LLN.xlsx"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The Law of Large Numbers says: as the sample size grows, the sample mean gets closer and closer to the true population mean. It's why averaging more observations gives more trustworthy answers.")
    ap(d, "قانون الأعداد الكبيرة يقول: كلما زاد حجم العينة، اقترب متوسط العينة من المتوسط الحقيقى للمجتمع. لهذا متوسط مشاهدات أكثر يعطى إجابات أوثق.")
    callout(d, "key",
            "Roll a fair six-sided die. The expected average is (1+2+3+4+5+6)/6 = 3.5. After 10 rolls you might be at 2.8 or 4.1. After 10,000 rolls, you'll be within 0.01 of 3.5.",
            "ارمى نرداً عادلاً. المتوسط المتوقع (1+2+3+4+5+6)/6 = 3.5. بعد 10 رميات قد تكون عند 2.8 أو 4.1. بعد 10,000 رمية، ستكون فى نطاق 0.01 من 3.5.")
L.section("intro", "1 · The Statement", "1 · النص", s1)

def s2(d):
    eq(d, "lim (n→∞)  x̄ₙ  =  μ",
       note_en="As n grows, sample mean converges to population mean",
       note_ar="كلما زاد n، يقترب متوسط العينة من متوسط المجتمع")
    L.data["formulas"].append({"id":"F-M04-L08-01","name_en":"Law of Large Numbers","name_ar":"قانون الأعداد الكبيرة",
        "expr":"lim (n→∞) x̄ₙ = μ",
        "where_en":"x̄ₙ = mean of n samples, μ = true mean","where_ar":"x̄ₙ متوسط n عينة، μ المتوسط الحقيقى",
        "when_en":"Justifying that bigger samples are more reliable.","when_ar":"تبرير أن عينات أكبر أوثق."})
    ep(d, "Two flavours:")
    ap(d, "نوعان:")
    eb(d, "Weak Law: sample mean converges 'in probability' — for any ε>0, P(|x̄ₙ−μ|>ε) → 0.")
    eb(d, "Strong Law: with probability 1, x̄ₙ → μ as n → ∞.")
    ab(d, "الضعيف: متوسط العينة يقترب 'احتمالياً' — لأى ε>0، P(|x̄ₙ−μ|>ε) ← 0.")
    ab(d, "القوى: باحتمال 1، x̄ₙ ← μ مع n ← ∞.")
L.section("formulas", "2 · The Math (in 1 Line)", "2 · الرياضيات (فى سطر واحد)", s2)

def s3(d):
    callout(d, "example",
            "From the course workbook (03 - CLT & LLN.xlsx → 'Results' sheet): rolling a fair die many times. Average after 10 rolls fluctuates wildly. After 100, calmer. After 1,000, very close to 3.5. After 10,000, virtually pinned at 3.5.",
            "من ملف الدورة (03 - CLT & LLN.xlsx ← ورقة Results): رمى نرد عادل مرات كثيرة. المتوسط بعد 10 يتقلب بشدة. بعد 100 أهدأ. بعد 1,000 قريب جداً من 3.5. بعد 10,000 ثابت تقريباً عند 3.5.")
    h3(d, "Excel mini-workshop", "ورشة Excel صغيرة")
    en_(d, "Cell A2: =RANDBETWEEN(1,6). Drag down 10,000 rows.")
    en_(d, "Cell B2: =AVERAGE($A$2:A2). Drag down. This is the running average.")
    en_(d, "Insert a line chart of column B. Watch it stabilise near 3.5.")
    an_(d, "خلية A2: =RANDBETWEEN(1,6). اسحب لـ 10,000 صف.")
    an_(d, "خلية B2: =AVERAGE($A$2:A2). اسحب. هذا المتوسط المتراكم.")
    an_(d, "أدرج مخططاً خطياً للعمود B. شاهده يستقر قرب 3.5.")
L.section("example", "3 · Worked Example - Die Rolls",
          "3 · مثال محلول - رمى النرد", s3)

def s4(d):
    callout(d, "use",
            "Insurance: an insurer collects premiums from many policyholders. By LLN, the average loss per policyholder approaches the true expected loss as the book grows. That's why insurance works.",
            "التأمين: شركة التأمين تجمع أقساطاً من حاملى الوثائق. بـ LLN، متوسط الخسارة لكل حامل يقترب من الخسارة المتوقعة الحقيقية مع نمو المحفظة. لهذا يعمل التأمين.")
    callout(d, "use",
            "Casinos: the house edge is tiny (1-5%), but with millions of bets per year, LLN guarantees the casino's average gain converges to the true expectation. Individual gamblers may win short-term; long-term, the math wins.",
            "الكازينوهات: حافة البيت صغيرة (1-5%)، لكن مع ملايين الرهانات سنوياً، LLN يضمن اقتراب متوسط ربح الكازينو من التوقع الحقيقى. اللاعب الفردى قد يربح قصير المدى؛ طويلاً، الرياضيات تفوز.")
    callout(d, "use",
            "Manufacturing: average defect rate across a million units approaches the true production-line defect rate. Sample 100 and you might miss it; sample 100,000 and you can't.",
            "التصنيع: معدل العيوب المتوسط عبر مليون وحدة يقترب من المعدل الحقيقى. عينة 100 قد تفوّتها؛ عينة 100,000 لن تفوّتها.")
L.section("practice", "4 · Business Uses", "4 · استخدامات أعمال", s4)

def s5(d):
    callout(d, "trap",
            "Believing LLN means 'a streak of bad luck WILL be balanced by good luck soon'. That is the Gambler's Fallacy. LLN says the AVERAGE converges; it does NOT say individual outcomes balance out.",
            "الاعتقاد بأن LLN يعنى 'سلسلة سيئة ستوازنها سلسلة جيدة قريباً'. هذه مغالطة المقامر. LLN يقول إن المتوسط يقترب؛ لا يقول إن النتائج الفردية تتوازن.")
L.section("traps", "5 · The Gambler's Fallacy",
          "5 · مغالطة المقامر", s5)

L.add_memory_aid("Bigger sample → mean closer to truth. Always.",
                 "عينة أكبر ← متوسط أقرب للحقيقة. دائماً.")
L.add_memory_aid("LLN ≠ 'losing streaks reverse'. Past doesn't owe the future a balance.",
                 "LLN ليس 'الخسائر ستنعكس'. الماضى لا يدين للمستقبل بتوازن.")
L.summary(
    en="The Law of Large Numbers is the foundation that makes inference work. As n grows, the sample mean inevitably approaches the true population mean. It's why insurance, casinos, and quality-control sampling all function. Beware the Gambler's Fallacy — LLN works on averages, never on guaranteeing individual outcomes balance.",
    ar="قانون الأعداد الكبيرة هو الأساس الذى يجعل الاستدلال يعمل. كلما زاد n، يقترب متوسط العينة حتماً من متوسط المجتمع. لذا يعمل التأمين والكازينوهات وعينات ضبط الجودة. احذر مغالطة المقامر — LLN يعمل على المتوسطات، لا يضمن أبداً توازن النتائج الفردية."
)
L.render_memory()

L.quiz_q(qid="M04-L08-Q1", qtype="MCQ",
         q_en="The Law of Large Numbers says, as n → ∞:",
         q_ar="قانون الأعداد الكبيرة يقول إنه مع n ← ∞:",
         opts_en=["Sample variance approaches 0","Sample mean approaches population mean",
                  "Population mean approaches sample mean","Nothing changes"],
         opts_ar=["تباين العينة يقترب من 0","متوسط العينة يقترب من متوسط المجتمع",
                  "متوسط المجتمع يقترب من متوسط العينة","لا شىء يتغير"], ans="B",
         exp_en="x̄ₙ → μ as n → ∞.", exp_ar="x̄ₙ ← μ مع n ← ∞.")

L.quiz_q(qid="M04-L08-Q2", qtype="Calc",
         q_en="Rolling a fair six-sided die, the long-run average should approach:",
         q_ar="رمى نرد عادل، المتوسط طويل المدى يقترب من:",
         opts_en=["3.0","3.5","4.0","6.0"], opts_ar=["3.0","3.5","4.0","6.0"], ans="B",
         exp_en="(1+2+3+4+5+6)/6 = 3.5.", exp_ar="(1+2+3+4+5+6)/6 = 3.5.")

L.quiz_q(qid="M04-L08-Q3", qtype="TrueFalse",
         q_en="True or False: After 100 heads in a row, the next flip is MORE likely to be tails (LLN guarantees a balance).",
         q_ar="صح أم خطأ: بعد 100 صورة متتالية، الرمية التالية أكثر احتمالاً أن تكون شعار (LLN يضمن التوازن).",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Gambler's Fallacy. Each fair-coin flip is independent: 50/50.",
         exp_ar="مغالطة المقامر. كل رمية مستقلة: 50/50.")

L.quiz_q(qid="M04-L08-Q4", qtype="MCQ",
         q_en="Insurance companies rely on LLN because:",
         q_ar="شركات التأمين تعتمد على LLN لأن:",
         opts_en=["Each customer pays the average loss",
                  "Across many customers, average loss approaches true expected loss",
                  "Customers all lose equally","No real reason"],
         opts_ar=["كل عميل يدفع الخسارة المتوسطة",
                  "عبر عملاء كثر، متوسط الخسارة يقترب من المتوقع الحقيقى",
                  "كل العملاء يخسرون بالتساوى","لا سبب حقيقى"], ans="B",
         exp_en="Risk pooling + LLN = predictable aggregate losses.",
         exp_ar="تجميع المخاطر + LLN = خسائر إجمالية متوقعة.")

L.quiz_q(qid="M04-L08-Q5", qtype="MCQ",
         q_en="A coffee shop wants to estimate its true average daily customers. Best approach (LLN-wise):",
         q_ar="مقهى يريد تقدير المتوسط اليومى الحقيقى للعملاء. الأفضل (بمنطق LLN):",
         opts_en=["3 days of data","30 days","300 days","Any sample size"],
         opts_ar=["3 أيام","30 يوماً","300 يوم","أى حجم"], ans="C",
         exp_en="More days → x̄ closer to μ. 300 is much better than 30 or 3.",
         exp_ar="أيام أكثر ← x̄ أقرب لـ μ. 300 أفضل بكثير من 30 أو 3.")
L.render_quiz(); L.save()

# ============================================================
# L04-09 — Central Limit Theorem
# ============================================================
L = Lesson(
    lid="M04-L09", module=4,
    title_en="The Central Limit Theorem",
    title_ar="نظرية النهاية المركزية",
    objectives_en=[
        "State the Central Limit Theorem (CLT).",
        "Distinguish CLT from LLN (they answer different questions).",
        "Use CLT to justify using Normal-based tests on non-Normal data.",
    ],
    objectives_ar=[
        "صياغة نظرية النهاية المركزية (CLT).",
        "التمييز بين CLT و LLN (يجيبان عن أسئلة مختلفة).",
        "استخدام CLT لتبرير اختبارات تعتمد على الطبيعى على بيانات غير طبيعية.",
    ],
    sources=["04 Inferential Statistics/Central Limit Theorem.mp4",
             "04 Inferential Statistics/Excel - Central Limit Theorem.mp4",
             "Slides PDF: pages 57-60",
             "Glossary: Central Limit Theorem",
             "Workbook: 03 - CLT & LLN.xlsx"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The Central Limit Theorem is the deepest, most-magical result in classical statistics. It says: take samples from ANY distribution — skewed, weird, bimodal — compute their means, and those sample means will be Normally distributed (for large enough sample size).")
    ap(d, "نظرية النهاية المركزية هى أعمق وأسحر نتيجة فى الإحصاء الكلاسيكى. تقول: خذ عينات من أى توزيع — منحرف، غريب، ثنائى المنوال — احسب متوسطاتها، وستكون هذه المتوسطات موزعة طبيعياً (لحجم عينة كبير كاف).")
    callout(d, "key",
            "It doesn't matter how UGLY the original distribution is. The distribution of SAMPLE MEANS smooths out into a Normal bell curve. This is what lets us use Normal-distribution tools (z-test, t-test, confidence intervals) on real-world non-Normal data.",
            "لا يهم كم يكون التوزيع الأصلى قبيحاً. توزيع متوسطات العينات يتشكّل كمنحنى جرسى طبيعى. هذا ما يجعلنا نستخدم أدوات الطبيعى (اختبار z، t، فترات الثقة) على بيانات حقيقية غير طبيعية.")
L.section("intro", "1 · The Magic Statement", "1 · النص السحرى", s1)

def s2(d):
    eq(d, "x̄  ~  N(μ, σ²/n)    (for large n)",
       note_en="Sample mean is approximately Normal with mean μ and variance σ²/n",
       note_ar="متوسط العينة طبيعى تقريباً، متوسطه μ وتباينه σ²/n")
    eq(d, "Standard Error of the Mean:  SE  =  σ / √n",
       note_en="The standard deviation of the sample-mean distribution",
       note_ar="الانحراف المعيارى لتوزيع متوسط العينة")
    for f in [
        ("F-M04-L09-01","Central Limit Theorem","نظرية النهاية المركزية",
         "x̄ ~ N(μ, σ²/n) for large n",
         "Sample mean approximately Normal","متوسط العينة طبيعى تقريباً",
         "Inference on means with large samples.","الاستدلال عن المتوسطات بعينات كبيرة."),
        ("F-M04-L09-02","Standard error of mean","الخطأ المعيارى للمتوسط","SE = σ / √n",
         "Spread of the sampling distribution of x̄","تشتت توزيع متوسطات العينات",
         "Building confidence intervals.","بناء فترات الثقة."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
    callout(d, "meaning",
            "Rule of thumb: 'large enough' usually means n ≥ 30. With heavily skewed data, you may need 50–100.",
            "قاعدة عامة: 'كبير بما يكفى' عادة n ≥ 30. مع بيانات منحرفة بشدة، قد تحتاج 50–100.")
L.section("formulas", "2 · The Math", "2 · الرياضيات", s2)

def s3(d):
    ep(d, "From the course slides: distribution of sample means stays Normal even when underlying data is wildly different — Customer NPS responses, US Treasury daily price changes, university exam scores. The histograms of MEANS all look bell-curved.")
    ap(d, "من شرائح الدورة: توزيع متوسطات العينات يبقى طبيعياً حتى لو كانت البيانات الأصلية مختلفة جذرياً — استجابات NPS، تغير أسعار سندات الخزانة اليومى، درجات امتحانات جامعية. كل رسوم تكرار المتوسطات تبدو جرسية.")
    h3(d, "What happens visually", "ماذا يحدث بصرياً")
    en_(d, "Roll a die once (n=1): distribution of single rolls is flat (uniform).")
    en_(d, "Average 2 rolls (n=2): distribution becomes triangular.")
    en_(d, "Average 6 rolls (n=6): distribution rounds toward a bell shape.")
    en_(d, "Average 40 rolls (n=40): almost perfectly Normal.")
    an_(d, "ارمى نرداً مرة واحدة (n=1): توزيع رميات فردية مسطّح (منتظم).")
    an_(d, "متوسط 2 رميات (n=2): يصبح ثلاثياً.")
    an_(d, "متوسط 6 رميات (n=6): يستدير نحو شكل جرسى.")
    an_(d, "متوسط 40 رمية (n=40): طبيعى تقريباً تماماً.")
L.section("example", "3 · CLT in Action - From Uniform to Normal",
          "3 · CLT فى الفعل - من المنتظم إلى الطبيعى", s3)

def s4(d):
    h3(d, "Excel workshop", "ورشة Excel")
    en_(d, "Create a population: 1,000 numbers from a heavily skewed distribution (e.g., =EXP(RAND()*5)).")
    en_(d, "Take 100 samples of size 30 each. Compute the mean of each sample.")
    en_(d, "Plot a histogram of the 100 means. See the Normal bell curve, even though the original data is far from Normal.")
    en_(d, "Reference workbook: 03 - CLT & LLN.xlsx → 'Samples' and 'Results' sheets show this exact experiment.")
    an_(d, "أنشئ مجتمعاً: 1,000 رقم من توزيع منحرف بشدة (مثال: =EXP(RAND()*5)).")
    an_(d, "خذ 100 عينة بحجم 30 لكل واحدة. احسب متوسط كل عينة.")
    an_(d, "ارسم رسماً تكرارياً لمتوسطات الـ 100. شاهد المنحنى الجرسى رغم أن البيانات الأصلية بعيدة عن الطبيعى.")
    an_(d, "الملف المرجعى: 03 - CLT & LLN.xlsx ← ورقتا Samples و Results يعرضان هذه التجربة بالضبط.")
L.section("practice", "4 · Excel Hands-on", "4 · تطبيق Excel", s4)

def s5(d):
    callout(d, "use",
            "A/B tests on revenue per visitor (skewed): with n=30+ visitors per group, the sample means are approximately Normal, so you can use Z- or t-tests without worrying about the skew of individual values.",
            "اختبارات A/B على الإيرادات لكل زائر (منحرفة): مع n=30+ زائر لكل مجموعة، متوسطات العينات تقريباً طبيعية، فيمكن استخدام اختبار z أو t دون قلق على الانحراف.")
    callout(d, "use",
            "Quality control of part dimensions: take samples of 50, compute means. Even if individual parts vary irregularly, the X-bar chart works because of CLT.",
            "ضبط جودة قياسات القطع: خذ عينات 50، احسب المتوسطات. حتى لو تباينت القطع الفردية بشكل غير منتظم، مخطط X-bar يعمل بفضل CLT.")
    callout(d, "use",
            "Election polls: each respondent's answer is binary (0/1). The sample PROPORTION (sum of 0s and 1s / n) is approximately Normal for n ≥ 100 — letting us compute the ±3% margin of error.",
            "استطلاعات الانتخابات: إجابة كل مستجيب ثنائية (0/1). نسبة العينة (مجموع 0 و 1 / n) تقريباً طبيعية لـ n ≥ 100 — مما يتيح حساب هامش الخطأ ±3%.")
L.section("interpret", "5 · Why CLT is the Backbone of Inference",
          "5 · لماذا CLT العمود الفقرى للاستدلال", s5)

def s6(d):
    kvtable(d, header=["Concept / المفهوم","LLN","CLT"],
        rows=[
            ("What converges? / ما الذى يقترب؟",
             "x̄ → μ.\nx̄ ← μ.",
             "Distribution of x̄ → Normal.\nتوزيع x̄ ← طبيعى."),
            ("Tells us about / يخبرنا عن",
             "Where the mean settles.\nأين يستقر المتوسط.",
             "How it varies around there.\nكيف يتذبذب حوله."),
            ("Needs / يتطلب",
             "Independence, finite mean.\nاستقلال، متوسط محدود.",
             "Independence, finite mean + variance, large enough n.\nاستقلال، متوسط وتباين محدودان، n كبير."),
        ])
L.section("definitions", "6 · LLN vs CLT - Two Different Theorems",
          "6 · LLN مقابل CLT - نظريتان مختلفتان", s6)

def s7(d):
    callout(d, "trap",
            "Confusing 'data is Normal' with 'sample means are Normal'. CLT is about the SAMPLE MEANS distribution, not the individual data points.",
            "الخلط بين 'البيانات طبيعية' و'متوسطات العينات طبيعية'. CLT عن توزيع متوسطات العينات، لا النقاط الفردية.")
    callout(d, "trap",
            "Using CLT with n=5 on extremely skewed data. The 'large enough' is data-dependent — Cauchy distributions, for instance, NEVER converge under standard CLT (no finite variance).",
            "استخدام CLT بـ n=5 على بيانات شديدة الانحراف. 'الكبير الكافى' يعتمد على البيانات — توزيع كوشى مثلاً لا يقترب أبداً تحت CLT العادى (لا تباين محدود).")
L.section("traps", "7 · Common Traps", "7 · الفخاخ الشائعة", s7)

L.add_memory_aid("CLT: distribution of sample means becomes Normal as n grows, no matter the original data.",
                 "CLT: توزيع متوسطات العينات يصبح طبيعياً مع زيادة n، مهما كانت البيانات الأصلية.")
L.add_memory_aid("LLN tells you WHERE x̄ goes. CLT tells you HOW SPREAD OUT x̄ is around there.",
                 "LLN يخبرك أين يذهب x̄. CLT يخبرك كم ينتشر x̄ حوله.")
L.add_memory_aid("Rule of thumb: n ≥ 30 is 'large enough' for CLT in most cases.",
                 "قاعدة عامة: n ≥ 30 'كبير كاف' لـ CLT فى معظم الحالات.")
L.summary(
    en="The Central Limit Theorem is what lets statistics work in messy real-world data. No matter how skewed, lumpy, or weird your underlying data, the distribution of sample means is approximately Normal for n ≥ 30. This unlocks the entire toolkit of Normal-based tests (Z-tests, t-tests, confidence intervals). CLT is different from LLN: LLN says the mean converges; CLT describes its distribution.",
    ar="نظرية النهاية المركزية هى ما يجعل الإحصاء يعمل فى البيانات الفوضوية. مهما كانت بياناتك منحرفة أو غريبة، توزيع متوسطات العينات يكون طبيعياً تقريباً لـ n ≥ 30. هذا يفتح كل أدوات الطبيعى (اختبارات z، t، فترات الثقة). CLT يختلف عن LLN: LLN يقول إن المتوسط يقترب؛ CLT يصف توزيعه."
)
L.render_memory()

L.quiz_q(qid="M04-L09-Q1", qtype="MCQ",
         q_en="The Central Limit Theorem states that as n grows, the distribution of:",
         q_ar="نظرية النهاية المركزية تقول إنه مع زيادة n، توزيع:",
         opts_en=["Original data becomes Normal","Sample means becomes approximately Normal",
                  "Variance becomes zero","Population grows"],
         opts_ar=["البيانات الأصلية يصبح طبيعياً","متوسطات العينات يصبح طبيعياً تقريباً",
                  "التباين يصبح صفراً","المجتمع يكبر"], ans="B",
         exp_en="CLT is about the distribution of SAMPLE MEANS.",
         exp_ar="CLT عن توزيع متوسطات العينات.")

L.quiz_q(qid="M04-L09-Q2", qtype="MCQ",
         q_en="The rule-of-thumb sample size for CLT to kick in is approximately:",
         q_ar="حجم العينة كقاعدة عامة لتفعيل CLT تقريباً:",
         opts_en=["n ≥ 5","n ≥ 30","n ≥ 100","n ≥ 1000"],
         opts_ar=["n ≥ 5","n ≥ 30","n ≥ 100","n ≥ 1000"], ans="B",
         exp_en="n ≥ 30 is the common threshold.", exp_ar="n ≥ 30 هو الحد الشائع.")

L.quiz_q(qid="M04-L09-Q3", qtype="Calc",
         q_en="A population has σ = 20. For n = 100, what is the Standard Error of the Mean?",
         q_ar="مجتمع σ = 20. لـ n = 100، ما الخطأ المعيارى للمتوسط؟",
         opts_en=["20","2","0.2","200"], opts_ar=["20","2","0.2","200"], ans="B",
         exp_en="SE = σ/√n = 20/10 = 2.", exp_ar="SE = σ/√n = 20/10 = 2.",
         formula_used="SE = σ / √n")

L.quiz_q(qid="M04-L09-Q4", qtype="TrueFalse",
         q_en="True or False: CLT requires the original population to be Normally distributed.",
         q_ar="صح أم خطأ: CLT يتطلب أن يكون المجتمع طبيعياً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="CLT works regardless of the original distribution (with finite variance).",
         exp_ar="CLT يعمل بغض النظر عن التوزيع الأصلى (بتباين محدود).")

L.quiz_q(qid="M04-L09-Q5", qtype="MCQ",
         q_en="A company tracks daily revenue (highly right-skewed). Wants an A/B test on monthly mean revenue. CLT is helpful because:",
         q_ar="شركة تتتبع إيرادات يومية (منحرفة بشدة). تريد اختبار A/B على متوسط الإيراد الشهرى. CLT مفيد لأن:",
         opts_en=["It makes the daily data Normal","It makes the monthly mean Normal-distributed across many months",
                  "It increases revenue","It removes skew from data"],
         opts_ar=["يجعل البيانات اليومية طبيعية","يجعل المتوسط الشهرى طبيعى التوزيع عبر شهور كثيرة",
                  "يزيد الإيراد","يحذف الانحراف"], ans="B",
         exp_en="The mean of 30 daily values is approximately Normal even when daily data isn't.",
         exp_ar="متوسط 30 قيمة يومية طبيعى تقريباً حتى لو لم تكن البيانات اليومية كذلك.")
L.render_quiz(); L.save()

# ============================================================
# L04-10 — Controlling for Bias
# ============================================================
L = Lesson(
    lid="M04-L10", module=4,
    title_en="Controlling for Bias in Sampling",
    title_ar="التحكم فى التحيز فى أخذ العينات",
    objectives_en=[
        "Recognise common types of sampling bias.",
        "Distinguish representative sampling from random sampling.",
        "Apply stratified, cluster, and systematic sampling where appropriate.",
    ],
    objectives_ar=[
        "التعرف على أنواع التحيز الشائعة.",
        "التمييز بين العينة التمثيلية والعشوائية.",
        "تطبيق المعاينة الطبقية والعنقودية والمنتظمة عند اللزوم.",
    ],
    sources=["04 Inferential Statistics/Controlling for Bias.mp4",
             "Slides PDF: pages 61-63"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "All the math in the world cannot rescue a biased sample. If your data doesn't represent the population you care about, every statistic you compute is misleading — confidently misleading.")
    ap(d, "كل رياضيات العالم لا يمكنها إنقاذ عينة متحيزة. إذا كانت بياناتك لا تمثل المجتمع الذى يهمك، فكل إحصاء تحسبه مضلّل — مضلّل بثقة.")
L.section("intro", "1 · The Hidden Killer", "1 · القاتل الخفى", s1)

def s2(d):
    kvtable(d, header=["Bias type / نوع التحيز","Description / الوصف","Example / مثال"],
        rows=[
            ("Selection bias","Sample not chosen at random.\nالعينة ليست عشوائية.",
             "Surveying only at upscale stores.\nاستطلاع فى المتاجر الراقية فقط."),
            ("Self-selection bias","Respondents pick themselves.\nالمستجيبون يختارون أنفسهم.",
             "Online reviews skew to extremes.\nالمراجعات تميل للتطرف."),
            ("Survivorship bias","Only the survivors are in the data.\nالناجون فقط فى البيانات.",
             "Studying successful companies.\nدراسة الشركات الناجحة فقط."),
            ("Response bias","Phrasing influences the answer.\nالصياغة تؤثر فى الإجابة.",
             "'Did you enjoy our amazing service?'\n'هل استمتعت بخدمتنا الرائعة؟'"),
            ("Non-response bias","Those who don't respond differ.\nمن لا يرد يختلفون.",
             "Busy customers ignore surveys.\nالمشغولون يتجاهلون."),
            ("Confirmation bias","Analyst seeks supportive data.\nالمحلل يبحث عن دليل داعم.",
             "Discarding 'inconvenient' results.\nالتخلص من النتائج 'غير المريحة'."),
        ])
L.section("definitions", "2 · Common Types of Bias",
          "2 · أنواع التحيز الشائعة", s2)

def s3(d):
    h3(d, "Sampling techniques", "تقنيات المعاينة")
    kvtable(d, header=["Method / الطريقة","How it works / كيف تعمل","Best for / الأفضل لـ"],
        rows=[
            ("Simple random","Each unit has equal chance.\nكل وحدة لها نفس الفرصة.",
             "Homogeneous populations.\nمجتمعات متجانسة."),
            ("Stratified","Divide into groups; random within each.\nقسّم لطبقات؛ عشوائى داخل كل طبقة.",
             "When subgroups matter (age, region).\nحين تهم المجموعات الفرعية."),
            ("Cluster","Randomly pick clusters; survey all in each.\nاختر عناقيد عشوائياً؛ استطلع كل من فيها.",
             "Geographically spread populations.\nمجتمعات منتشرة جغرافياً."),
            ("Systematic","Pick every k-th element.\nاختر كل العنصر k.",
             "Ordered lists, quick implementation.\nقوائم مرتبة، تنفيذ سريع."),
        ])
L.section("concept", "3 · Sampling Techniques", "3 · تقنيات المعاينة", s3)

def s4(d):
    callout(d, "example",
            "Tree-planting study: One inexperienced staff (David) is told to choose 'good-looking' trees for Fertiliser B → all easy trees go to B. Experienced staff (Clark) picks tough trees for Fertiliser A. Now A and B are tested on DIFFERENT KINDS OF TREES — pure selection bias.\nFix: assign trees to A or B by random number, not by who looks at them.",
            "دراسة زراعة الأشجار: موظف غير خبير (دافيد) قيل له يختار الأشجار 'الجيدة' للسماد B → كل الأشجار السهلة تذهب لـ B. الموظف الخبير (كلارك) يختار الأشجار الصعبة للسماد A. الآن A و B يُختبران على أنواع أشجار مختلفة — تحيز اختيار صرف.\nالحل: عيّن الأشجار لـ A أو B بعدد عشوائى، لا بحسب من ينظر إليها.")
L.section("example", "4 · Worked Example - The David vs Clark Problem",
          "4 · مثال محلول - مشكلة دافيد وكلارك", s4)

def s5(d):
    callout(d, "use",
            "Customer surveys: stratify by region, age, plan type. Sample 50 from each segment, not 500 random — otherwise the largest segment dominates.",
            "استبيانات العملاء: اقسم بحسب المنطقة والعمر ونوع الباقة. عينة 50 من كل قطاع، لا 500 عشوائياً — وإلا يسيطر القطاع الأكبر.")
    callout(d, "use",
            "A/B testing: hash user IDs to assign variants. Avoid letting people self-select into a 'beta'.",
            "اختبارات A/B: استخدم تجزئة معرفات المستخدم لتعيين النسخ. تجنّب ترك الناس يختارون 'النسخة التجريبية' بأنفسهم.")
    callout(d, "use",
            "Quality control: don't sample only the easy-to-reach parts on a production line. Use systematic (every 100th part) plus random checks of awkward locations.",
            "ضبط الجودة: لا تأخذ عينات من القطع سهلة الوصول فقط على خط الإنتاج. استخدم منتظماً (كل القطعة 100) مع فحوصات عشوائية للأماكن الصعبة.")
L.section("practice", "5 · Business Uses", "5 · استخدامات أعمال", s5)

def s6(d):
    callout(d, "trap",
            "'We have lots of data, so bias doesn't matter' — false. The Literary Digest 1936 poll surveyed 2.4 million, was wildly biased, and got the prediction backwards. Bias scales with sample size.",
            "'لدينا بيانات كثيرة، فالتحيز لا يهم' — خاطئ. استطلاع Literary Digest 1936 استطلع 2.4 مليون، كان متحيزاً بشدة، وأخطأ التوقع تماماً. التحيز يكبر مع حجم العينة.")
L.section("traps", "6 · Common Trap", "6 · فخّ شائع", s6)

L.add_memory_aid("Random > big. A biased huge sample beats no sample, but it also misleads.",
                 "العشوائية > الحجم. عينة كبيرة متحيزة قد تخدع.")
L.add_memory_aid("Use stratification when subgroups (age, region, plan) matter to your question.",
                 "استخدم الطبقات حين تهم المجموعات الفرعية لسؤالك.")
L.summary(
    en="Bias is the silent killer of inference. The Math behind statistics is correct only when sampling is fair. Recognise selection, self-selection, survivorship, response, non-response, and confirmation bias. Use random assignment (or stratified, cluster, or systematic sampling) to make the math work for you, not against you. A biased huge sample produces confidently wrong results — the worst kind of statistics.",
    ar="التحيز هو القاتل الصامت للاستدلال. رياضيات الإحصاء صحيحة فقط عندما تكون المعاينة عادلة. تعرّف على تحيز الاختيار، الذاتى، النجاة، الاستجابة، عدم الاستجابة، والتأكيد. استخدم التعيين العشوائى (أو الطبقات أو العناقيد أو المنتظم) لتعمل الرياضيات لصالحك. عينة كبيرة متحيزة تنتج نتائج خاطئة بثقة — أسوأ أنواع الإحصاء."
)
L.render_memory()

L.quiz_q(qid="M04-L10-Q1", qtype="MCQ",
         q_en="Online product reviews tend to be biased because:",
         q_ar="مراجعات المنتجات عبر الإنترنت متحيزة لأن:",
         opts_en=["Reviewers self-select (often extreme experiences)",
                  "Reviewers are paid","All reviewers are bots","Reviewers are random"],
         opts_ar=["المراجعون يختارون أنفسهم (تجارب متطرفة عادة)",
                  "المراجعون مدفوعون","كلهم بوتات","كلهم عشوائيون"], ans="A",
         exp_en="Self-selection bias: happiest and angriest are loudest.",
         exp_ar="تحيز ذاتى: الأسعد والأغضب هم الأعلى صوتاً.")

L.quiz_q(qid="M04-L10-Q2", qtype="MCQ",
         q_en="To survey customer satisfaction across regions, the best approach is:",
         q_ar="لاستطلاع رضا العملاء عبر مناطق، الأفضل:",
         opts_en=["Random sample from all customers",
                  "Stratified sample by region",
                  "Survey only complainers","Skip statistics"],
         opts_ar=["عينة عشوائية من كل العملاء","عينة طبقية بالمنطقة","استطلاع المشتكين فقط","تخطى الإحصاء"],
         ans="B", exp_en="Stratification ensures every region is represented.",
         exp_ar="الطبقات تضمن تمثيل كل منطقة.")

L.quiz_q(qid="M04-L10-Q3", qtype="MCQ",
         q_en="Studying 'successful' companies to figure out the recipe for success ignores:",
         q_ar="دراسة الشركات 'الناجحة' لمعرفة وصفة النجاح تتجاهل:",
         opts_en=["Tax law","Survivorship bias","CLT","Mean reversion"],
         opts_ar=["قانون الضرائب","تحيز النجاة","CLT","الارتداد للمتوسط"], ans="B",
         exp_en="Failed companies are invisible. We see only winners.",
         exp_ar="الشركات الفاشلة غير مرئية. نرى الفائزين فقط.")

L.quiz_q(qid="M04-L10-Q4", qtype="TrueFalse",
         q_en="True or False: A sample of 1 million emails can never be biased.",
         q_ar="صح أم خطأ: عينة من مليون بريد لا يمكن أن تكون متحيزة.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Bigger does not fix bias. Any size can be biased.",
         exp_ar="الأكبر لا يصلح التحيز. أى حجم قد يكون متحيزاً.")

L.quiz_q(qid="M04-L10-Q5", qtype="MCQ",
         q_en="To run a clean A/B test, users should be assigned to variants by:",
         q_ar="لتشغيل اختبار A/B نظيف، يُعيَّن المستخدمون للنسخ بـ:",
         opts_en=["Their own choice","Account creation date","A random hash of user ID","Whoever volunteers"],
         opts_ar=["اختيارهم","تاريخ إنشاء الحساب","تجزئة عشوائية لمعرف المستخدم","المتطوعون"],
         ans="C", exp_en="Random assignment kills selection bias.",
         exp_ar="التعيين العشوائى يقتل تحيز الاختيار.")
L.render_quiz(); L.save()

print("L04-07, L04-08, L04-09, L04-10 done.")

# Build Module 04 index
module_index = {
    "id":"M04","module":4,"type":"module",
    "title_en":"Inferential Statistics","title_ar":"الإحصاء الاستدلالى",
    "lessons":[
        {"id":"M04-L01","title_en":"Chapter Introduction - Inferential Statistics","title_ar":"مقدمة الفصل"},
        {"id":"M04-L02","title_en":"Populations and Samples","title_ar":"المجتمعات والعينات"},
        {"id":"M04-L03","title_en":"P-values - The Concept","title_ar":"قيم p - المفهوم"},
        {"id":"M04-L04","title_en":"Calculating P-values","title_ar":"حساب قيم p"},
        {"id":"M04-L05","title_en":"Alphas and Confidence Levels","title_ar":"ألفا ومستويات الثقة"},
        {"id":"M04-L06","title_en":"Proving a Difference vs Rejecting No Difference","title_ar":"إثبات الفرق مقابل رفض عدم الفرق"},
        {"id":"M04-L07","title_en":"Determining Sample Size","title_ar":"تحديد حجم العينة"},
        {"id":"M04-L08","title_en":"The Law of Large Numbers","title_ar":"قانون الأعداد الكبيرة"},
        {"id":"M04-L09","title_en":"The Central Limit Theorem","title_ar":"نظرية النهاية المركزية"},
        {"id":"M04-L10","title_en":"Controlling for Bias in Sampling","title_ar":"التحكم فى التحيز"},
    ],
    "sources":["04 Inferential Statistics/*","Slides PDF pages 36-64",
               "Workbook: 03 - CLT & LLN.xlsx"]
}
with open(os.path.join(OUT,"M04 - Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(module_index, f, ensure_ascii=False, indent=2)
print("Module 04 index written.")
