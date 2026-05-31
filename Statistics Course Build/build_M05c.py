import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 05 - Hypothesis Testing"

# ============================================================
# L05-07 — The One-Tailed Test
# ============================================================
L = Lesson(
    lid="M05-L07", module=5,
    title_en="The One-Tailed Test",
    title_ar="اختبار ذيل واحد",
    objectives_en=[
        "Set up a one-tailed hypothesis test correctly.",
        "Identify when a one-tailed test is justified.",
        "Visualise the single rejection region.",
    ],
    objectives_ar=[
        "إعداد اختبار ذيل واحد بشكل صحيح.",
        "تحديد متى يكون اختبار ذيل واحد مبرراً.",
        "تصوّر منطقة الرفض الواحدة.",
    ],
    sources=["05 Hypothesis Testing/One-Tailed Test.mp4",
             "Slides PDF: pages 73-74"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Sometimes we don't care whether the new variant is better OR worse — we only care if it's BETTER. Or only WORSE. That's when one-tailed tests shine.")
    ap(d, "أحياناً لا نهتم هل النسخة الجديدة أفضل أم أسوأ — نهتم فقط بأن تكون أفضل. أو فقط أسوأ. هنا يتألق اختبار الذيل الواحد.")
    callout(d, "key",
            "One-tailed tests concentrate all your alpha in ONE direction. That makes them more sensitive — but only valid if the directional hypothesis was committed BEFORE the data.",
            "اختبار الذيل الواحد يركّز كل ألفا فى اتجاه واحد. هذا يجعله أكثر حساسية — صحيح فقط إذا التزمت بالاتجاه قبل البيانات.")
L.section("intro", "1 · When Direction Matters",
          "1 · حين يهمّ الاتجاه", s1)

def s2(d):
    callout(d, "example",
            "Warranty case re-framed:\n• Old question (two-tail): Are 5-year and 7-year warranties different?\n• New question (one-tail): Does offering a 5-year warranty BEAT no warranty at all?\nFor the new question:\n• Sample: 750 visitors with no warranty (mean revenue) and 750 with 5-year (mean revenue).\n• H0: μ_no = μ_5y\n• Ha: μ_5y > μ_no (specifically better — one-tailed right)",
            "حالة الضمان معاد صياغتها:\n• السؤال القديم (ذيلَين): هل ضمانا 5 و 7 سنوات مختلفان؟\n• السؤال الجديد (ذيل واحد): هل عرض ضمان 5 سنوات يتفوّق على عدم وجود ضمان؟\nللسؤال الجديد:\n• العينة: 750 زائراً بلا ضمان و 750 بضمان 5 سنوات.\n• H0: μ_لا = μ_5سنة\n• Ha: μ_5سنة > μ_لا (أفضل تحديداً — ذيل أيمن)")
L.section("example", "2 · Worked Example - Re-framing the Warranty",
          "2 · مثال محلول - إعادة صياغة الضمان", s2)

def s3(d):
    h3(d, "Right-tailed test", "اختبار ذيل أيمن")
    en_(d, "Ha contains > (e.g., μ > μ₀).")
    en_(d, "All α in the right tail of the test distribution.")
    en_(d, "Reject H0 if test statistic > critical value on the right.")
    an_(d, "Ha تحتوى > (مثل μ > μ₀).")
    an_(d, "كل ألفا فى الذيل الأيمن.")
    an_(d, "ارفض H0 إذا تجاوز إحصاء الاختبار القيمة الحرجة اليمنى.")
    h3(d, "Left-tailed test", "اختبار ذيل أيسر")
    en_(d, "Ha contains < (e.g., μ < μ₀).")
    en_(d, "All α in the left tail.")
    en_(d, "Reject H0 if test statistic < critical value on the left.")
    an_(d, "Ha تحتوى < (مثل μ < μ₀).")
    an_(d, "كل ألفا فى الذيل الأيسر.")
    an_(d, "ارفض H0 إذا قلّ إحصاء الاختبار عن القيمة الحرجة اليسرى.")
L.section("definitions", "3 · Right vs Left",
          "3 · أيمن مقابل أيسر", s3)

def s4(d):
    callout(d, "use",
            "Drug approval: regulators ONLY care if the drug is better than placebo. Right-tailed Ha: μ_drug > μ_placebo.",
            "موافقة دواء: المنظّمون يهتمون فقط بأن يكون الدواء أفضل من العلاج الوهمى. Ha ذيل أيمن: μ_دواء > μ_وهمى.")
    callout(d, "use",
            "Quality control: we want to detect if defect rate INCREASES above an acceptable threshold. Right-tailed Ha: p > p₀.",
            "ضبط جودة: نريد كشف هل معدل العيوب ارتفع فوق حد مقبول. Ha ذيل أيمن: p > p₀.")
    callout(d, "use",
            "Cost-cutting initiative: we want to confirm avg processing time DECREASED. Left-tailed Ha: μ_after < μ_before.",
            "مبادرة خفض تكلفة: نريد التأكد أن متوسط زمن المعالجة انخفض. Ha ذيل أيسر: μ_بعد < μ_قبل.")
L.section("practice", "4 · When to Use One-Tailed",
          "4 · متى تستخدم ذيل واحد", s4)

def s5(d):
    callout(d, "trap",
            "Picking one-tailed because 'I want a smaller p'. This is a logical and ethical error — it doubles the Type I error rate compared to two-tailed.",
            "اختيار ذيل واحد لأن 'أريد قيمة p أصغر'. خطأ منطقى وأخلاقى — يضاعف معدل الخطأ من النوع الأول مقارنة بذيلَين.")
    callout(d, "trap",
            "Forgetting that if the result goes the OPPOSITE direction, a one-tailed test cannot reject H0 — even with overwhelming evidence in the wrong direction.",
            "نسيان أنه إذا ذهبت النتيجة فى الاتجاه المعاكس، فاختبار ذيل واحد لا يمكنه رفض H0 — حتى مع دليل ساحق فى الاتجاه الخاطئ.")
L.section("traps", "5 · The Two Big Traps",
          "5 · الفخّان الكبيران", s5)

L.add_memory_aid("One-tail: more power but only if direction is pre-committed.",
                 "ذيل واحد: قوة أعلى لكن فقط مع التزام مسبق بالاتجاه.")
L.add_memory_aid("Right-tailed: Ha has >. Left-tailed: Ha has <.",
                 "أيمن: Ha بـ >. أيسر: Ha بـ <.")
L.summary(
    en="One-tailed tests focus all of alpha in a single direction — making them more sensitive to detecting a specific kind of change. They are appropriate when the business question itself is directional ('Is the drug better?' / 'Did time decrease?'). They are NEVER appropriate when chosen after seeing data, and they cannot reject H0 if the data go the other direction.",
    ar="اختبار ذيل واحد يركّز كل ألفا فى اتجاه واحد — مما يجعله أكثر حساسية لكشف نوع محدد من التغيير. مناسب حين يكون السؤال التجارى نفسه اتجاهياً ('هل الدواء أفضل؟' / 'هل الزمن انخفض؟'). لا يصلح أبداً إذا اخترته بعد رؤية البيانات، ولا يمكنه رفض H0 إذا ذهبت البيانات فى الاتجاه الآخر."
)
L.render_memory()

L.quiz_q(qid="M05-L07-Q1", qtype="MCQ",
         q_en="A one-tailed test puts all of alpha:",
         q_ar="اختبار ذيل واحد يضع كل ألفا:",
         opts_en=["Equally in both tails","In one specific tail","In the middle","Outside the distribution"],
         opts_ar=["بالتساوى فى الذيلَين","فى ذيل محدد","فى الوسط","خارج التوزيع"], ans="B",
         exp_en="All in one tail = more sensitive there.", exp_ar="كل ألفا فى ذيل واحد = أكثر حساسية هناك.")

L.quiz_q(qid="M05-L07-Q2", qtype="MCQ",
         q_en="Ha: μ > 100 indicates:",
         q_ar="Ha: μ > 100 تشير إلى:",
         opts_en=["Two-tailed","Right-tailed","Left-tailed","Bayesian"],
         opts_ar=["ذيلَين","ذيل أيمن","ذيل أيسر","بيزى"], ans="B",
         exp_en="> = right-tailed.", exp_ar="> = أيمن.")

L.quiz_q(qid="M05-L07-Q3", qtype="TrueFalse",
         q_en="True or False: A one-tailed test can reject H0 even if data goes the opposite direction.",
         q_ar="صح أم خطأ: ذيل واحد يمكنه رفض H0 حتى لو ذهبت البيانات فى الاتجاه المعاكس.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="One-tail can only reject in its committed direction.",
         exp_ar="ذيل واحد يرفض فقط فى اتجاهه الملتزَم.")

L.quiz_q(qid="M05-L07-Q4", qtype="MCQ",
         q_en="A bank wants to confirm new fraud system DETECTS MORE fraud. Right Ha:",
         q_ar="بنك يريد التأكد من نظام احتيال جديد يكشف فعلاً أكثر. Ha الصحيحة:",
         opts_en=["p_new = p_old","p_new ≠ p_old","p_new > p_old","p_new < p_old"],
         opts_ar=["p_جديد = p_قديم","p_جديد ≠ p_قديم","p_جديد > p_قديم","p_جديد < p_قديم"],
         ans="C", exp_en="'More fraud detected' = right-tailed.",
         exp_ar="'احتيال أكثر' = ذيل أيمن.")

L.quiz_q(qid="M05-L07-Q5", qtype="MCQ",
         q_en="The biggest ethical risk of one-tailed tests is:",
         q_ar="أكبر مخاطر أخلاقية لذيل واحد:",
         opts_en=["Excel can't handle them",
                  "Picking the direction after seeing data (p-hacking)",
                  "More math required","Bad font in output"],
         opts_ar=["Excel لا يستطيع","اختيار الاتجاه بعد البيانات (تلاعب)","رياضيات أكثر","خط سيئ"],
         ans="B", exp_en="The classical p-hacking trap.", exp_ar="فخ التلاعب الكلاسيكى.")
L.render_quiz(); L.save()

# ============================================================
# L05-08 — Running a One-Tailed Test
# ============================================================
L = Lesson(
    lid="M05-L08", module=5,
    title_en="Running a One-Tailed Hypothesis Test",
    title_ar="تشغيل اختبار فرضيات ذيل واحد",
    objectives_en=[
        "Execute a one-tailed test end-to-end in Excel.",
        "Read 'p one-tail' from the Excel ToolPak output.",
        "Write a defensible directional conclusion.",
    ],
    objectives_ar=[
        "تنفيذ اختبار ذيل واحد كامل فى Excel.",
        "قراءة 'p ذيل واحد' من مخرج Excel ToolPak.",
        "كتابة استنتاج اتجاهى قابل للدفاع.",
    ],
    sources=["05 Hypothesis Testing/Running a One-Tailed Hypothesis Test.mp4",
             "Workbook: 04 - Hypothesis Testing.xlsx (One-Tail sheet)"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Same warranty data, but now a directional question: does the 5-year warranty BEAT no-warranty? Let's run the full one-tailed test in Excel.")
    ap(d, "نفس بيانات الضمان، لكن سؤال اتجاهى: هل ضمان 5 سنوات يتفوّق على عدم وجود ضمان؟ لنشغّل اختبار ذيل واحد كامل فى Excel.")
L.section("intro", "1 · The Directional Question",
          "1 · السؤال الاتجاهى", s1)

def s2(d):
    h3(d, "Step 1: Hypotheses", "خطوة 1: الفرضيات")
    eq(d, "H0:  μ_no  =  μ_5y      Ha:  μ_5y  >  μ_no")
    h3(d, "Step 2: Alpha", "خطوة 2: ألفا")
    ep(d, "α = 0.05.")
    ap(d, "ألفا = 0.05.")
    h3(d, "Step 3: Test choice", "خطوة 3: اختيار الاختبار")
    ep(d, "Two-sample t-test, ONE-TAILED right (we look only for 'better').")
    ap(d, "اختبار t لعينتَين، ذيل أيمن (نبحث فقط عن 'أفضل').")
    h3(d, "Step 4: Compute in Excel", "خطوة 4: الحساب فى Excel")
    en_(d, "Data → Data Analysis → t-Test: Two-Sample Assuming Unequal Variances.")
    en_(d, "Variable 1 = 5-year revenue, Variable 2 = no-warranty revenue.")
    en_(d, "Read the row labelled 'P(T<=t) one-tail'. That's your p-value.")
    an_(d, "البيانات ← تحليل البيانات ← t-Test: Two-Sample Assuming Unequal Variances.")
    an_(d, "المتغير 1 = إيراد 5 سنوات، المتغير 2 = إيراد لا ضمان.")
    an_(d, "اقرأ صف 'P(T<=t) one-tail'. هذه قيمة p.")
    h3(d, "Step 5: Decide", "خطوة 5: القرار")
    ep(d, "If p (one-tail) < 0.05 AND the difference is in the predicted direction → reject H0. Otherwise fail to reject.")
    ap(d, "إذا p (ذيل واحد) < 0.05 و الفرق فى الاتجاه المتوقع ← ارفض H0. وإلا لا ترفض.")
L.section("example", "2 · The Full Five-Step Walk-Through",
          "2 · المرور الكامل عبر الخمس خطوات", s2)

def s3(d):
    callout(d, "key",
            "Excel ALWAYS gives you both 'p one-tail' and 'p two-tail'. Pick the right one based on YOUR pre-registered Ha. Don't pick the smaller one.",
            "Excel يعطيك دائماً كلا الـ 'p one-tail' و 'p two-tail'. اختر الصحيح بناءً على Ha المسجّلة مسبقاً. لا تختر الأصغر.")
    callout(d, "trap",
            "If the difference goes the WRONG way (5-year revenue < no-warranty revenue), even a tiny p one-tail doesn't let you reject. The data must go in the predicted direction.",
            "إذا ذهب الفرق فى الاتجاه الخاطئ (إيراد 5 سنوات < إيراد لا ضمان)، حتى p one-tail صغيرة لا تتيح الرفض. يجب أن تذهب البيانات فى الاتجاه المتوقع.")
L.section("interpret", "3 · Reading the Excel Output Carefully",
          "3 · قراءة مخرج Excel بحرص", s3)

def s4(d):
    h3(d, "How to write the conclusion", "كيف تكتب الاستنتاج")
    ep(d, "If reject: 'We have evidence (p = X, α = 0.05) that customers offered the 5-year warranty generate higher revenue per visitor than those offered no warranty.'")
    ap(d, "إذا رفضنا: 'لدينا دليل (p = X، ألفا = 0.05) على أن العملاء الذين عُرض عليهم ضمان 5 سنوات يولّدون إيراداً أعلى لكل زائر من الذين لم يُعرض عليهم ضمان.'")
    ep(d, "If fail to reject: 'We did not find evidence (p = X) that the 5-year warranty increases revenue per visitor relative to no warranty.'")
    ap(d, "إذا لم نرفض: 'لم نجد دليلاً (p = X) على أن ضمان 5 سنوات يزيد الإيراد لكل زائر مقارنة بعدم وجود ضمان.'")
L.section("practice", "4 · Writing the Conclusion",
          "4 · كتابة الاستنتاج", s4)

L.add_memory_aid("In Excel ToolPak, look at 'P(T<=t) one-tail' for one-tailed tests.",
                 "فى Excel، انظر إلى 'P(T<=t) one-tail' لاختبارات ذيل واحد.")
L.add_memory_aid("One-tail requires: pre-committed direction + data in that direction.",
                 "ذيل واحد يتطلب: اتجاه مُلتَزم مسبقاً + بيانات فى ذلك الاتجاه.")
L.summary(
    en="Running a one-tailed test is the same five-step procedure, but you read 'p one-tail' from the Excel output. The test is more sensitive in the chosen direction — but you cannot retroactively switch directions, and you cannot reject H0 if the data went the opposite way. Conclusions should be directional and use defensible language.",
    ar="تشغيل اختبار ذيل واحد هو نفس الخمس خطوات، لكن تقرأ 'p one-tail' من مخرج Excel. الاختبار أكثر حساسية فى الاتجاه المختار — لكن لا يمكن التبديل بأثر رجعى، ولا يمكن رفض H0 إذا ذهبت البيانات فى الاتجاه المعاكس. الاستنتاجات يجب أن تكون اتجاهية وبلغة قابلة للدفاع."
)
L.render_memory()

L.quiz_q(qid="M05-L08-Q1", qtype="MCQ",
         q_en="In Excel's t-test ToolPak output, the value to use for a one-tailed test is:",
         q_ar="فى مخرج t-test ToolPak فى Excel، القيمة لاختبار ذيل واحد:",
         opts_en=["t Stat","df","P(T<=t) one-tail","t Critical two-tail"],
         opts_ar=["t Stat","df","P(T<=t) one-tail","t Critical two-tail"], ans="C",
         exp_en="The one-tail p is labelled exactly so.",
         exp_ar="قيمة p للذيل الواحد مسماة هكذا.")

L.quiz_q(qid="M05-L08-Q2", qtype="MCQ",
         q_en="One-tail Ha: μ_new > μ_old. Data shows x̄_new < x̄_old by a lot. Decision:",
         q_ar="Ha ذيل واحد: μ_جديد > μ_قديم. البيانات تظهر x̄_جديد < x̄_قديم بكثير. القرار:",
         opts_en=["Reject H0 (huge effect)","Fail to reject H0 (wrong direction)",
                  "Switch to left-tailed","Re-run as two-tailed"],
         opts_ar=["ارفض H0 (تأثير ضخم)","لا ترفض H0 (اتجاه خاطئ)",
                  "غيّر لذيل أيسر","أعد كذيلَين"], ans="B",
         exp_en="Wrong direction = cannot reject right-tailed H0, no matter how extreme.",
         exp_ar="اتجاه خاطئ = لا يمكن رفض H0 لذيل أيمن، مهما كان متطرفاً.")

L.quiz_q(qid="M05-L08-Q3", qtype="TrueFalse",
         q_en="True or False: For a one-tailed test, p (one-tail) is approximately half of p (two-tail).",
         q_ar="صح أم خطأ: لاختبار ذيل واحد، p one-tail ≈ نصف p two-tail.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="Two-tail is symmetric around 0; one-tail is half (in the right direction).",
         exp_ar="ذيلَين متماثل حول 0؛ ذيل واحد نصف (فى الاتجاه الصحيح).")

L.quiz_q(qid="M05-L08-Q4", qtype="MCQ",
         q_en="A quality engineer wants to test if a new process reduces defect rate below 2%. Test type:",
         q_ar="مهندس جودة يختبر هل عملية جديدة تخفض العيوب تحت 2%. نوع الاختبار:",
         opts_en=["Two-tailed","Right-tailed","Left-tailed","Chi-square"],
         opts_ar=["ذيلَين","أيمن","أيسر","Chi-square"], ans="C",
         exp_en="'Below' = left-tailed.", exp_ar="'تحت' = أيسر.")

L.quiz_q(qid="M05-L08-Q5", qtype="MCQ",
         q_en="After rejecting H0 in a right-tailed test (Ha: μ > 80, p = 0.01), the best wording is:",
         q_ar="بعد رفض H0 فى ذيل أيمن (Ha: μ > 80, p = 0.01)، أفضل صياغة:",
         opts_en=["μ = 80 is proven false","We have evidence μ > 80 (p=0.01)",
                  "All means are > 80","The data are wrong"],
         opts_ar=["μ=80 مُثبَت خطأها","لدينا دليل μ > 80 (p=0.01)","كل المتوسطات > 80","البيانات خاطئة"],
         ans="B", exp_en="Directional, evidence-based language.",
         exp_ar="لغة اتجاهية مدعومة بالأدلة.")
L.render_quiz(); L.save()

# ============================================================
# L05-09 — Errors in Hypothesis Testing
# ============================================================
L = Lesson(
    lid="M05-L09", module=5,
    title_en="Errors in Hypothesis Testing",
    title_ar="الأخطاء فى اختبار الفرضيات",
    objectives_en=[
        "Define Type I and Type II errors.",
        "Understand the trade-off between alpha (α) and beta (β).",
        "Recognise Power = 1 − β and how to increase it.",
    ],
    objectives_ar=[
        "تعريف الخطأ من النوع الأول والثانى.",
        "فهم المقايضة بين ألفا (α) و بيتا (β).",
        "إدراك القوة = 1 − β وكيف تزيدها.",
    ],
    sources=["05 Hypothesis Testing/Errors in Hypothesis Testing.mp4",
             "Slides PDF: pages 75-78",
             "Glossary: False Negative"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Every hypothesis test can be wrong in TWO ways. Knowing the difference — and which error you're optimising against — is the mark of a serious analyst.")
    ap(d, "كل اختبار يمكن أن يخطئ بطريقتَين. معرفة الفرق — وأى خطأ تحسّن — علامة المحلل الجاد.")
L.section("intro", "1 · Two Ways to Be Wrong",
          "1 · طريقتان للخطأ", s1)

def s2(d):
    kvtable(d, header=["Reality / الواقع","Decision: Reject H0 / رفض H0","Decision: Fail to reject H0 / عدم الرفض"],
        rows=[
            ("H0 is true (no effect).\nH0 صحيحة.",
             "Type I error (false positive) — prob = α.\nنوع 1 (إيجابى كاذب) — احتماله α.",
             "Correct (true negative). Prob = 1 − α.\nصحيح (سالب صادق). احتماله 1 − α."),
            ("H0 is false (real effect).\nH0 خاطئة.",
             "Correct (true positive) — Power = 1 − β.\nصحيح (إيجابى صادق) — القوة = 1 − β.",
             "Type II error (false negative) — prob = β.\nنوع 2 (سالب كاذب) — احتماله β."),
        ])
L.section("definitions", "2 · The 2×2 Error Table",
          "2 · جدول الخطأ 2×2", s2)

def s3(d):
    eq(d, "α = P(reject H0 | H0 true)        β = P(fail to reject H0 | H0 false)")
    eq(d, "Power  =  1 − β  =  P(reject H0 | H0 false)",
       note_en="Power = chance of detecting a real effect", note_ar="القوة = فرصة كشف تأثير حقيقى")
    for f in [
        ("F-M05-L09-01","Type I error rate","معدل الخطأ من النوع الأول","α = P(reject H0 | H0 true)",
         "Set by analyst","يحدّده المحلل","Always.","دائماً."),
        ("F-M05-L09-02","Type II error rate","معدل الخطأ من النوع الثانى","β = P(fail to reject H0 | H0 false)",
         "Function of n, effect size, α","دالة فى n، حجم التأثير، ألفا",
         "Power analysis.","تحليل القوة."),
        ("F-M05-L09-03","Power of a test","قوة الاختبار","Power = 1 − β",
         "Often targeted at 0.80","غالباً يُستهدف 0.80",
         "Designing experiments.","تصميم التجارب."),
    ]:
        L.data["formulas"].append({"id":f[0],"name_en":f[1],"name_ar":f[2],"expr":f[3],
            "where_en":f[4],"where_ar":f[5],"when_en":f[6],"when_ar":f[7]})
L.section("formulas", "3 · The Math of α, β, Power",
          "3 · رياضيات ألفا وبيتا والقوة", s3)

def s4(d):
    callout(d, "example",
            "Imaginary case (from the slides):\n• Type I error: an INNOCENT person is punished for a crime they didn't commit.\n• Type II error: a GUILTY person goes free.\nWhich is worse? Society generally chooses to minimise Type I (don't punish innocents) by setting a small α (high burden of proof). Sometimes we accept Type II errors as the cost.",
            "حالة خيالية (من الشرائح):\n• خطأ النوع 1: شخص برىء يُعاقب على جريمة لم يرتكبها.\n• خطأ النوع 2: شخص مذنب يُفلت.\nأيهما أسوأ؟ المجتمع عموماً يفضّل تقليل النوع 1 (لا تعاقب الأبرياء) بضبط ألفا صغير (عبء إثبات عالٍ). أحياناً نقبل أخطاء النوع 2 كتكلفة.")
L.section("example", "4 · Worked Example - Justice System Analogy",
          "4 · مثال محلول - تشبيه القضاء", s4)

def s5(d):
    h3(d, "How to increase Power (= reduce β)", "كيف تزيد القوة (= تقلل β)")
    eb(d, "Increase sample size n (the most reliable lever).")
    eb(d, "Increase α (but only if Type I error cost is acceptable).")
    eb(d, "Decrease variance (cleaner data, better measurement).")
    eb(d, "Test a bigger effect size (if business allows).")
    ab(d, "زِد حجم العينة n (أكثر ذراع موثوقة).")
    ab(d, "زِد ألفا (فقط إن قبلت تكلفة النوع 1).")
    ab(d, "قلّل التباين (بيانات أنظف، قياس أفضل).")
    ab(d, "اختبر حجم تأثير أكبر (إن سمحت الأعمال).")
    callout(d, "meaning",
            "Industry convention: aim for Power ≥ 0.80 (80% chance of catching a real effect). Below 0.50 your test is barely better than a coin flip at detecting real effects.",
            "العرف الصناعى: استهدف القوة ≥ 0.80 (80% فرصة لاصطياد تأثير حقيقى). دون 0.50، اختبارك بالكاد أفضل من قلب عملة فى كشف تأثيرات حقيقية.")
L.section("concept", "5 · Increasing Power",
          "5 · زيادة القوة", s5)

def s6(d):
    callout(d, "use",
            "A/B test design: power analysis is done BEFORE running the test. 'To detect a 2% conversion lift with α=0.05 and Power=0.80, I need n ≈ 25,000 per arm.' Run with less, and you risk a Type II error (missing a real effect).",
            "تصميم A/B: تحليل القوة يُجرى قبل التشغيل. 'لكشف ارتفاع 2% بـ ألفا=0.05 و قوة=0.80، أحتاج n ≈ 25,000 لكل ذراع.' شغّل بأقل، تخاطر بخطأ النوع الثانى.")
    callout(d, "use",
            "Medical screening: a Type II error (missed disease) can kill. Tests are designed with high power AND low alpha — at the cost of large, expensive samples.",
            "فحص طبى: خطأ النوع 2 (مرض مفقود) قاتل. الاختبارات مصممة بقوة عالية وألفا منخفض — بثمن عينات كبيرة مكلفة.")
    callout(d, "use",
            "A/B testing on rare events (conversion < 1%): you need MUCH bigger n to keep Power up. Plan for 10× to 100× more traffic than a 10% conversion test.",
            "اختبار A/B على أحداث نادرة (تحويل < 1%): تحتاج n أكبر بكثير لإبقاء القوة عالية. خطّط لـ 10× إلى 100× حركة مرور أكثر.")
L.section("practice", "6 · Real-World Power Considerations",
          "6 · اعتبارات القوة الواقعية", s6)

def s7(d):
    callout(d, "trap",
            "Underpowered tests (n too small): you'll keep getting 'not significant' results even when a real effect exists. Don't confuse a non-rejection with absence of effect.",
            "اختبارات منخفضة القوة (n صغيرة جداً): ستحصل دائماً على 'لا دلالة' حتى مع تأثير حقيقى. لا تخلط عدم الرفض بغياب التأثير.")
    callout(d, "trap",
            "Confusing α with the probability that H0 is true. α is the probability of WRONGLY REJECTING H0 GIVEN H0 IS TRUE. Different thing.",
            "الخلط بين ألفا واحتمال أن H0 صحيحة. ألفا = احتمال رفض H0 خطأً بشرط أن H0 صحيحة. شىء مختلف.")
L.section("traps", "7 · Common Traps", "7 · الفخاخ الشائعة", s7)

L.add_memory_aid("Type I = false alarm (reject true H0). Type II = miss the effect (fail to reject false H0).",
                 "نوع 1 = إنذار كاذب. نوع 2 = فقدان التأثير.")
L.add_memory_aid("α controls Type I. Sample size controls Type II.",
                 "ألفا يضبط النوع 1. حجم العينة يضبط النوع 2.")
L.add_memory_aid("Power = 1 − β. Industry standard target: 0.80 or higher.",
                 "القوة = 1 − β. هدف صناعى: 0.80 أو أعلى.")
L.summary(
    en="Every hypothesis test risks two kinds of error: rejecting a true H0 (Type I, probability = α) or failing to reject a false H0 (Type II, probability = β). Power = 1 − β is the chance of catching a real effect. You control Type I by setting α; you control Type II mainly by sample size. Plan for both — and especially do a Power analysis BEFORE running an underpowered test.",
    ar="كل اختبار فرضيات يخاطر بخطأَين: رفض H0 صحيحة (نوع 1، احتماله ألفا) أو عدم رفض H0 خاطئة (نوع 2، احتماله β). القوة = 1 − β فرصة اصطياد تأثير حقيقى. تتحكم فى النوع 1 بضبط ألفا؛ وفى النوع 2 أساساً بحجم العينة. خطّط لكليهما — وخصوصاً نفّذ تحليل قوة قبل تشغيل اختبار منخفض القوة."
)
L.render_memory()

L.quiz_q(qid="M05-L09-Q1", qtype="MCQ",
         q_en="A Type I error is:",
         q_ar="الخطأ من النوع الأول:",
         opts_en=["Rejecting H0 when H0 is true","Failing to reject H0 when H0 is false",
                  "Not running a test","Using the wrong formula"],
         opts_ar=["رفض H0 وهى صحيحة","عدم رفض H0 وهى خاطئة","عدم تشغيل اختبار","صيغة خاطئة"],
         ans="A", exp_en="False positive.", exp_ar="إيجابى كاذب.")

L.quiz_q(qid="M05-L09-Q2", qtype="MCQ",
         q_en="A Type II error is also called:",
         q_ar="الخطأ من النوع الثانى يُسمى أيضاً:",
         opts_en=["False positive","False negative","Standard error","Variance"],
         opts_ar=["إيجابى كاذب","سالب كاذب","خطأ معيارى","تباين"], ans="B",
         exp_en="False negative = missed real effect.", exp_ar="سالب كاذب = تأثير حقيقى مفقود.")

L.quiz_q(qid="M05-L09-Q3", qtype="Calc",
         q_en="Power of a test is defined as:",
         q_ar="قوة الاختبار تُعرَّف بـ:",
         opts_en=["α","β","1 − α","1 − β"],
         opts_ar=["α","β","1 − α","1 − β"], ans="D",
         exp_en="Power = chance of detecting a real effect = 1 − β.",
         exp_ar="القوة = فرصة كشف تأثير حقيقى = 1 − β.")

L.quiz_q(qid="M05-L09-Q4", qtype="MCQ",
         q_en="The single best lever to reduce Type II error is:",
         q_ar="أفضل ذراع لتقليل خطأ النوع الثانى:",
         opts_en=["Lower alpha","Increase sample size n",
                  "Change H0","Use a different test name"],
         opts_ar=["خفض ألفا","زيادة n","تغيير H0","اسم اختبار مختلف"], ans="B",
         exp_en="Bigger n = lower β = higher power.",
         exp_ar="n أكبر = β أقل = قوة أعلى.")

L.quiz_q(qid="M05-L09-Q5", qtype="MCQ",
         q_en="In the justice system analogy, a Type I error is:",
         q_ar="فى تشبيه القضاء، خطأ النوع الأول:",
         opts_en=["A guilty person goes free","An innocent person is punished",
                  "A trial is delayed","Defendant doesn't show up"],
         opts_ar=["مذنب يفلت","برىء يُعاقب","تأخير المحاكمة","غياب المتهم"], ans="B",
         exp_en="Reject 'innocent' when actually innocent = punish innocent.",
         exp_ar="رفض 'برىء' عندما هو فعلاً برىء = معاقبة البرىء.")
L.render_quiz(); L.save()

# ============================================================
# L05-10 — Reporting the Results
# ============================================================
L = Lesson(
    lid="M05-L10", module=5,
    title_en="Reporting the Results",
    title_ar="إعداد التقرير",
    objectives_en=[
        "Report a hypothesis test result so a non-statistician can act on it.",
        "Include sample size, effect size, and uncertainty alongside p.",
        "Avoid the common reporting pitfalls (and p-hacking temptations).",
    ],
    objectives_ar=[
        "إعداد تقرير اختبار يفهمه غير المتخصص ويعمل به.",
        "تضمين حجم العينة وحجم التأثير وعدم اليقين مع p.",
        "تجنّب أخطاء التقارير الشائعة (وإغراءات التلاعب).",
    ],
    sources=["05 Hypothesis Testing/Reporting the Results.mp4",
             "Slides PDF: page 79"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "A great test, reported badly, becomes a bad decision. Reporting is half the job — and the half that managers, executives, regulators, and reviewers will actually read.")
    ap(d, "اختبار رائع، يُبلَّغ بشكل سيئ، يصبح قراراً سيئاً. التقرير نصف العمل — والنصف الذى يقرؤه المديرون والتنفيذيون والمنظّمون فعلاً.")
L.section("intro", "1 · Reporting Is Half the Job",
          "1 · التقرير نصف العمل", s1)

def s2(d):
    h3(d, "Essential elements", "العناصر الأساسية")
    en_(d, "The pre-registered hypothesis (H0, Ha) and α.")
    en_(d, "Sample size n per group (and total).")
    en_(d, "Observed sample means / proportions and their std devs.")
    en_(d, "The test used (e.g., two-sample t-test, one-tailed).")
    en_(d, "The test statistic, df, p-value (one-tail or two-tail clearly indicated).")
    en_(d, "Effect size (e.g., difference in means, Cohen's d) — NOT just p.")
    en_(d, "Confidence interval for the effect.")
    en_(d, "Clear decision and business implication.")
    an_(d, "الفرضية المسجّلة (H0، Ha) و ألفا.")
    an_(d, "حجم العينة n لكل مجموعة (والإجمالى).")
    an_(d, "المتوسطات/النسب المرصودة وانحرافاتها.")
    an_(d, "الاختبار المستخدم (مثلاً t لعينتَين، ذيل واحد).")
    an_(d, "إحصاء الاختبار، df، قيمة p (ذيل واحد أم ذيلَين بوضوح).")
    an_(d, "حجم التأثير (فرق المتوسطات، Cohen's d) — لا قيمة p فقط.")
    an_(d, "فاصل ثقة للتأثير.")
    an_(d, "قرار واضح وأثر تجارى.")
L.section("definitions", "2 · The Reporting Checklist",
          "2 · قائمة التقرير", s2)

def s3(d):
    callout(d, "example",
            "From the slides (vacuum-warranty case): the analyst writes:\n• 'We compared two groups of n=750 visitors each (5-year warranty vs 7-year warranty).'\n• 'Mean revenue per visitor differed by 13% (5-year: $24, 7-year: $27).'\n• 'A two-sample two-tailed t-test gave p = 0.08 (α=0.05) — we FAIL TO REJECT H0.'\n• 'There were a small number of high-value buyers in each group, contributing to high variance.'\n• 'Recommendation: do not yet declare the 7-year warranty better; consider a larger trial or focus on conversion rate.'",
            "من الشرائح (حالة الضمان): المحلل يكتب:\n• 'قارنّا مجموعتَين n=750 لكل (ضمان 5 سنوات مقابل 7 سنوات).'\n• 'متوسط الإيراد لكل زائر اختلف بنسبة 13% (5: 24 دولار، 7: 27 دولاراً).'\n• 'اختبار t لعينتَين ذيلَين أعطى p = 0.08 (ألفا=0.05) — لا نرفض H0.'\n• 'هناك عدد قليل من المشترين عالى القيمة فى كل مجموعة، مما أسهم فى تباين عالٍ.'\n• 'التوصية: لا تعلن بعد أن ضمان 7 سنوات أفضل؛ فكّر فى تجربة أكبر أو ركّز على معدل التحويل.'")
L.section("example", "3 · Worked Example - A Clear Report",
          "3 · مثال محلول - تقرير واضح", s3)

def s4(d):
    callout(d, "trap",
            "Reporting ONLY p. Without effect size, n, and CI, p means little. A tiny effect with n=1M can have p<0.001 and be business-irrelevant.",
            "تقديم p فقط. بدون حجم التأثير و n و فاصل الثقة، p تعنى القليل. تأثير ضئيل مع n=1M قد يكون p<0.001 وغير مهم تجارياً.")
    callout(d, "trap",
            "Removing outliers to 'make it significant'. Once you start tinkering, your p-value loses all meaning.",
            "حذف القيم المتطرفة 'لجعلها ذات دلالة'. بمجرد بدء العبث، تفقد p كل معناها.")
    callout(d, "trap",
            "Adding 'just a few more' data points until p < 0.05. This is p-hacking. The test was designed for a fixed n.",
            "إضافة 'بضع نقاط بيانات أكثر' حتى تصبح p < 0.05. هذا تلاعب. الاختبار صُمم لـ n ثابت.")
    callout(d, "trap",
            "Changing variables / test type until something 'works'. Each new test inflates the false-positive rate.",
            "تغيير المتغيرات / نوع الاختبار حتى يعمل شىء. كل اختبار جديد يضخّم معدل الإيجابيات الكاذبة.")
L.section("traps", "4 · The Four Reporting Sins",
          "4 · الذنوب الأربعة فى التقارير", s4)

def s5(d):
    callout(d, "use",
            "Stakeholder-friendly template:\n'On our [date] A/B test, the new variant produced X% lift vs control. With n=Y per arm, the difference was statistically significant (p=Z, α=0.05). Confidence interval for the lift: [low%, high%]. We recommend [rollout / hold / iterate].'",
            "قالب لصانع القرار:\n'فى اختبار A/B بتاريخ [يوم]، أنتجت النسخة الجديدة ارتفاعاً X% مقابل التحكم. مع n=Y لكل ذراع، الفرق ذو دلالة (p=Z، ألفا=0.05). فاصل الثقة: [منخفض%، عالٍ%]. التوصية: [إطلاق / إيقاف / تكرار].'")
    callout(d, "use",
            "Negative result template:\n'Our test did not detect a difference (p=W, α=0.05, n=X per arm, observed lift = Y%). This could mean (1) the effect is zero, or (2) the effect exists but is too small to detect at this sample size. Power analysis indicated we'd need n=Z to detect a Y% lift with 80% power.'",
            "قالب نتيجة سالبة:\n'اختبارنا لم يكشف فرقاً (p=W، ألفا=0.05، n=X لكل ذراع، الارتفاع المرصود = Y%). قد يعنى ذلك (1) التأثير صفر، أو (2) التأثير موجود لكنه أصغر من أن يُكشف بهذا الحجم. تحليل القوة يشير إلى أننا نحتاج n=Z لكشف ارتفاع Y% بقوة 80%.'")
L.section("practice", "5 · Two Reporting Templates",
          "5 · قالبا تقرير", s5)

L.add_memory_aid("Report: H0/Ha, n, means, test, statistic, p, effect size, CI, decision.",
                 "اذكر فى التقرير: H0/Ha، n، المتوسطات، الاختبار، الإحصاء، p، حجم التأثير، CI، القرار.")
L.add_memory_aid("Never: remove outliers, add data, or change tests to chase p<0.05.",
                 "أبداً: لا تحذف متطرفات، لا تضف بيانات، لا تغيّر اختبارات لمطاردة p<0.05.")
L.add_memory_aid("Negative results matter. Report them honestly with sample size and observed effect.",
                 "النتائج السالبة مهمة. أبلغها بصدق مع حجم العينة والتأثير المرصود.")
L.summary(
    en="A complete hypothesis-test report contains hypotheses, sample size, observed statistics, test type, test statistic, p-value, EFFECT SIZE, confidence interval, and a clear business decision. Reporting only p — or worse, manipulating data until p < 0.05 — is the heart of the replication crisis. Honest reporting of negative results is just as valuable as positive findings.",
    ar="تقرير اختبار كامل يحوى الفرضيات، حجم العينة، الإحصاءات المرصودة، نوع الاختبار، الإحصاء، قيمة p، حجم التأثير، فاصل الثقة، وقراراً تجارياً واضحاً. تقديم p فقط — أو الأسوأ، التلاعب بالبيانات حتى p < 0.05 — هو لب أزمة عدم تكرار النتائج. التقرير الأمين للنتائج السالبة بنفس قيمة الإيجابية."
)
L.render_memory()

L.quiz_q(qid="M05-L10-Q1", qtype="MCQ",
         q_en="Which is NOT required in a good hypothesis-test report?",
         q_ar="أى مما يلى ليس مطلوباً فى تقرير جيد؟",
         opts_en=["Sample size n","p-value","Effect size and CI","The analyst's personal opinion"],
         opts_ar=["حجم العينة","قيمة p","حجم التأثير و CI","رأى المحلل الشخصى"], ans="D",
         exp_en="Personal opinion is not a statistical artefact. Stick to the numbers and the decision.",
         exp_ar="الرأى الشخصى ليس أداة إحصائية. التزم بالأرقام والقرار.")

L.quiz_q(qid="M05-L10-Q2", qtype="MCQ",
         q_en="A report says only 'p = 0.04'. The biggest missing piece is:",
         q_ar="تقرير يقول فقط 'p = 0.04'. أكبر شىء مفقود:",
         opts_en=["More p-values","Effect size and sample size for context",
                  "More variables","Color graphs"],
         opts_ar=["قيم p أكثر","حجم التأثير و n للسياق","متغيرات أكثر","رسوم ملوّنة"],
         ans="B", exp_en="p alone is meaningless without context.",
         exp_ar="p وحدها لا معنى لها دون سياق.")

L.quiz_q(qid="M05-L10-Q3", qtype="TrueFalse",
         q_en="True or False: It's acceptable to drop outliers until the p-value crosses 0.05.",
         q_ar="صح أم خطأ: مقبول حذف المتطرفات حتى تعبر p حد 0.05.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="That's p-hacking, the cardinal sin.", exp_ar="هذا تلاعب، الذنب الأكبر.")

L.quiz_q(qid="M05-L10-Q4", qtype="MCQ",
         q_en="A negative result (failed to reject H0) should be:",
         q_ar="نتيجة سالبة (لم نرفض H0) يجب:",
         opts_en=["Hidden","Reported with sample size and observed effect",
                  "Re-framed as positive","Discarded"],
         opts_ar=["إخفاؤها","تقديمها مع حجم العينة والتأثير المرصود",
                  "إعادة صياغتها كإيجابية","رميها"], ans="B",
         exp_en="Honest negative reports help the field and your team.",
         exp_ar="التقارير السالبة الأمينة تخدم المجال والفريق.")

L.quiz_q(qid="M05-L10-Q5", qtype="MCQ",
         q_en="A massive sample finds p < 0.001 but the effect is only +0.05% conversion. The best report:",
         q_ar="عينة ضخمة تجد p < 0.001 لكن التأثير +0.05% تحويل فقط. أفضل تقرير:",
         opts_en=["Declare big win","Report statistically significant but discuss practical relevance",
                  "Hide the result","Re-run the test"],
         opts_ar=["إعلان فوز كبير","ذو دلالة إحصائية لكن ناقش الأهمية العملية",
                  "إخفاء","إعادة الاختبار"], ans="B",
         exp_en="Statistical significance ≠ business value. Both deserve mention.",
         exp_ar="الدلالة الإحصائية ≠ القيمة التجارية. كلاهما يستحق الذكر.")
L.render_quiz(); L.save()

print("L05-07, L05-08, L05-09, L05-10 done.")

# Build Module 05 index
module_index = {
    "id":"M05","module":5,"type":"module",
    "title_en":"Hypothesis Testing","title_ar":"اختبار الفرضيات",
    "lessons":[
        {"id":"M05-L01","title_en":"Chapter Introduction - Hypothesis Testing","title_ar":"مقدمة الفصل"},
        {"id":"M05-L02","title_en":"The 5 Steps of Hypothesis Testing","title_ar":"الخمس خطوات"},
        {"id":"M05-L03","title_en":"Stating the Hypothesis","title_ar":"صياغة الفرضيات"},
        {"id":"M05-L04","title_en":"Setting the Alpha and Confidence Level","title_ar":"ضبط ألفا والثقة"},
        {"id":"M05-L05","title_en":"Selecting the Hypothesis Test","title_ar":"اختيار الاختبار"},
        {"id":"M05-L06","title_en":"Running a Two-Tailed Hypothesis Test","title_ar":"تشغيل اختبار ذيلَين"},
        {"id":"M05-L07","title_en":"The One-Tailed Test","title_ar":"اختبار ذيل واحد"},
        {"id":"M05-L08","title_en":"Running a One-Tailed Hypothesis Test","title_ar":"تشغيل اختبار ذيل واحد"},
        {"id":"M05-L09","title_en":"Errors in Hypothesis Testing","title_ar":"الأخطاء فى اختبار الفرضيات"},
        {"id":"M05-L10","title_en":"Reporting the Results","title_ar":"إعداد التقرير"},
    ],
    "sources":["05 Hypothesis Testing/*","Slides PDF pages 65-80",
               "Workbook: 04 - Hypothesis Testing.xlsx"]
}
with open(os.path.join(OUT,"M05 - Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(module_index, f, ensure_ascii=False, indent=2)
print("Module 05 index written.")
