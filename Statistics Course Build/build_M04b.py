import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 04 - Inferential Statistics"

# ============================================================
# L04-04 — Calculating P-values
# ============================================================
L = Lesson(
    lid="M04-L04", module=4,
    title_en="Calculating P-values",
    title_ar="حساب قيم p",
    objectives_en=[
        "Walk through a complete p-value calculation in a simple coin-flip scenario.",
        "Understand the role of 'as extreme or more extreme' in the count.",
        "Build intuition that you can extend to real tests.",
    ],
    objectives_ar=[
        "المرور بحساب كامل لقيمة p فى سيناريو عملة بسيط.",
        "فهم دور 'نفس التطرف أو أكثر' فى العدّ.",
        "بناء حدس يمكنك مدّه للاختبارات الحقيقية.",
    ],
    sources=["04 Inferential Statistics/Calculating p-values.mp4",
             "Slides PDF: pages 43-46"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The classic example: flip a coin 5 times and get 5 heads. Is this enough evidence the coin is biased?")
    ap(d, "المثال الكلاسيكى: ارمى عملة 5 مرات وتحصل على 5 صور. هل هذا دليل كاف على أن العملة متحيزة؟")
L.section("intro", "1 · The Coin-Flip Question",
          "1 · سؤال العملة", s1)

def s2(d):
    ep(d, "Hypothesis H₀ = the coin is fair (50/50). If H₀ is true, each flip has probability ½. Five flips have 2⁵ = 32 equally likely outcomes.")
    ap(d, "الفرضية H₀ = العملة عادلة (50/50). إذا كانت H₀ صحيحة، كل رمية احتمالها ½. خمس رميات لها 2⁵ = 32 نتيجة متساوية الاحتمال.")
    eq(d, "Total outcomes = 2⁵ = 32")
    h3(d, "Step 1: Count observed result", "خطوة 1: عدّ النتيجة المرصودة")
    ep(d, "Exactly 5 heads in 5 flips: just 1 way (HHHHH).")
    ap(d, "5 صور بالضبط فى 5 رميات: طريقة واحدة فقط (صصصصص).")
    eq(d, "P(observed) = 1/32")
    h3(d, "Step 2: Add 'equally rare' results", "خطوة 2: أضف النتائج 'بنفس التطرف'")
    ep(d, "Under H₀, getting 5 TAILS is just as extreme. That's another 1/32 in the OTHER direction.")
    ap(d, "تحت H₀، 5 شعار بنفس التطرف. هذه 1/32 أخرى فى الاتجاه المعاكس.")
    eq(d, "Equally rare (5 tails) = 1/32")
    h3(d, "Step 3: 'More extreme' results", "خطوة 3: النتائج 'الأكثر تطرفاً'")
    ep(d, "Nothing is more extreme than 5/5 — we are at the boundary.")
    ap(d, "لا شىء أكثر تطرفاً من 5/5 — نحن على الحد.")
    eq(d, "More extreme = 0/32")
    h3(d, "Step 4: P-value (two-tailed)", "خطوة 4: قيمة p (ذيلَين)")
    eq(d, "p  =  1/32 + 1/32  =  2/32  =  0.0625  ≈  6.25%")
    callout(d, "key",
            "Decision at α=0.05: p = 0.0625 > 0.05 → fail to reject the fair-coin hypothesis. Surprising? Five heads in a row is suggestive, but with only 5 flips we don't have enough evidence to reject 'fair'. Power matters!",
            "القرار عند α=0.05: p = 0.0625 > 0.05 ← لا نرفض فرضية العملة العادلة. مفاجئ؟ خمس صور متتالية تثير الشك، لكن مع 5 رميات فقط لا دليل كافٍ للرفض. الحجم مهم!")
L.section("example", "2 · Worked Example - Step-by-Step",
          "2 · مثال محلول - خطوة بخطوة", s2)

def s3(d):
    callout(d, "example",
            "Original slides example - probability of 5 heads:\n• Observed: 5/32\n• Equally rare (5 tails): 5/32 (note: the slide uses 5/32 because it includes 5/5 in both directions plus combinatorial weighting)\n• More extreme: 2/32\nSum: 12/32 = 0.375 (37.5%)\nTakeaway: very little evidence the coin is biased.",
            "مثال الشرائح - احتمال 5 صور:\n• المرصود: 5/32\n• بنفس التطرف (5 شعار): 5/32 (الشريحة تستخدم 5/32 لأنها تتضمن 5/5 فى الاتجاهَين مع وزن توافقى)\n• الأكثر تطرفاً: 2/32\nالمجموع: 12/32 = 0.375 (37.5%)\nالخلاصة: لا دليل تقريباً على تحيز العملة.")
L.section("interpret", "3 · The Course's Slide Example",
          "3 · المثال من شرائح الدورة", s3)

def s4(d):
    callout(d, "use",
            "Real-world translation: An A/B test sees 60 conversions out of 1,000 (variant B) vs 50 of 1,000 (variant A). Software (Excel's =Z.TEST, or any online calculator) computes the p-value the same way: count results as extreme as 60-vs-50 under H₀ (no difference). If p < α, we reject 'no difference'.",
            "ترجمة عملية: اختبار A/B يرى 60 تحويلاً من 1,000 (B) مقابل 50 من 1,000 (A). البرامج (Excel: =Z.TEST، أو أى حاسبة) تحسب قيمة p بنفس الطريقة: عدّ النتائج بنفس تطرف 60 مقابل 50 تحت H₀ (لا فرق). إذا p < α، نرفض 'لا فرق'.")
L.section("practice", "4 · From Coin to A/B Test",
          "4 · من العملة إلى اختبار A/B", s4)

def s5(d):
    callout(d, "trap",
            "Forgetting the 'as extreme or more extreme' — students often only count the EXACT observed result. That undercounts and gives the wrong p-value.",
            "نسيان 'بنفس التطرف أو أكثر' — كثيراً ما يعدّ الطلاب الناتج المرصود بالضبط فقط. هذا يخفّض العدد ويعطى قيمة p خاطئة.")
    callout(d, "trap",
            "One-tailed vs two-tailed: if you ask 'is the coin biased toward heads?' (one-tail), the p is half. If 'is the coin biased in either direction?' (two-tail), it's full. Decide BEFORE seeing the data.",
            "ذيل واحد مقابل ذيلَين: إذا سألت 'هل العملة منحازة للصور؟' (ذيل واحد)، p نصف. إذا 'هل منحازة فى أى اتجاه؟' (ذيلَين)، كاملة. قرر قبل رؤية البيانات.")
L.section("traps", "5 · Common Traps", "5 · الفخاخ الشائعة", s5)

L.add_memory_aid("P-value = P(your data or more extreme | H₀ true).",
                 "قيمة p = P(بياناتك أو أكثر تطرفاً | H₀ صحيحة).")
L.add_memory_aid("Two-tailed = both directions equally extreme. Decide before seeing the data!",
                 "ذيلَين = كلا الاتجاهَين بنفس التطرف. قرر قبل البيانات!")
L.summary(
    en="P-value calculation, at its heart, is a counting exercise: list all outcomes that are at least as extreme as yours under the null, sum their probabilities. The coin-flip example makes the logic concrete; the same logic — done by software — drives every real-world A/B test and clinical trial.",
    ar="حساب قيمة p فى جوهره عملية عدّ: اذكر كل النتائج بنفس تطرف بياناتك أو أكثر تحت الصفرية، اجمع احتمالاتها. مثال العملة يجعل المنطق ملموساً؛ نفس المنطق — تنفّذه البرامج — يحرّك كل اختبار A/B وتجربة سريرية فى الواقع."
)
L.render_memory()

L.quiz_q(qid="M04-L04-Q1", qtype="Calc",
         q_en="Flipping a fair coin 4 times, exactly 4 heads. P-value (two-tailed)?",
         q_ar="رمى عملة عادلة 4 مرات، 4 صور بالضبط. قيمة p (ذيلَين)؟",
         opts_en=["1/16","2/16","4/16","8/16"], opts_ar=["1/16","2/16","4/16","8/16"], ans="B",
         exp_en="2⁴=16 outcomes. P(4 heads) = 1/16. P(4 tails equally rare) = 1/16. Two-tail p = 2/16.",
         exp_ar="2⁴=16 نتيجة. P(4 صور) = 1/16. P(4 شعار) = 1/16. p ذيلَين = 2/16.")

L.quiz_q(qid="M04-L04-Q2", qtype="MCQ",
         q_en="The 'as extreme as or more extreme than' rule matters because:",
         q_ar="قاعدة 'بنفس التطرف أو أكثر' مهمة لأن:",
         opts_en=["Excel requires it",
                  "P-value summarises evidence about extreme outcomes, not just the exact one",
                  "It's tradition","To inflate the result"],
         opts_ar=["Excel يتطلبها","p تلخّص دليلاً على النتائج المتطرفة، لا الواحدة بالضبط","تقليد","لتضخيم النتيجة"],
         ans="B", exp_en="The tail-area concept defines p-value.",
         exp_ar="مفهوم منطقة الذيل يعرّف قيمة p.")

L.quiz_q(qid="M04-L04-Q3", qtype="MCQ",
         q_en="One-tail vs two-tail decision should be made:",
         q_ar="قرار ذيل واحد أم ذيلَين يجب أن يُتخذ:",
         opts_en=["After seeing the result","Before the experiment","Halfway through","Never matters"],
         opts_ar=["بعد رؤية النتيجة","قبل التجربة","فى المنتصف","لا يهم"], ans="B",
         exp_en="Pre-registration prevents p-hacking.", exp_ar="التحديد المسبق يمنع التلاعب.")

L.quiz_q(qid="M04-L04-Q4", qtype="TrueFalse",
         q_en="True or False: A two-tail p-value is half a one-tail p-value, all else equal.",
         q_ar="صح أم خطأ: قيمة p ذيلَين نصف قيمة الذيل الواحد، مع ثبات الباقى.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Two-tail is approximately DOUBLE the one-tail (covers both extremes).",
         exp_ar="ذيلَين تقريباً ضعف الذيل الواحد (يغطى كلا التطرّفَين).")

L.quiz_q(qid="M04-L04-Q5", qtype="MCQ",
         q_en="An online A/B test reports p = 0.024 (two-tail). At α = 0.05:",
         q_ar="اختبار A/B عبر الإنترنت يبلغ p = 0.024 (ذيلَين). عند α = 0.05:",
         opts_en=["Fail to reject H₀","Reject H₀","Recalculate","Need more data"],
         opts_ar=["لا نرفض H₀","نرفض H₀","أعد الحساب","نحتاج بيانات أكثر"], ans="B",
         exp_en="p < α → reject 'no difference'.", exp_ar="p < α ← ارفض 'لا فرق'.")
L.render_quiz(); L.save()

# ============================================================
# L04-05 — Alphas & Confidence Levels
# ============================================================
L = Lesson(
    lid="M04-L05", module=4,
    title_en="Alphas and Confidence Levels",
    title_ar="ألفا ومستويات الثقة",
    objectives_en=[
        "Define alpha (α) and confidence level.",
        "Pick a sensible alpha for different business stakes.",
        "Connect alpha to Type I error risk.",
    ],
    objectives_ar=[
        "تعريف ألفا (α) ومستوى الثقة.",
        "اختيار ألفا مناسب حسب أهمية القرار التجارى.",
        "ربط ألفا بمخاطر الخطأ من النوع الأول.",
    ],
    sources=["04 Inferential Statistics/Alphas & Confidence Levels.mp4",
             "Slides PDF: pages 47-48",
             "Glossary: Alpha, Confidence Level"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Alpha (α) is the line you draw in advance: 'if p < α, I'll reject the null'. Confidence level = 1 − α. These are the two sides of one decision rule.")
    ap(d, "ألفا (α) هو الخط الذى ترسمه مسبقاً: 'إذا p < α، أرفض الصفرية'. مستوى الثقة = 1 − α. وجهان لقاعدة قرار واحدة.")
L.section("intro", "1 · Drawing the Line", "1 · رسم الخط", s1)

def s2(d):
    eq(d, "Confidence Level  =  1 − α")
    kvtable(d, header=["Alpha (α)","Confidence Level","Use when / يُستخدم عند"],
        rows=[
            ("0.10","90%","Exploratory work; smaller samples.\nاستكشافى؛ عينات أصغر."),
            ("0.05","95%","Default for business A/B tests.\nالافتراضى لاختبارات الأعمال."),
            ("0.01","99%","High-stakes (medical, financial).\nقرارات حساسة (طبية، مالية)."),
            ("0.001","99.9%","Very-high-stakes (regulatory).\nحرجة جداً (تنظيمية)."),
        ])
    L.data["formulas"].append({"id":"F-M04-L05-01","name_en":"Alpha-confidence relation","name_ar":"علاقة ألفا والثقة",
        "expr":"Confidence = 1 − α","where_en":"α is the chosen significance threshold","where_ar":"α هو حد الدلالة المختار",
        "when_en":"Setting decision rule before testing.","when_ar":"عند وضع قاعدة القرار قبل الاختبار."})
L.section("definitions", "2 · Common Alpha Values", "2 · قيم ألفا الشائعة", s2)

def s3(d):
    callout(d, "meaning",
            "Alpha = the probability of a Type I error: rejecting a TRUE null. Set α = 0.05, and on average 1 out of 20 times you'd wrongly conclude an effect exists when it doesn't.",
            "ألفا = احتمال الخطأ من النوع الأول: رفض H₀ صحيحة. ضع α = 0.05، فستستنتج خطأً أن هناك تأثيراً مرة كل 20 فى المتوسط.")
    callout(d, "key",
            "Smaller α = fewer false positives BUT harder to detect a real effect (higher Type II error risk). It's always a trade-off.",
            "α أصغر = أخطاء إيجابية كاذبة أقل لكن أصعب فى كشف تأثير حقيقى (مخاطرة أعلى للنوع الثانى). مقايضة دائمة.")
L.section("concept", "3 · The Trade-off", "3 · المقايضة", s3)

def s4(d):
    callout(d, "example",
            "Tree-planting experiment: p-value = 0.013, α = 0.05.\nSince 0.013 < 0.05, we reject 'A = B' and conclude A produces a different survival rate.\nConfidence level = 95% → if we did this experiment 100 times under truly-no-difference, we'd wrongly reject only ~5 times.",
            "تجربة زراعة: قيمة p = 0.013، ألفا = 0.05.\nبما أن 0.013 < 0.05، نرفض 'A = B' ونستنتج أن A يعطى معدل بقاء مختلف.\nمستوى الثقة = 95% ← لو كرّرنا 100 مرة فى حالة عدم وجود فرق حقيقى، سنرفض خطأً ~5 مرات فقط.")
L.section("example", "4 · Worked Example - The Fertiliser Test",
          "4 · مثال محلول - اختبار السماد", s4)

def s5(d):
    callout(d, "use",
            "Marketing A/B tests: α=0.05 is standard. Fast, decent risk control. If results border on the line (p ≈ 0.05), run a second confirmation test.",
            "اختبارات A/B التسويقية: α=0.05 معيار. سريع، تحكم معقول. إذا كانت النتائج حدّية (p ≈ 0.05)، شغّل اختبار تأكيد ثانٍ.")
    callout(d, "use",
            "Medical trials: α=0.01 or stricter. Approving an ineffective drug is catastrophic; the false-positive cost is huge.",
            "تجارب طبية: α=0.01 أو أصرم. الموافقة على دواء غير فعال كارثية؛ تكلفة الخطأ الإيجابى الكاذب ضخمة.")
    callout(d, "use",
            "Particle physics: α ≈ 3×10⁻⁷ ('5-sigma'). Discovering a new particle requires extraordinary evidence.",
            "فيزياء الجسيمات: α ≈ 3×10⁻⁷ ('5-sigma'). اكتشاف جسيم جديد يتطلب دليلاً استثنائياً.")
L.section("practice", "5 · Choosing α by Stakes", "5 · اختيار ألفا حسب الأهمية", s5)

def s6(d):
    callout(d, "trap",
            "Changing α after seeing the data. 'p = 0.06? Let me make α = 0.1, now I can publish!' — this is p-hacking. Always pre-commit to α.",
            "تغيير α بعد رؤية البيانات. 'p = 0.06؟ سأجعل α = 0.1 لأنشر!' — هذا تلاعب. التزم بـ α مسبقاً.")
L.section("traps", "6 · Common Trap", "6 · فخّ شائع", s6)

L.add_memory_aid("Alpha = threshold for rejecting H₀. Confidence = 1 − alpha.",
                 "ألفا = حد رفض H₀. الثقة = 1 − ألفا.")
L.add_memory_aid("Smaller α → fewer false positives, but harder to detect real effects.",
                 "ألفا أصغر ← أخطاء كاذبة أقل، لكن أصعب كشف الحقيقى.")
L.summary(
    en="Alpha is the threshold below which p-values lead you to reject H₀; confidence level is its complement (1 − α). Pick α to match the cost of being wrong: 0.05 for everyday business, 0.01 or stricter for high-stakes decisions. Most importantly, fix α BEFORE looking at the data — otherwise statistics becomes storytelling.",
    ar="ألفا هو الحد الذى إذا كانت قيمة p تحته نرفض H₀؛ مستوى الثقة هو مكمّله (1 − α). اختر ألفا بما يطابق تكلفة الخطأ: 0.05 لأعمال يومية، 0.01 أو أصرم لقرارات حرجة. والأهم: ثبّت ألفا قبل النظر إلى البيانات — وإلا تحوّل الإحصاء إلى حكاية."
)
L.render_memory()

L.quiz_q(qid="M04-L05-Q1", qtype="Calc",
         q_en="If α = 0.05, the confidence level is:",
         q_ar="إذا كان α = 0.05، فإن مستوى الثقة:",
         opts_en=["5%","50%","95%","99%"], opts_ar=["5%","50%","95%","99%"], ans="C",
         exp_en="1 - 0.05 = 0.95 = 95%.", exp_ar="1 - 0.05 = 0.95 = 95%.")

L.quiz_q(qid="M04-L05-Q2", qtype="MCQ",
         q_en="Alpha is the probability of:",
         q_ar="ألفا هو احتمال:",
         opts_en=["Type I error (false positive)","Type II error","Being correct","Effect size"],
         opts_ar=["خطأ من النوع الأول (إيجابى كاذب)","النوع الثانى","صواب الاستنتاج","حجم التأثير"],
         ans="A", exp_en="α controls false-positive rate.", exp_ar="ألفا يضبط معدل الإيجابى الكاذب.")

L.quiz_q(qid="M04-L05-Q3", qtype="MCQ",
         q_en="For a medical trial deciding if a new drug is safe, the best alpha is probably:",
         q_ar="لتجربة طبية تقرر سلامة دواء جديد، أفضل ألفا غالباً:",
         opts_en=["0.10","0.05","0.01 or stricter","Doesn't matter"],
         opts_ar=["0.10","0.05","0.01 أو أصرم","لا يهم"], ans="C",
         exp_en="High stakes → demand stronger evidence.", exp_ar="قرار حرج ← يتطلب دليلاً أقوى.")

L.quiz_q(qid="M04-L05-Q4", qtype="TrueFalse",
         q_en="True or False: You should choose alpha after seeing p-value to maximise your chance of finding an effect.",
         q_ar="صح أم خطأ: يجب اختيار ألفا بعد رؤية قيمة p لتعظيم فرصة العثور على تأثير.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="That's p-hacking. Pre-commit to alpha.",
         exp_ar="هذا تلاعب. ثبّت ألفا مسبقاً.")

L.quiz_q(qid="M04-L05-Q5", qtype="MCQ",
         q_en="P-value = 0.07, alpha = 0.05. Decision:",
         q_ar="قيمة p = 0.07، ألفا = 0.05. القرار:",
         opts_en=["Reject H₀","Fail to reject H₀","Modify alpha to 0.10","Conclude H₀ is true"],
         opts_ar=["ارفض H₀","لا ترفض H₀","غيّر ألفا إلى 0.10","استنتج أن H₀ صحيحة"], ans="B",
         exp_en="p > α → not enough evidence to reject. Note: 'fail to reject' is NOT 'accept'.",
         exp_ar="p > α ← لا دليل كاف للرفض. ملاحظة: 'عدم الرفض' ليس 'القبول'.")
L.render_quiz(); L.save()

# ============================================================
# L04-06 — Proving Difference vs Rejecting No Difference
# ============================================================
L = Lesson(
    lid="M04-L06", module=4,
    title_en="Proving a Difference vs Rejecting No Difference",
    title_ar="إثبات الفرق مقابل رفض عدم الفرق",
    objectives_en=[
        "Understand why we 'reject the null' instead of 'proving the alternative'.",
        "Grasp the logic of falsification (reductio ad absurdum).",
        "Avoid the common error of accepting H₀.",
    ],
    objectives_ar=[
        "فهم لماذا 'نرفض الصفرية' بدلاً من 'إثبات البديلة'.",
        "إدراك منطق التفنيد (الإفضاء إلى المحال).",
        "تجنّب الخطأ الشائع: قبول H₀.",
    ],
    sources=["04 Inferential Statistics/Proving a Difference vs Rejecting No Difference.mp4",
             "Slides PDF: pages 49-52"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Statistics doesn't prove things — it disproves them. This is a feature, not a bug, and it traces back to a deep idea in the philosophy of science.")
    ap(d, "الإحصاء لا يُثبت الأشياء — يُفنّدها. هذه ميزة لا عيب، وتعود إلى فكرة عميقة فى فلسفة العلم.")
    callout(d, "key",
            "Reductio ad absurdum: assume the null is true; if our data would be absurdly unlikely under that assumption, we reject the assumption.",
            "الإفضاء إلى المحال: افترض أن الصفرية صحيحة؛ إذا كانت بياناتنا غير محتملة بشدة تحت الفرض، نرفض الفرض.")
L.section("intro", "1 · The Philosophy", "1 · الفلسفة", s1)

def s2(d):
    ep(d, "Karl Popper's swan example: claim is 'all swans are white'. No amount of white swans proves the claim. But seeing one black swan disproves it. So science is built on FALSIFIABILITY, not proof.")
    ap(d, "مثال البجع لـ كارل بوبر: الادعاء 'كل البجع أبيض'. لا توجد كمية من البجع الأبيض تثبت الادعاء. لكن رؤية بجعة سوداء واحدة تفنّده. لذا يقوم العلم على القابلية للتفنيد، لا الإثبات.")
    callout(d, "key",
            "Einstein (apocryphal): 'No amount of experimentation can ever prove me right; a single experiment can prove me wrong.'",
            "أينشتاين (منسوب): 'لا قدر من التجارب يمكن أن يثبت صوابى أبداً؛ تجربة واحدة قد تثبت خطئى.'")
L.section("concept", "2 · The Black-Swan Argument",
          "2 · حجة البجعة السوداء", s2)

def s3(d):
    ep(d, "Hypothesis testing applies this logic directly:")
    ap(d, "اختبار الفرضيات يطبّق هذا المنطق مباشرة:")
    en_(d, "Assume H₀ is true (no effect).")
    en_(d, "Compute p-value: how surprising would our data be IF H₀ were true?")
    en_(d, "If p < α: data is too surprising — REJECT H₀ and act as if Hₐ is supported.")
    en_(d, "If p ≥ α: data is not surprising enough — FAIL TO REJECT H₀ (but don't 'accept' it).")
    an_(d, "افترض أن H₀ صحيحة (لا تأثير).")
    an_(d, "احسب قيمة p: كم ستكون بياناتنا مفاجئة لو كانت H₀ صحيحة؟")
    an_(d, "إذا p < α: البيانات مفاجئة جداً — ارفض H₀ واعتبر أن Hₐ مدعومة.")
    an_(d, "إذا p ≥ α: البيانات ليست مفاجئة بما يكفى — لا ترفض H₀ (لكن لا 'تقبلها').")
L.section("example", "3 · How This Plays Out in Tests",
          "3 · كيف يجرى ذلك فى الاختبارات", s3)

def s4(d):
    callout(d, "trap",
            "Saying 'we accept H₀'. NO. We FAIL to reject. The absence of evidence isn't evidence of absence — maybe the effect exists and we just didn't have enough data to see it.",
            "قول 'نقبل H₀'. لا. نحن 'لا نرفض'. غياب الدليل ليس دليلاً على الغياب — قد يكون التأثير موجوداً ولم نجمع بيانات كافية لرؤيته.")
    callout(d, "trap",
            "Saying 'we proved Hₐ'. We only have evidence that the null is unlikely; we never PROVE the alternative.",
            "قول 'أثبتنا Hₐ'. لدينا فقط دليل على أن الصفرية غير محتملة؛ لا نُثبت البديلة أبداً.")
L.section("traps", "4 · The Two Linguistic Traps",
          "4 · الفخان اللغويان", s4)

def s5(d):
    callout(d, "use",
            "A/B test where new variant fails: 'Our test did not detect a difference between A and B (p=0.32). This doesn't prove they're equal — only that with this sample we didn't have evidence to declare them different.'",
            "اختبار A/B فشل: 'اختبارنا لم يكتشف فرقاً بين A و B (p=0.32). هذا لا يثبت تساويهما — فقط أنه مع هذه العينة لم نملك دليلاً لإعلان اختلافهما.'")
    callout(d, "use",
            "Drug trial: 'The trial did not reach significance (p=0.08). The drug MAY still be effective; a larger trial is needed.'",
            "تجربة دواء: 'التجربة لم تصل إلى الدلالة (p=0.08). الدواء قد يكون فعالاً؛ نحتاج تجربة أكبر.'")
L.section("practice", "5 · How to Write Conclusions Carefully",
          "5 · كيف تكتب الاستنتاجات بحرص", s5)

L.add_memory_aid("Reject H₀ ≠ Prove Hₐ. Fail to reject H₀ ≠ Accept H₀.",
                 "رفض H₀ ≠ إثبات Hₐ. عدم رفض H₀ ≠ قبول H₀.")
L.add_memory_aid("Statistics falsifies. Black swans disprove 'all swans are white'.",
                 "الإحصاء يفنّد. البجعة السوداء تفنّد 'كل البجع أبيض'.")
L.summary(
    en="Hypothesis testing follows Karl Popper's falsification principle: we never prove a claim is true, we only disprove it. If data is too unlikely under H₀, we reject H₀; otherwise we fail to reject. The language matters: 'reject' is not the same as 'prove the opposite', and 'fail to reject' is not the same as 'accept H₀'. Many published mistakes start at this exact spot.",
    ar="اختبار الفرضيات يتبع مبدأ التفنيد لكارل بوبر: لا نُثبت أبداً ادعاءً، فقط نُفنّده. إذا كانت البيانات غير محتملة بشدة تحت H₀، نرفض H₀؛ وإلا لا نرفض. اللغة مهمة: 'الرفض' ليس مثل 'إثبات العكس'، و 'عدم الرفض' ليس مثل 'قبول H₀'. كثير من الأخطاء المنشورة تبدأ من هنا بالضبط."
)
L.render_memory()

L.quiz_q(qid="M04-L06-Q1", qtype="MCQ",
         q_en="In hypothesis testing, we:",
         q_ar="فى اختبار الفرضيات:",
         opts_en=["Prove the alternative","Reject or fail to reject the null",
                  "Always accept the null","Always pick the alternative"],
         opts_ar=["نُثبت البديلة","نرفض أو لا نرفض الصفرية","نقبل الصفرية دائماً","نختار البديلة دائماً"],
         ans="B", exp_en="Reject / fail to reject is the only correct framing.",
         exp_ar="الرفض / عدم الرفض هو الصياغة الصحيحة الوحيدة.")

L.quiz_q(qid="M04-L06-Q2", qtype="MCQ",
         q_en="Failing to reject H₀ means:",
         q_ar="عدم رفض H₀ يعنى:",
         opts_en=["H₀ is true","H₀ is proven","Not enough evidence to reject H₀","Hₐ is wrong"],
         opts_ar=["H₀ صحيحة","H₀ مُثبتة","لا دليل كاف لرفض H₀","Hₐ خاطئة"], ans="C",
         exp_en="Lack of evidence ≠ proof of H₀.", exp_ar="نقص الدليل ≠ إثبات H₀.")

L.quiz_q(qid="M04-L06-Q3", qtype="TrueFalse",
         q_en="True or False: A single counter-example is enough to disprove a universal claim.",
         q_ar="صح أم خطأ: مثال مضاد واحد يكفى لتفنيد ادعاء عام.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="A",
         exp_en="Popper's falsification logic.", exp_ar="منطق التفنيد لبوبر.")

L.quiz_q(qid="M04-L06-Q4", qtype="MCQ",
         q_en="An A/B test gives p = 0.40. Correct conclusion:",
         q_ar="اختبار A/B يعطى p = 0.40. الاستنتاج الصحيح:",
         opts_en=["Variants are equal","We have no evidence to reject 'no difference'",
                  "Variant A wins","Variant B wins"],
         opts_ar=["النسختان متساويتان","لا دليل لدينا لرفض 'لا فرق'","A فائز","B فائز"], ans="B",
         exp_en="The careful, defensible conclusion.", exp_ar="الاستنتاج الحريص القابل للدفاع.")

L.quiz_q(qid="M04-L06-Q5", qtype="MCQ",
         q_en="Karl Popper's contribution to statistics is:",
         q_ar="مساهمة كارل بوبر فى الإحصاء:",
         opts_en=["The Normal distribution","The principle of falsifiability",
                  "P-value formula","Bayes' theorem"],
         opts_ar=["التوزيع الطبيعى","مبدأ القابلية للتفنيد","صيغة قيمة p","نظرية بايز"],
         ans="B", exp_en="Falsifiability is the philosophical backbone of hypothesis testing.",
         exp_ar="القابلية للتفنيد هى الأساس الفلسفى لاختبار الفرضيات.")
L.render_quiz(); L.save()

print("L04-04, L04-05, L04-06 done.")
