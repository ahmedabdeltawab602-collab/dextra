import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 04 - Inferential Statistics"

# ───────── Cover ─────────
d = setup(); fc(d)
titlep(d, module_no=4,
       module_en="Inferential Statistics",
       module_ar="الإحصاء الاستدلالى",
       subtitle_en="P-values · Alphas · CLT · LLN · Sample Size · Bias",
       subtitle_ar="قيم p · ألفا · النظرية المركزية · الأعداد الكبيرة · حجم العينة · التحيز")
end_m(d, os.path.join(OUT, "M04 - Cover.docx"))

# ============================================================
# L04-01 — Chapter Introduction
# ============================================================
L = Lesson(
    lid="M04-L01", module=4,
    title_en="Chapter Introduction - Inferential Statistics",
    title_ar="مقدمة الفصل - الإحصاء الاستدلالى",
    objectives_en=[
        "Define inferential statistics and how it differs from descriptive.",
        "Map out the journey: sample size, p-values, alpha, confidence.",
        "Build intuition for why we can trust a small sample at all.",
    ],
    objectives_ar=[
        "تعريف الإحصاء الاستدلالى والفرق بينه وبين الوصفى.",
        "رسم خريطة الرحلة: حجم العينة، قيم p، ألفا، الثقة.",
        "بناء حدس لماذا يمكن أن نثق فى عينة صغيرة أصلاً.",
    ],
    sources=["04 Inferential Statistics/Chapter Introduction - Inferential Statistics.mp4",
             "Slides PDF: page 36"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Inferential statistics is what makes statistics worth doing. We rarely have data on a whole population — but using a sample and a few clever ideas (the Law of Large Numbers, the Central Limit Theorem) we can make defensible claims about that population.")
    ap(d, "الإحصاء الاستدلالى هو ما يجعل الإحصاء يستحق العمل. نادراً ما نملك بيانات مجتمع كامل — لكن باستخدام عينة وأفكار ذكية (قانون الأعداد الكبيرة، النظرية المركزية) يمكننا تقديم ادعاءات مدعومة عن ذلك المجتمع.")
    callout(d, "key",
            "Descriptive statistics describes WHAT IS in your data. Inferential statistics tells you WHAT IS LIKELY TRUE beyond your data, and HOW SURE you can be.",
            "الإحصاء الوصفى يصف ما هو موجود فى بياناتك. الاستدلالى يخبرك بما يُرجَّح أن يكون صحيحاً خارج بياناتك، وكم أنت متأكد.")
L.section("intro", "1 · Why We Need Inference", "1 · لماذا نحتاج الاستدلال", s1)

def s2(d):
    ep(d, "This module gives you the inferential toolkit in 10 lessons:")
    ap(d, "هذا الموديول يعطيك أدوات الاستدلال فى 10 دروس:")
    kvtable(d, header=["#","Lesson / الدرس","Big idea / الفكرة الكبرى"],
        rows=[
            ("01","Chapter intro","Where we're going.\nأين سنذهب."),
            ("02","Populations & Samples","The two worlds.\nالعالمان."),
            ("03","P-values - concept","Is this difference real?\nهل هذا الفرق حقيقى؟"),
            ("04","Calculating P-values","Coin-flip walkthrough.\nمثال العملة."),
            ("05","Alphas & Confidence","Where do we draw the line?\nأين نرسم الخط؟"),
            ("06","Prove vs Reject","Why falsify, not prove.\nلماذا نرفض لا نُثبت."),
            ("07","Sample size","How many is enough?\nكم نحتاج؟"),
            ("08","Law of Large Numbers","More data → truer average.\nمزيد من البيانات ← متوسط أصدق."),
            ("09","Central Limit Theorem","Why normality emerges.\nلماذا تظهر الطبيعية."),
            ("10","Controlling for Bias","Random > clever.\nالعشوائية > الذكاء."),
        ])
L.section("concept", "2 · Roadmap of This Module", "2 · خريطة الموديول", s2)

def s3(d):
    callout(d, "use",
            "Every A/B test, every survey, every poll, every clinical trial uses these tools. By the end of Module 05 you'll be able to design, run, and defend the conclusions of a real-world hypothesis test.",
            "كل اختبار A/B، كل استبيان، كل استطلاع، كل تجربة سريرية تستخدم هذه الأدوات. بنهاية الموديول الخامس ستتمكن من تصميم وتشغيل والدفاع عن استنتاجات اختبار فرضيات حقيقى.")
L.section("practice", "3 · Where This Lives in the Real World",
          "3 · أين يعيش هذا فى العالم الحقيقى", s3)

L.add_memory_aid("Inference = generalising from a sample to a population, with quantified uncertainty.",
                 "الاستدلال = التعميم من عينة إلى مجتمع، مع قياس عدم اليقين.")
L.summary(
    en="Inferential statistics extends what we see in a sample to what is probably true in the population. This module walks you through the entire toolkit: how big a sample, how to compute a p-value, how to set an alpha, and how to use the two pillars (LLN and CLT) that make it all work.",
    ar="الإحصاء الاستدلالى يمدّ ما نراه فى عينة إلى ما هو الأرجح فى المجتمع. هذا الموديول يأخذك عبر كل الأدوات: كم حجم العينة، كيف تحسب قيمة p، كيف تضع ألفا، وكيف تستخدم العمودَين (LLN و CLT) اللذَين يجعلان هذا ممكناً."
)
L.render_memory()

L.quiz_q(qid="M04-L01-Q1", qtype="MCQ",
         q_en="Inferential statistics is mainly about:",
         q_ar="الإحصاء الاستدلالى يدور أساساً حول:",
         opts_en=["Describing my data","Generalising from a sample to a population",
                  "Building dashboards","Cleaning data"],
         opts_ar=["وصف بياناتى","التعميم من عينة إلى مجتمع","بناء لوحات بيانات","تنظيف البيانات"],
         ans="B", exp_en="Going beyond the sample is inference.",
         exp_ar="تجاوز العينة هو الاستدلال.")

L.quiz_q(qid="M04-L01-Q2", qtype="TrueFalse",
         q_en="True or False: Inferential statistics produces certain conclusions.",
         q_ar="صح أم خطأ: الإحصاء الاستدلالى ينتج استنتاجات يقينية.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Inference always has uncertainty — quantified by p-values and confidence levels.",
         exp_ar="الاستدلال دائماً فيه عدم يقين — مقيّس بقيم p ومستويات الثقة.")

L.quiz_q(qid="M04-L01-Q3", qtype="MCQ",
         q_en="An A/B test for a new website is an application of:",
         q_ar="اختبار A/B لموقع جديد تطبيق لـ:",
         opts_en=["Descriptive","Inferential","Web design","Project management"],
         opts_ar=["وصفى","استدلالى","تصميم ويب","إدارة مشاريع"], ans="B",
         exp_en="Testing whether the new design is really better = inference.",
         exp_ar="اختبار هل التصميم الجديد أفضل = استدلال.")

L.quiz_q(qid="M04-L01-Q4", qtype="MCQ",
         q_en="Two pillars that make inference from samples possible are:",
         q_ar="العمودان اللذان يجعلان الاستدلال من العينات ممكناً:",
         opts_en=["Mean and median","LLN and CLT","p-value and alpha","Excel and Python"],
         opts_ar=["المتوسط والوسيط","LLN و CLT","قيمة p و ألفا","Excel و Python"], ans="B",
         exp_en="Law of Large Numbers + Central Limit Theorem.", exp_ar="قانون الأعداد الكبيرة + النظرية المركزية.")

L.quiz_q(qid="M04-L01-Q5", qtype="MCQ",
         q_en="The biggest practical advantage of inference over a census is:",
         q_ar="أكبر ميزة عملية للاستدلال على الإحصاء الشامل:",
         opts_en=["More accurate","Faster and cheaper using a sample",
                  "Eliminates uncertainty","Avoids using mean"],
         opts_ar=["أكثر دقة","أسرع وأرخص باستخدام عينة","يلغى عدم اليقين","يتجنب المتوسط"],
         ans="B", exp_en="Sampling makes statistics economical and feasible.",
         exp_ar="المعاينة تجعل الإحصاء اقتصادياً ممكناً.")
L.render_quiz(); L.save()

# ============================================================
# L04-02 — Populations & Samples
# ============================================================
L = Lesson(
    lid="M04-L02", module=4,
    title_en="Populations and Samples",
    title_ar="المجتمعات والعينات",
    objectives_en=[
        "Distinguish a population from a sample.",
        "Understand why random sampling matters more than size.",
        "See how sampling variation produces different results from different samples.",
    ],
    objectives_ar=[
        "التمييز بين المجتمع والعينة.",
        "فهم لماذا تكون العشوائية أهم من الحجم.",
        "رؤية كيف يُنتج التباين فى المعاينة نتائج مختلفة من عينات مختلفة.",
    ],
    sources=["04 Inferential Statistics/Population & Samples.mp4",
             "Slides PDF: pages 37-38"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "A population is everyone or everything we want to learn about. A sample is the subset we actually collect data on. In inferential statistics, samples speak FOR the population — but only if they're chosen well.")
    ap(d, "المجتمع هو كل من نريد أن نعرف عنه. العينة هى الجزء الذى نجمع منه البيانات فعلاً. فى الاستدلال، العينات تتحدث نيابة عن المجتمع — لكن فقط إن اخترناها جيداً.")
L.section("intro", "1 · The Two Worlds", "1 · العالمان", s1)

def s2(d):
    kvtable(d, header=["Concept / المفهوم","Notation / الرمز","Example / مثال"],
        rows=[
            ("Population mean","μ (mu)","Average spend across ALL customers."),
            ("Population std dev","σ (sigma)","Std dev across ALL customers."),
            ("Sample mean","x̄ (x-bar)","Average across 500 surveyed customers."),
            ("Sample std dev","s","Std dev across the surveyed 500."),
            ("Sample size","n","500"),
            ("Population size","N","850,000"),
        ])
    callout(d, "key",
            "Greek letters = parameters of the population (often unknown). Latin letters = statistics from the sample (computed from data). Inference is the art of using s, x̄ to learn about σ, μ.",
            "الحروف اليونانية = معاملات المجتمع (غالباً غير معروفة). الحروف اللاتينية = إحصاءات من العينة (محسوبة من البيانات). الاستدلال هو فن استخدام s، x̄ لمعرفة σ، μ.")
L.section("definitions", "2 · The Vocabulary", "2 · المفردات", s2)

def s3(d):
    ep(d, "When samples are random AND large enough, they're REPRESENTATIVE of the population. But due to chance, two random samples from the same population WILL give slightly different means. That's called sampling variation — and quantifying it is what most of this module is about.")
    ap(d, "حين تكون العينات عشوائية وكبيرة بما يكفى، فإنها تمثّل المجتمع. لكن بسبب الصدفة، عينتان عشوائيتان من نفس المجتمع ستعطيان متوسطات مختلفة قليلاً. هذا يُسمى تباين المعاينة — وقياسه هو ما يدور حوله معظم هذا الموديول.")
L.section("concept", "3 · Why Random Matters More Than Size",
          "3 · لماذا العشوائية أهم من الحجم", s3)

def s4(d):
    callout(d, "example",
            "Tree-planting study. The TRUE population survival rate after fertiliser A = 70%. We take 4 random samples of 100 trees each:\n• Sample 1: 71% survived.\n• Sample 2: 68%.\n• Sample 3: 73%.\n• Sample 4: 65%.\nAll close to 70%, none exactly 70%. That's sampling variation.",
            "دراسة زراعة أشجار. معدل البقاء الحقيقى للسماد A = 70%. نأخذ 4 عينات عشوائية من 100 شجرة:\n• العينة 1: 71% بقت.\n• 2: 68%.\n• 3: 73%.\n• 4: 65%.\nكلها قريبة من 70%، لا واحدة بالضبط 70%. هذا هو تباين المعاينة.")
L.section("example", "4 · Worked Example - Sampling Variation",
          "4 · مثال محلول - تباين المعاينة", s4)

def s5(d):
    callout(d, "use",
            "Customer surveys: Polling 1,000 random customers gives a very tight confidence interval (±3%). Polling 10,000 self-selected enthusiasts is biased — even with 10× more data, the conclusion may be wrong.",
            "استبيانات العملاء: استطلاع 1,000 عميل عشوائياً يعطى فاصل ثقة ضيقاً (±3%). استطلاع 10,000 من المتحمسين الذين اختاروا أنفسهم بأنفسهم متحيز — حتى مع 10 أضعاف البيانات، الاستنتاج قد يكون خاطئاً.")
    callout(d, "use",
            "Election polls: A bad sample of 100,000 (e.g., calls only landlines, missing young voters) can be wildly more wrong than a clean random sample of 1,000.",
            "استطلاعات الانتخابات: عينة سيئة من 100,000 (مكالمات هواتف أرضية، تفوّت الشباب) قد تكون أكثر خطأً بكثير من عينة عشوائية نظيفة من 1,000.")
L.section("practice", "5 · Real-World Business Uses", "5 · استخدامات واقعية", s5)

def s6(d):
    callout(d, "trap",
            "Bigger ≠ better. A huge biased sample is worse than a small random one. The classic example: a 1936 Literary Digest poll of 2.4 million predicted Landon would beat Roosevelt — they polled phone owners (rich, Republican). Roosevelt won 61-37%.",
            "الأكبر ليس أفضل. عينة كبيرة متحيزة أسوأ من صغيرة عشوائية. المثال الكلاسيكى: استطلاع 1936 لـ Literary Digest استطلع 2.4 مليون وتوقع فوز Landon على Roosevelt — استطلعوا أصحاب الهواتف (الأغنياء، الجمهوريون). فاز Roosevelt 61-37%.")
L.section("traps", "6 · Common Trap", "6 · فخّ شائع", s6)

L.add_memory_aid("μ, σ = population (unknown). x̄, s = sample (computed).",
                 "μ، σ = المجتمع (مجهول). x̄، s = العينة (محسوبة).")
L.add_memory_aid("Random + large enough = representative. Big and biased = useless.",
                 "عشوائى + كبير كاف = تمثيلى. كبير ومتحيز = عديم النفع.")
L.summary(
    en="Inferential statistics relies on samples to learn about populations. The population's parameters (μ, σ) are usually unknown; sample statistics (x̄, s) are our estimates. Random selection is more important than size. Different random samples give different estimates — that's sampling variation, the central problem that p-values and confidence intervals exist to manage.",
    ar="الإحصاء الاستدلالى يعتمد على العينات لمعرفة المجتمعات. معاملات المجتمع (μ، σ) غالباً مجهولة؛ إحصاءات العينة (x̄، s) هى تقديراتنا. الاختيار العشوائى أهم من الحجم. عينات عشوائية مختلفة تعطى تقديرات مختلفة — هذا تباين المعاينة، وهو المشكلة المركزية التى توجد قيم p وفترات الثقة لإدارتها."
)
L.render_memory()

L.quiz_q(qid="M04-L02-Q1", qtype="MCQ",
         q_en="μ refers to:",
         q_ar="μ تشير إلى:",
         opts_en=["Sample mean","Population mean","Sample size","Population size"],
         opts_ar=["متوسط العينة","متوسط المجتمع","حجم العينة","حجم المجتمع"], ans="B",
         exp_en="Greek = population.", exp_ar="يونانى = مجتمع.")

L.quiz_q(qid="M04-L02-Q2", qtype="MCQ",
         q_en="A sample that is REPRESENTATIVE of the population is:",
         q_ar="عينة ممثّلة للمجتمع هى:",
         opts_en=["Very large","Carefully picked friends","Random and large enough","Always 30 items"],
         opts_ar=["كبيرة جداً","أصدقاء مختارون بعناية","عشوائية وكبيرة كافية","30 دائماً"],
         ans="C", exp_en="Random + enough size = representative.", exp_ar="عشوائى + حجم كاف = تمثيلى.")

L.quiz_q(qid="M04-L02-Q3", qtype="TrueFalse",
         q_en="True or False: A large biased sample is generally better than a small random one.",
         q_ar="صح أم خطأ: عينة كبيرة متحيزة عموماً أفضل من صغيرة عشوائية.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Bias is far more dangerous than small size.", exp_ar="التحيز أخطر بكثير من صغر الحجم.")

L.quiz_q(qid="M04-L02-Q4", qtype="MCQ",
         q_en="Two random samples from the same population give different means because:",
         q_ar="عينتان عشوائيتان من نفس المجتمع تعطيان متوسطات مختلفة لأن:",
         opts_en=["The population changed",
                  "Random chance — sampling variation",
                  "One of them is wrong",
                  "Different units"],
         opts_ar=["المجتمع تغيّر","الصدفة العشوائية — تباين المعاينة","أحدهما خاطئ","وحدات مختلفة"],
         ans="B", exp_en="Different draws → different points → different means.",
         exp_ar="سحبات مختلفة ← نقاط مختلفة ← متوسطات مختلفة.")

L.quiz_q(qid="M04-L02-Q5", qtype="MCQ",
         q_en="A bank wants to estimate average loan default rate. Best approach:",
         q_ar="بنك يريد تقدير معدل التعثر. الأفضل:",
         opts_en=["Sample only VIP loans","Sample only failed loans",
                  "Random sample across all loans","Use only last month's data"],
         opts_ar=["عينة قروض VIP فقط","عينة القروض الفاشلة فقط","عينة عشوائية من كل القروض","شهر الماضى فقط"],
         ans="C", exp_en="Representativeness requires randomness across the full population.",
         exp_ar="التمثيلية تتطلب عشوائية عبر المجتمع كاملاً.")
L.render_quiz(); L.save()

# ============================================================
# L04-03 — P-values (Concept)
# ============================================================
L = Lesson(
    lid="M04-L03", module=4,
    title_en="P-values - The Concept",
    title_ar="قيم p - المفهوم",
    objectives_en=[
        "State exactly what a p-value means (in plain words).",
        "Use the p-value to judge whether an observed effect could be just chance.",
        "Recognise small p-values as evidence against the no-effect hypothesis.",
    ],
    objectives_ar=[
        "صياغة معنى قيمة p بالضبط (بكلمات بسيطة).",
        "استخدام قيمة p للحكم على ما إذا كان التأثير المرصود قد يكون صدفة.",
        "إدراك أن قيمة p الصغيرة دليل ضد فرضية عدم التأثير.",
    ],
    sources=["04 Inferential Statistics/P-values.mp4",
             "Slides PDF: pages 39-42",
             "Glossary: P Value"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "P-value is the single most powerful — and most abused — concept in inferential statistics. Get its meaning right, and the rest of hypothesis testing becomes natural.")
    ap(d, "قيمة p هى أقوى — وأكثر إساءة استخدام — مفهوم فى الإحصاء الاستدلالى. أتقن معناها، وكل اختبار الفرضيات سيصبح طبيعياً.")
    callout(d, "key",
            "P-value = the probability of observing a result AS EXTREME AS YOURS (or more), ASSUMING THERE IS NO REAL EFFECT.",
            "قيمة p = احتمال رؤية نتيجة بنفس تطرّف نتيجتك (أو أكثر تطرّفاً)، بافتراض أنه لا يوجد تأثير حقيقى.")
L.section("intro", "1 · The Definition That Matters",
          "1 · التعريف الذى يهمّ", s1)

def s2(d):
    ep(d, "Imagine: without any new fertiliser, trees survive at 67%. We try Fertiliser A and B; A is at 90%, B at 80%. The difference (10%) looks real. But could random chance alone produce a 10% gap, even if A and B were identical?")
    ap(d, "تخيل: بدون أى سماد جديد، تبقى 67% من الأشجار. نجرب سمادَى A و B؛ A بـ 90%، B بـ 80%. الفرق (10%) يبدو حقيقياً. لكن هل يمكن للصدفة وحدها أن تنتج فجوة 10% حتى لو كان A و B متطابقَين؟")
    callout(d, "meaning",
            "P-value answers exactly this. Small p → 'random chance is a poor explanation, the effect is probably real.' Large p → 'random chance could easily produce this gap; we have no proof of a real effect.'",
            "قيمة p تجيب هذا بالضبط. p صغيرة ← 'الصدفة تفسير ضعيف، التأثير على الأرجح حقيقى.' p كبيرة ← 'الصدفة قد تنتج هذه الفجوة بسهولة؛ لا دليل على تأثير حقيقى.'")
L.section("concept", "2 · The Intuition", "2 · الحدس", s2)

def s3(d):
    kvtable(d, header=["p-value / قيمة p","Interpretation / التفسير","Decision / القرار"],
        rows=[
            ("0.16","16% chance of seeing this gap by chance.\n16% احتمال رؤية الفجوة بالصدفة.",
             "Don't reject the no-effect idea.\nلا نرفض فرضية عدم التأثير."),
            ("0.08","8%.\n8%.",
             "Borderline; usually don't reject at α=0.05.\nحدّى؛ لا نرفض عند α=0.05."),
            ("0.013","1.3%.\n1.3%.",
             "Reject - probably real effect.\nنرفض - تأثير حقيقى على الأرجح."),
            ("4 × 10⁻¹⁰","1 in 2.5 billion.\n1 من 2.5 مليار.",
             "Definitely reject - virtually impossible by chance.\nنرفض بشدة - مستحيل تقريباً بالصدفة."),
        ])
L.section("example", "3 · Reading P-values - Real Numbers",
          "3 · قراءة قيم p - أرقام حقيقية", s3)

def s4(d):
    callout(d, "trap",
            "Common mistake #1: 'P-value is the probability that H₀ is true.' WRONG. P-value assumes H₀ is true, then asks how likely the data would be.",
            "خطأ شائع #1: 'قيمة p هى احتمال أن H₀ صحيحة.' خاطئ. قيمة p تفترض أن H₀ صحيحة، ثم تسأل كم احتمال البيانات.")
    callout(d, "trap",
            "Common mistake #2: 'P=0.05 means there's only a 5% chance of being wrong.' WRONG. P-value is about the data under H₀, not about your ultimate decision being right.",
            "خطأ شائع #2: 'p=0.05 يعنى احتمال 5% فقط للخطأ.' خاطئ. p عن البيانات تحت H₀، لا عن صواب قرارك النهائى.")
    callout(d, "trap",
            "Common mistake #3: 'A small p-value means a big effect.' WRONG. With huge samples, even tiny meaningless differences produce tiny p-values. P-value measures evidence, not size.",
            "خطأ شائع #3: 'p صغيرة تعنى تأثير كبير.' خاطئ. مع عينات ضخمة، حتى فروق صغيرة لا معنى لها تنتج p صغيرة. p تقيس الدليل، لا الحجم.")
L.section("traps", "4 · Three Big Misconceptions", "4 · ثلاثة سوء فهم كبرى", s4)

def s5(d):
    callout(d, "use",
            "A/B test on a checkout button colour. After 20,000 visitors, new design has +0.3% conversion, p=0.001. P-value is tiny so the effect is real, but the effect SIZE is tiny — is 0.3% worth a redesign? Statistics alone can't tell you. Business judgment must follow statistics.",
            "اختبار A/B لزر دفع. بعد 20,000 زائر، التصميم الجديد +0.3% تحويل، p=0.001. p صغيرة جداً فالتأثير حقيقى، لكن الحجم ضئيل — هل 0.3% تستحق إعادة تصميم؟ الإحصاء وحده لا يقول لك. الحكم التجارى يتبع الإحصاء.")
    callout(d, "use",
            "Banking fraud detection: a new model flags 0.1% more fraud than the old one. With millions of transactions, p < 0.0001. The tiny improvement is statistically significant AND business-critical (each fraud = thousands of dollars).",
            "كشف احتيال بنكى: نموذج جديد يكشف 0.1% احتيال إضافى. مع ملايين المعاملات، p < 0.0001. التحسن الصغير ذو دلالة إحصائية ومهم تجارياً (كل احتيال = آلاف الدولارات).")
L.section("practice", "5 · Statistical vs Practical Significance",
          "5 · الدلالة الإحصائية مقابل العملية", s5)

L.add_memory_aid("P-value = 'How likely is data this extreme if there is no effect?'",
                 "قيمة p = 'كم احتمال بيانات بهذا التطرف لو لم يكن هناك تأثير؟'")
L.add_memory_aid("Small p → reject 'no effect'. Large p → not enough evidence to reject.",
                 "p صغيرة ← ارفض 'لا تأثير'. p كبيرة ← لا دليل كافٍ للرفض.")
L.add_memory_aid("Statistical significance ≠ practical significance. Always ask: is this effect worth acting on?",
                 "الدلالة الإحصائية ≠ الدلالة العملية. اسأل دائماً: هل التأثير يستحق العمل عليه؟")
L.summary(
    en="A p-value is the probability of seeing data at least as extreme as yours IF the null hypothesis (no effect) were true. Small p-values are evidence against the null; large p-values are not. P-value tells you about evidence, not effect size, and never about how likely the null is to be true. Used carelessly, it has fueled a replication crisis; used well, it remains a workhorse of modern decision-making.",
    ar="قيمة p هى احتمال رؤية بيانات بنفس تطرّف بياناتك (أو أكثر تطرّفاً) لو كانت الفرضية الصفرية (لا تأثير) صحيحة. قيم p الصغيرة دليل ضدّ الصفرية؛ الكبيرة ليست كذلك. قيمة p تخبرك عن الدليل، لا عن حجم التأثير، ولا عن احتمال صحة الفرضية الصفرية. استخدامها بإهمال تسبّب فى أزمة عدم تكرار النتائج؛ استخدامها جيداً يبقيها حصان عمل لاتخاذ القرار."
)
L.render_memory()

L.quiz_q(qid="M04-L03-Q1", qtype="MCQ",
         q_en="P-value is best defined as:",
         q_ar="قيمة p تُعرَّف أفضل بأنها:",
         opts_en=["The probability H₀ is true",
                  "The probability of seeing data as extreme as observed, assuming H₀ is true",
                  "How big the effect is",
                  "1 minus alpha"],
         opts_ar=["احتمال أن H₀ صحيحة",
                  "احتمال رؤية بيانات بنفس تطرف الملاحَظ، بفرض أن H₀ صحيحة",
                  "حجم التأثير","1 ناقص ألفا"],
         ans="B", exp_en="The technical definition.", exp_ar="التعريف التقنى.")

L.quiz_q(qid="M04-L03-Q2", qtype="MCQ",
         q_en="A small p-value indicates:",
         q_ar="قيمة p صغيرة تشير إلى:",
         opts_en=["H₀ is definitely false",
                  "Strong evidence against H₀",
                  "A large effect",
                  "The conclusion is certainly correct"],
         opts_ar=["H₀ خاطئة بالتأكيد","دليل قوى ضد H₀","تأثير كبير","الاستنتاج صحيح بالتأكيد"],
         ans="B", exp_en="Small p = strong evidence against H₀, not proof.",
         exp_ar="p صغيرة = دليل قوى ضد H₀، لا برهان.")

L.quiz_q(qid="M04-L03-Q3", qtype="TrueFalse",
         q_en="True or False: P-value tells you the size of the effect.",
         q_ar="صح أم خطأ: قيمة p تخبرك بحجم التأثير.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="P measures evidence, not effect size. Tiny effects can have tiny p with huge samples.",
         exp_ar="p تقيس الدليل، لا حجم التأثير. تأثيرات صغيرة قد تعطى p صغيرة مع عينات ضخمة.")

L.quiz_q(qid="M04-L03-Q4", qtype="MCQ",
         q_en="An A/B test gives p = 0.003. The most defensible statement:",
         q_ar="اختبار A/B يعطى p = 0.003. أكثر العبارات قابلة للدفاع:",
         opts_en=["The new design is definitely better",
                  "There's strong evidence the difference isn't just chance",
                  "p = 0.003 means 0.3% chance of being wrong",
                  "The effect is huge"],
         opts_ar=["التصميم الجديد أفضل قطعاً",
                  "هناك دليل قوى على أن الفرق ليس مجرد صدفة",
                  "p=0.003 يعنى 0.3% احتمال للخطأ","التأثير ضخم"],
         ans="B", exp_en="Tight, defensible language.", exp_ar="لغة دقيقة قابلة للدفاع.")

L.quiz_q(qid="M04-L03-Q5", qtype="MCQ",
         q_en="With 5 million users, an A/B test finds a difference with p = 0.001 but the effect is +0.05% conversion. The correct next step:",
         q_ar="مع 5 مليون مستخدم، اختبار A/B يجد فرقاً بـ p=0.001 لكن التأثير +0.05% تحويل. الخطوة الصحيحة:",
         opts_en=["Roll out immediately",
                  "Ignore - p too small to trust",
                  "Statistically real but evaluate business value before rollout",
                  "Re-run the test"],
         opts_ar=["إطلاق فورى","تجاهل - p صغيرة جداً للثقة",
                  "حقيقى إحصائياً لكن قيّم القيمة التجارية قبل الإطلاق","أعد التشغيل"],
         ans="C", exp_en="Statistical significance is achieved; practical value still requires judgment.",
         exp_ar="الدلالة الإحصائية متحققة؛ القيمة العملية تحتاج تقييماً.")
L.render_quiz(); L.save()

print("L04-01, L04-02, L04-03 done.")
