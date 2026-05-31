import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 02 - The Statistical Landscape"

# ============================================================
# L02-05 — Assumptions in Inferential Statistics
# ============================================================
L = Lesson(
    lid="M02-L05", module=2,
    title_en="Assumptions in Inferential Statistics",
    title_ar="الافتراضات فى الإحصاء الاستدلالى",
    objectives_en=[
        "Understand why every inferential method rests on assumptions.",
        "See how different assumptions can lead to different conclusions on the same data.",
        "Develop the habit of checking assumptions before trusting results.",
    ],
    objectives_ar=[
        "فهم لماذا يقوم كل منهج استدلالى على افتراضات.",
        "رؤية كيف يمكن لافتراضات مختلفة أن تنتج نتائج مختلفة على نفس البيانات.",
        "بناء عادة التحقق من الافتراضات قبل الوثوق بالنتائج.",
    ],
    sources=["02 The Statistical Landscape/Assumptions in Inferential Statistics.mp4",
             "Slides PDF: page 13"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Inferential statistics never operates in a vacuum. Every method makes assumptions — about how the data was collected, what 'random' means here, what 'fair' looks like. Different assumptions can lead to different conclusions, even on the same data.")
    ap(d, "الإحصاء الاستدلالى لا يعمل فى فراغ. كل منهج يصنع افتراضات — عن كيف جُمعت البيانات، وما المقصود بـ 'العشوائية'، وكيف يبدو 'العدل'. افتراضات مختلفة قد تنتج استنتاجات مختلفة على نفس البيانات.")
    callout(d, "key",
            "A core skill of every data analyst is matching the assumptions to the approach. If your assumptions are wrong, your p-values and confidence intervals are also wrong.",
            "مهارة جوهرية لأى محلل بيانات هى مطابقة الافتراضات مع المنهج. إذا كانت افتراضاتك خاطئة، فقيم p وفترات الثقة لديك خاطئة أيضاً.")
L.section("intro", "1 · No Method Is Assumption-Free",
          "1 · لا منهج يخلو من الافتراضات", s1)

def s2(d):
    ep(d, "Same question: 'Which of two finalists will win the world Rock-Paper-Scissors championship?' — three approaches, three different answers, all driven by different assumptions.")
    ap(d, "نفس السؤال: 'أى المتأهلَين للنهائى سيفوز ببطولة العالم لحجر-ورق-مقص؟' — ثلاثة مناهج، ثلاث إجابات مختلفة، كلها مدفوعة بافتراضات مختلفة.")
    kvtable(d, header=["Approach / المنهج","Assumption / الافتراض","Conclusion / الاستنتاج"],
        rows=[
            ("Classical\nكلاسيكى",
             "RPS has no real skill — outcomes equally likely.\nلا توجد مهارة فعلية — نتائج متساوية.",
             "50/50.\n50/50."),
            ("Bayesian\nبيزى",
             "Historically lefties beat righties 62% of the time.\nتاريخياً الأعسر يفوز 62%.",
             "Higher odds for the lefty.\nفرص أعلى للأعسر."),
            ("Frequentist\nتكرارى",
             "We can simulate matches and collect observations.\nيمكن محاكاة المباريات وجمع الملاحظات.",
             "Whatever the simulation produces (objective but environment-dependent).\nأياً كان ما تنتجه المحاكاة (موضوعى لكن يعتمد على البيئة)."),
        ])
L.section("example", "2 · Same Data, Three Conclusions",
          "2 · نفس البيانات، ثلاثة استنتاجات", s2)

def s3(d):
    h3(d, "Typical assumptions in frequentist tests","افتراضات شائعة فى الاختبارات التكرارية")
    eb(d, "Random sampling — every member of the population had a real chance to appear.")
    eb(d, "Independence — observations don't influence each other.")
    eb(d, "Distributional shape — often Normal, especially for parametric tests.")
    eb(d, "Sample size — large enough for the Central Limit Theorem to kick in.")
    ab(d, "العينة عشوائية — كل عضو فى المجتمع له فرصة حقيقية للظهور.")
    ab(d, "الاستقلال — المشاهدات لا تؤثر فى بعضها.")
    ab(d, "شكل التوزيع — عادة طبيعى، خصوصاً للاختبارات البارامترية.")
    ab(d, "حجم العينة — كبير بما يكفى لظهور أثر نظرية النهاية المركزية.")
L.section("concept", "3 · The Big Assumptions to Watch",
          "3 · الافتراضات الكبرى الواجب مراقبتها", s3)

def s4(d):
    callout(d, "use",
            "In a SaaS A/B test: assume each visitor's behavior is independent. If one visitor sees variant A and tweets about it, influencing 10 friends to load variant B, independence is broken — the p-value is no longer trustworthy.",
            "فى اختبار A/B لـ SaaS: نفترض أن سلوك كل زائر مستقل. إذا رأى زائر النسخة A وغرّد عنها فأثّر فى 10 أصدقاء ليحمّلوا النسخة B، انكسر الاستقلال — وأصبحت قيمة p غير موثوقة.")
    callout(d, "use",
            "In banking: assume default risk is independent across borrowers. The 2008 crisis broke this assumption — defaults clustered when the housing market fell, and risk models collapsed.",
            "فى البنوك: نفترض أن خطر التعثر مستقل بين المقترضين. أزمة 2008 كسرت هذا الافتراض — تجمعت حالات التعثر حين انهار سوق الإسكان، فانهارت نماذج المخاطر.")
L.section("practice", "4 · Why Assumptions Matter in Business",
          "4 · لماذا تهمّ الافتراضات فى الأعمال", s4)

def s5(d):
    callout(d, "trap",
            "Treating a model as 'objective' because it uses numbers. The numbers are objective; the assumptions underneath them are choices made by humans.",
            "اعتبار نموذج 'موضوعياً' لأنه يستخدم أرقاماً. الأرقام موضوعية، لكن الافتراضات التى تحتها اختيارات بشرية.")
    callout(d, "trap",
            "Skipping diagnostic checks. Plotting residuals, checking sample randomness, eyeballing distributions — five minutes here saves five hours later.",
            "تخطّى الفحوصات التشخيصية. رسم البواقى، فحص العشوائية، النظر السريع للتوزيعات — خمس دقائق هنا توفر خمس ساعات لاحقاً.")
L.section("traps", "5 · Common Traps", "5 · الفخاخ الشائعة", s5)

L.add_memory_aid("Every test has assumptions. State them. Check them. Wrong assumptions = wrong p-values.",
                 "كل اختبار له افتراضات. اذكرها. افحصها. الافتراضات الخاطئة = قيم p خاطئة.")
L.summary(
    en="Inferential statistics is only as trustworthy as its assumptions. A perfect calculation built on bad assumptions gives confidently wrong answers. The professional skill is not just running tests but examining whether each test's assumptions actually hold in your specific data.",
    ar="الإحصاء الاستدلالى موثوق بقدر افتراضاته. حسابات مثالية فوق افتراضات خاطئة تعطى إجابات خاطئة بثقة. المهارة الاحترافية ليست تشغيل الاختبارات فقط، بل فحص ما إذا كانت افتراضات الاختبار صحيحة فى بياناتك تحديداً."
)
L.render_memory()

L.quiz_q(qid="M02-L05-Q1", qtype="MCQ",
         q_en="Different inferential approaches sometimes give different answers on the same data because:",
         q_ar="المناهج الاستدلالية المختلفة تعطى أحياناً إجابات مختلفة على نفس البيانات لأن:",
         opts_en=["The math is wrong","They rest on different assumptions","Statistics is broken","Random chance"],
         opts_ar=["الرياضيات خاطئة","تقوم على افتراضات مختلفة","الإحصاء معطّل","الصدفة"],
         ans="B", exp_en="Assumptions, not math, drive the divergence.",
         exp_ar="الافتراضات لا الرياضيات هى السبب فى الاختلاف.")

L.quiz_q(qid="M02-L05-Q2", qtype="TrueFalse",
         q_en="True or False: A test with valid math always gives valid conclusions.",
         q_ar="صح أم خطأ: الاختبار صحيح الرياضيات يعطى استنتاجات صحيحة دائماً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="If the assumptions don't hold, the conclusions are wrong regardless of math.",
         exp_ar="إذا لم تتحقق الافتراضات، الاستنتاجات خاطئة مهما صحّت الرياضيات.")

L.quiz_q(qid="M02-L05-Q3", qtype="MCQ",
         q_en="Which is NOT a typical assumption in frequentist hypothesis tests?",
         q_ar="أى مما يلى ليس افتراضاً نموذجياً فى اختبارات الفرضيات التكرارية؟",
         opts_en=["Random sampling","Independence","Normality (often)","The analyst believes the result"],
         opts_ar=["عشوائية العينة","الاستقلال","التوزيع الطبيعى (غالباً)","يعتقد المحلل بالنتيجة"],
         ans="D", exp_en="The analyst's beliefs are not a statistical assumption.",
         exp_ar="معتقدات المحلل ليست افتراضاً إحصائياً.")

L.quiz_q(qid="M02-L05-Q4", qtype="MCQ",
         q_en="A bank's credit-risk model assumed defaults were independent. In 2008 this assumption broke because:",
         q_ar="نموذج مخاطر ائتمان بنكى افترض أن التعثرات مستقلة. فى 2008 انكسر هذا الافتراض لأن:",
         opts_en=["The math was wrong","Defaults clustered together during the crash",
                  "Banks stopped using statistics","Customers became smarter"],
         opts_ar=["كانت الرياضيات خاطئة","تجمعت حالات التعثر فى الانهيار",
                  "تخلّت البنوك عن الإحصاء","أصبح العملاء أذكى"],
         ans="B", exp_en="Systemic shock broke independence — defaults correlated heavily across borrowers.",
         exp_ar="الصدمة النظامية كسرت الاستقلال — ارتبطت التعثرات بشدة بين المقترضين.")

L.quiz_q(qid="M02-L05-Q5", qtype="MCQ",
         q_en="The best response when you suspect an assumption is violated is:",
         q_ar="الاستجابة الأفضل حين تشك فى انتهاك افتراض هى:",
         opts_en=["Ignore it - the test is robust",
                  "Run the test anyway and report the p-value",
                  "Diagnose, fix data, or switch to a method that fits",
                  "Stop doing statistics"],
         opts_ar=["تجاهلها - الاختبار قوى",
                  "شغّل الاختبار وأبلغ عن قيمة p",
                  "شخّص، عالج البيانات، أو غيّر المنهج",
                  "توقف عن الإحصاء"],
         ans="C", exp_en="Diagnose → repair → re-run, or switch methods.",
         exp_ar="شخّص ← عالج ← أعد التشغيل، أو غيّر المنهج.")
L.render_quiz(); L.save()

# ============================================================
# L02-06 — The Flowchart - Analysis Selection
# ============================================================
L = Lesson(
    lid="M02-L06", module=2,
    title_en="The Flowchart - Analysis Selection",
    title_ar="مخطط اختيار التحليل",
    objectives_en=[
        "Follow a decision flowchart from question to analysis method.",
        "Recognise when descriptive stats is enough, when inference is needed, and when ML is appropriate.",
    ],
    objectives_ar=[
        "اتباع مخطط قرار من السؤال إلى المنهج التحليلى.",
        "إدراك متى يكفى الإحصاء الوصفى، ومتى نحتاج الاستدلالى، ومتى نلجأ للتعلم الآلى.",
    ],
    sources=["02 The Statistical Landscape/The Flow Chart.mp4",
             "Slides PDF: pages 14-15"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "The statistical toolbox is large. Pulling out the right tool requires asking the right questions, in the right order.")
    ap(d, "صندوق الأدوات الإحصائية كبير. اختيار الأداة الصحيحة يتطلب طرح الأسئلة الصحيحة بالترتيب الصحيح.")
L.section("intro", "1 · The Toolbox Problem", "1 · مشكلة الصندوق المتعدد", s1)

def s2(d):
    ep(d, "Walk down the flowchart by answering yes/no questions:")
    ap(d, "انزل عبر المخطط بإجابة أسئلة نعم/لا:")
    en_(d, "Do we understand the data? If NO → explore it (descriptive stats first).")
    en_(d, "Are we making a decision? If NO → just summarise (descriptive stats + visualisation).")
    en_(d, "Is the decision important and one-time? If YES → inferential statistics.")
    en_(d, "Are we making many automated decisions? If YES → machine learning.")
    an_(d, "هل نفهم البيانات؟ إذا لا → استكشفها (إحصاء وصفى أولاً).")
    an_(d, "هل نتخذ قراراً؟ إذا لا → فقط لخّص (وصفى + تصور).")
    an_(d, "هل القرار مهم ومرة واحدة؟ إذا نعم → الاستدلالى.")
    an_(d, "هل نتخذ قرارات كثيرة آلية؟ إذا نعم → تعلم آلى.")
L.section("concept", "2 · The Decision Flow", "2 · مسار القرار", s2)

def s3(d):
    kvtable(d, header=["Scenario / السيناريو","Best tool / الأداة الأفضل","Why / لماذا"],
        rows=[
            ("Monthly sales report.\nتقرير المبيعات الشهرى.",
             "Descriptive stats.\nالإحصاء الوصفى.",
             "Just describing what happened.\nمجرد وصف ما حدث."),
            ("Was the new campaign more profitable?\nهل الحملة الجديدة أكثر ربحاً؟",
             "Inferential test.\nاختبار استدلالى.",
             "One big yes/no decision.\nقرار كبير واحد."),
            ("Predict churn for 2M customers next month.\nتوقع مغادرة 2 مليون عميل الشهر القادم.",
             "Machine learning.\nتعلم آلى.",
             "Millions of small predictions.\nملايين التنبؤات الصغيرة."),
            ("Understand which features matter for churn.\nفهم أى الخصائص تهم فى المغادرة.",
             "Inferential modelling (regression).\nنمذجة استدلالية (انحدار).",
             "Need interpretable, defensible reasoning.\nنحتاج تفسيراً قابلاً للدفاع."),
        ])
L.section("example", "3 · Four Real Scenarios", "3 · أربعة سيناريوهات حقيقية", s3)

def s4(d):
    callout(d, "use",
            "Marketing teams typically: describe campaign performance daily (descriptive), test 1-2 big creatives per month (inferential), and use ML for bid optimisation on millions of ad slots (machine learning).",
            "فرق التسويق تفعل عادة: وصف أداء الحملة يومياً (وصفى)، اختبار 1-2 إبداع كبير شهرياً (استدلالى)، وتعلم آلى لتحسين العروض على ملايين خانات الإعلان (تعلم آلى).")
    callout(d, "use",
            "Finance: monthly P&L summary (descriptive), test a new credit policy on a pilot region (inferential), automate loan approvals (ML).",
            "المالية: ملخص الأرباح والخسائر الشهرى (وصفى)، اختبار سياسة ائتمان جديدة فى منطقة تجريبية (استدلالى)، أتمتة موافقات القروض (تعلم آلى).")
L.section("practice", "4 · The Mix in a Real Company", "4 · المزيج فى شركة حقيقية", s4)

def s5(d):
    callout(d, "trap",
            "Reaching for machine learning when descriptive statistics would suffice. Many 'AI projects' fail because the team skipped a simple bar chart that would have answered the question.",
            "اللجوء للتعلم الآلى حين يكفى الإحصاء الوصفى. كثير من 'مشاريع الذكاء الاصطناعى' تفشل لأن الفريق تجاوز رسماً بيانياً بسيطاً كان سيجيب السؤال.")
L.section("traps", "5 · Common Trap", "5 · فخّ شائع", s5)

L.add_memory_aid("Understand data → describe → infer (big decisions) → ML (many automated decisions).",
                 "افهم البيانات ← صف ← استدل (قرارات كبرى) ← تعلم آلى (قرارات آلية كثيرة).")
L.summary(
    en="Choosing the right tool is half of being a good analyst. The flowchart frames it as a sequence: understand the data, decide what kind of decision you're making, then pick descriptive, inferential, or ML accordingly. Most projects use a mix of all three.",
    ar="اختيار الأداة الصحيحة نصف مهارة المحلل الجيد. المخطط يضعها كتسلسل: افهم البيانات، حدد نوع القرار، ثم اختر الوصفى أو الاستدلالى أو التعلم الآلى. معظم المشاريع تمزج الثلاثة."
)
L.render_memory()

L.quiz_q(qid="M02-L06-Q1", qtype="MCQ",
         q_en="A daily revenue dashboard is primarily:",
         q_ar="لوحة إيرادات يومية هى أساساً:",
         opts_en=["ML","Inferential","Descriptive","Bayesian"], opts_ar=["تعلم آلى","استدلالى","وصفى","بيزى"],
         ans="C", exp_en="Reporting known totals = descriptive.", exp_ar="عرض إجماليات معروفة = وصفى.")

L.quiz_q(qid="M02-L06-Q2", qtype="MCQ",
         q_en="An e-commerce site wants to decide whether to roll out a new checkout. Best tool:",
         q_ar="موقع تجارة يريد تقرير إطلاق صفحة دفع جديدة. الأداة الأفضل:",
         opts_en=["Descriptive only","Inferential A/B test","ML model","Quick guess"],
         opts_ar=["وصفى فقط","اختبار A/B استدلالى","نموذج تعلم آلى","تخمين سريع"],
         ans="B", exp_en="One big yes/no decision → inferential test.",
         exp_ar="قرار كبير واحد نعم/لا ← اختبار استدلالى.")

L.quiz_q(qid="M02-L06-Q3", qtype="MCQ",
         q_en="A bank scoring 100,000 loan applications per day for default risk uses:",
         q_ar="بنك يقيّم 100,000 طلب قرض يومياً لمخاطر التعثر يستخدم:",
         opts_en=["Descriptive","Inferential","Machine learning","Bayesian only"],
         opts_ar=["وصفى","استدلالى","تعلم آلى","بيزى فقط"],
         ans="C", exp_en="Many automated decisions → ML.", exp_ar="قرارات آلية كثيرة ← تعلم آلى.")

L.quiz_q(qid="M02-L06-Q4", qtype="TrueFalse",
         q_en="True or False: You should always start a project with an ML model.",
         q_ar="صح أم خطأ: يجب البدء دائماً بنموذج تعلم آلى.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Almost every project benefits from descriptive exploration first.",
         exp_ar="معظم المشاريع تستفيد من الاستكشاف الوصفى أولاً.")

L.quiz_q(qid="M02-L06-Q5", qtype="MCQ",
         q_en="If we don't yet understand our data well, the first step is:",
         q_ar="إذا لم نفهم بياناتنا جيداً بعد، فالخطوة الأولى هى:",
         opts_en=["Train an ML model","Run a hypothesis test","Explore with descriptive stats and charts","Hire more analysts"],
         opts_ar=["تدريب نموذج تعلم آلى","تشغيل اختبار فرضيات","الاستكشاف بالإحصاء الوصفى والرسوم","توظيف محللين"],
         ans="C", exp_en="Always begin with exploration.",
         exp_ar="ابدأ دائماً بالاستكشاف.")
L.render_quiz(); L.save()

# ============================================================
# L02-07 — Business Intelligence vs Data Science
# ============================================================
L = Lesson(
    lid="M02-L07", module=2,
    title_en="Business Intelligence vs Data Science",
    title_ar="ذكاء الأعمال مقابل علم البيانات",
    objectives_en=[
        "Distinguish the roles of BI analyst, ML specialist, and data scientist.",
        "Understand where statistics fits in each role.",
        "Set yourself up to grow across the skill spectrum.",
    ],
    objectives_ar=[
        "التمييز بين أدوار محلل ذكاء الأعمال، أخصائى التعلم الآلى، وعالم البيانات.",
        "فهم أين يقع الإحصاء فى كل دور.",
        "تجهيز نفسك للنمو عبر طيف المهارات.",
    ],
    sources=["Slides PDF: page 16",
             "Translation: ترجمة الشريحة.docx page 16"], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Three closely-related roles live on a spectrum. Each leans on statistics, but in different proportions and for different purposes.")
    ap(d, "ثلاثة أدوار متقاربة تعيش على طيف واحد. كل دور يعتمد على الإحصاء، لكن بنسب مختلفة ولأغراض مختلفة.")
L.section("intro", "1 · A Spectrum, Not a Wall", "1 · طيف لا جدار", s1)

def s2(d):
    kvtable(d, header=["Role / الدور","Main focus / التركيز","Stats they use most / أكثر إحصاء يستخدمونه"],
        rows=[
            ("BI Analyst / Data Analyst\nمحلل BI / محلل بيانات",
             "Descriptive reporting, dashboards, visualisations.\nتقارير وصفية، لوحات بيانات، تصورات.",
             "Means, medians, totals, trends.\nمتوسطات، وسطاء، إجماليات، اتجاهات."),
            ("ML Specialist\nأخصائى تعلم آلى",
             "Building and scaling predictive models.\nبناء وتشغيل نماذج تنبؤية.",
             "Probability, optimisation, regularisation.\nالاحتمالات، التحسين، التنظيم."),
            ("Data Scientist\nعالم بيانات",
             "Connecting tools to assumptions and context.\nربط الأدوات بالافتراضات والسياق.",
             "Inference, experiment design, full stat toolkit.\nالاستدلال، تصميم التجارب، كامل الأدوات."),
        ])
L.section("definitions", "2 · The Three Roles", "2 · الأدوار الثلاثة", s2)

def s3(d):
    callout(d, "key",
            "What sets a data scientist apart is the awareness of which assumptions apply, which tool fits which context, and how to defend a conclusion. That awareness is exactly what this course aims to build.",
            "ما يميّز عالم البيانات هو وعيه بأى الافتراضات تنطبق، وأى أداة تناسب أى سياق، وكيف يدافع عن استنتاجه. هذا الوعى تحديداً هو ما تبنيه هذه الدورة.")
L.section("concept", "3 · The Differentiator", "3 · الفارق الجوهرى", s3)

def s4(d):
    h3(d, "A growth path", "مسار التطور")
    en_(d, "Start as a BI analyst — master descriptive stats and dashboards.")
    en_(d, "Add inferential statistics — start running A/B tests and reading p-values.")
    en_(d, "Add ML basics — build a simple regression and classification model.")
    en_(d, "Become a data scientist — understand why each tool works and when it breaks.")
    an_(d, "ابدأ محلل BI — أتقن الإحصاء الوصفى ولوحات البيانات.")
    an_(d, "أضف الاستدلال — اختبارات A/B وقراءة قيم p.")
    an_(d, "أضف أساسيات التعلم الآلى — انحدار وتصنيف بسيط.")
    an_(d, "كن عالم بيانات — افهم لماذا تعمل كل أداة ومتى تتعطل.")
L.section("practice", "4 · A Career Roadmap", "4 · خريطة طريق مهنية", s4)

L.add_memory_aid("BI describes. ML predicts at scale. Data Science wraps stats around context.",
                 "BI يصف. التعلم الآلى يتنبأ على نطاق واسع. علم البيانات يلفّ الإحصاء حول السياق.")
L.summary(
    en="BI analysts, ML specialists, and data scientists all live on the same data spectrum. The defining skill of a data scientist is matching assumptions to tools. Whichever role you target, statistics is your foundation — and this course is precisely that foundation.",
    ar="محللو BI، وأخصائيو التعلم الآلى، وعلماء البيانات يعيشون على نفس طيف البيانات. المهارة المميّزة لعالم البيانات هى مطابقة الافتراضات للأدوات. أياً كان الدور الذى تستهدفه، الإحصاء أساسك — وهذه الدورة هى ذلك الأساس بالضبط."
)
L.render_memory()

L.quiz_q(qid="M02-L07-Q1", qtype="MCQ",
         q_en="Building a real-time dashboard of yesterday's sales is the typical job of:",
         q_ar="بناء لوحة بيانات فورية لمبيعات الأمس هو عمل اعتيادى لـ:",
         opts_en=["ML specialist","BI analyst","Hardware engineer","Database admin"],
         opts_ar=["أخصائى تعلم آلى","محلل BI","مهندس عتاد","مسؤول قاعدة بيانات"],
         ans="B", exp_en="Dashboards = descriptive reporting = BI.",
         exp_ar="لوحات البيانات = تقارير وصفية = BI.")

L.quiz_q(qid="M02-L07-Q2", qtype="MCQ",
         q_en="Designing an A/B test, picking the test type, and defending the conclusion is the job of:",
         q_ar="تصميم اختبار A/B، اختيار نوع الاختبار، والدفاع عن الاستنتاج هو عمل:",
         opts_en=["BI analyst alone","ML specialist alone","Data scientist","Sales manager"],
         opts_ar=["محلل BI فقط","أخصائى تعلم آلى فقط","عالم بيانات","مدير مبيعات"],
         ans="C", exp_en="This is the matching-assumptions-to-tools skill.",
         exp_ar="هذه مهارة مطابقة الافتراضات بالأدوات.")

L.quiz_q(qid="M02-L07-Q3", qtype="TrueFalse",
         q_en="True or False: A data scientist doesn't need to know descriptive statistics.",
         q_ar="صح أم خطأ: عالم البيانات لا يحتاج معرفة الإحصاء الوصفى.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Descriptive stats are the foundation — you can't infer without first describing.",
         exp_ar="الوصفى هو الأساس — لا يمكن الاستدلال دون وصف.")

L.quiz_q(qid="M02-L07-Q4", qtype="MCQ",
         q_en="Across the three roles, the COMMON foundation is:",
         q_ar="عبر الأدوار الثلاثة، الأساس المشترك هو:",
         opts_en=["Python","Statistics","SQL","Tableau"], opts_ar=["بايثون","الإحصاء","SQL","Tableau"],
         ans="B", exp_en="Statistics underpins every analytical role.",
         exp_ar="الإحصاء يقع تحت كل دور تحليلى.")

L.quiz_q(qid="M02-L07-Q5", qtype="MCQ",
         q_en="A small startup hires its first analyst. They will mostly do:",
         q_ar="شركة ناشئة صغيرة توظف أول محلل لها. سيؤدى غالباً:",
         opts_en=["Descriptive stats and basic inference",
                  "Only build deep neural networks",
                  "Pure research papers",
                  "Nothing involving data"],
         opts_ar=["إحصاء وصفى واستدلال بسيط",
                  "بناء شبكات عصبية عميقة فقط",
                  "أوراق بحثية بحتة",
                  "لا شىء يتعلق بالبيانات"],
         ans="A", exp_en="Early stage = descriptive + light inference.",
         exp_ar="المراحل الأولى = وصفى + استدلال خفيف.")
L.render_quiz(); L.save()

print("L02-05, L02-06, L02-07 done.")

# Build Module 02 index
module_index = {
    "id":"M02","module":2,"type":"module",
    "title_en":"The Statistical Landscape",
    "title_ar":"المشهد الإحصائى",
    "lessons":[
        {"id":"M02-L01","title_en":"Chapter Introduction - The Statistical Landscape","title_ar":"مقدمة الفصل - المشهد الإحصائى"},
        {"id":"M02-L02","title_en":"The Anatomy of a Data Set","title_ar":"تشريح مجموعة البيانات"},
        {"id":"M02-L03","title_en":"Descriptive vs Inferential Statistics","title_ar":"الإحصاء الوصفى مقابل الاستدلالى"},
        {"id":"M02-L04","title_en":"Approaches to Inferring Probability","title_ar":"مناهج الاستدلال الاحتمالى"},
        {"id":"M02-L05","title_en":"Assumptions in Inferential Statistics","title_ar":"الافتراضات فى الإحصاء الاستدلالى"},
        {"id":"M02-L06","title_en":"The Flowchart - Analysis Selection","title_ar":"مخطط اختيار التحليل"},
        {"id":"M02-L07","title_en":"Business Intelligence vs Data Science","title_ar":"ذكاء الأعمال مقابل علم البيانات"},
    ],
    "sources":["02 The Statistical Landscape/*", "Slides PDF pages 3-16",
               "Glossary Terms: Descriptive Statistics, Inferential Statistics, Bayesian"]
}
with open(os.path.join(OUT,"M02 - Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(module_index, f, ensure_ascii=False, indent=2)
print("Module 02 index written.")
