import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 02 - The Statistical Landscape"
os.makedirs(OUT, exist_ok=True)

# ───────── Cover ─────────
d = setup(); fc(d)
titlep(d, module_no=2,
       module_en="The Statistical Landscape",
       module_ar="المشهد الإحصائى",
       subtitle_en="Population vs Sample · Descriptive vs Inferential · The Flowchart",
       subtitle_ar="المجتمع والعينة · الوصفى والاستدلالى · مخطط اختيار التحليل")
end_m(d, os.path.join(OUT, "M02 - Cover.docx"))

# ============================================================
# L02-01 — Chapter Introduction
# ============================================================
L = Lesson(
    lid="M02-L01", module=2,
    title_en="Chapter Introduction - The Statistical Landscape",
    title_ar="مقدمة الفصل - المشهد الإحصائى",
    objectives_en=[
        "Define statistics in plain language.",
        "See the three big themes: descriptive, inferential, and choosing the right approach.",
        "Connect statistics to everyday business decisions.",
    ],
    objectives_ar=[
        "تعريف الإحصاء بلغة بسيطة.",
        "رؤية المحاور الثلاثة الكبرى: الوصفى، الاستدلالى، واختيار المنهج المناسب.",
        "ربط الإحصاء بقرارات الأعمال اليومية.",
    ],
    sources=[
        "Folder: 02 The Statistical Landscape/Chapter Introduction.mp4",
        "Slides PDF: pages 3-4",
        "Translation: ترجمة الشريحة.docx pages 3-4",
        "OpenStax: Chapter 1.1",
    ], out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "Statistics is the process of collecting, summarising, and analysing data so we can make decisions or inferences.")
    ap(d, "الإحصاء هو عملية جمع البيانات وتلخيصها وتحليلها حتى نتمكن من اتخاذ القرارات أو الاستدلالات.")
    ep(d, "It is the universal language that lets us turn raw observations into evidence-based action — whether we are a hospital, a retailer, a bank, or a startup measuring user clicks.")
    ap(d, "هو اللغة العالمية التى تحوّل الملاحظات الخام إلى إجراءات مدعومة بالأدلة — سواء كنا مستشفى، أو متجراً، أو بنكاً، أو شركة ناشئة تقيس نقرات المستخدمين.")
L.section("intro", "1 · What This Chapter Is About", "1 · عمّا يدور هذا الفصل", s1)

def s2(d):
    ep(d, "The journey of this module unfolds in three stages:")
    ap(d, "تنقسم رحلة هذا الموديول إلى ثلاث مراحل:")
    en_(d, "First: define descriptive vs inferential statistics — the two foundational families.")
    en_(d, "Next: explore the three big approaches to inference — classical, frequentist, and Bayesian.")
    en_(d, "Finally: learn how to pick the right tool — descriptive statistics, inferential statistics, or machine learning.")
    an_(d, "أولاً: تعريف الإحصاء الوصفى مقابل الاستدلالى — العائلتان الأساسيتان.")
    an_(d, "ثانياً: استكشاف المناهج الثلاثة للاستدلال — الكلاسيكى، التكرارى، والبيزى.")
    an_(d, "أخيراً: تعلّم كيف تختار الأداة المناسبة — إحصاء وصفى، استدلالى، أم تعلّم آلى.")
L.section("concept", "2 · The Three-Stage Journey", "2 · الرحلة ذات الثلاث مراحل", s2)

def s3(d):
    ep(d, "Statistics is never abstract. It always answers a real question:")
    ap(d, "الإحصاء ليس مفهوماً مجرداً. هو دائماً يجيب على سؤال حقيقى:")
    eb(d, "A retailer asks: 'What is our average basket size this month, and is it bigger than last month?'")
    eb(d, "A bank asks: 'What fraction of our loan applicants default — and which features predict default?'")
    eb(d, "A SaaS company asks: 'Did the new onboarding flow truly raise activation, or are we fooling ourselves?'")
    eb(d, "An e-commerce team asks: 'How many customers do we need to survey to know real satisfaction within ±3 points?'")
    ab(d, "متجر يسأل: ما متوسط حجم السلة هذا الشهر، وهل هو أكبر من الشهر الماضى؟")
    ab(d, "بنك يسأل: ما نسبة المتعثرين فى تطبيق القروض — وأى الخصائص تتنبأ بالتعثر؟")
    ab(d, "شركة SaaS تسأل: هل رفعت تدفقات الترحيب الجديدة نسبة التفعيل فعلاً، أم نخدع أنفسنا؟")
    ab(d, "فريق تجارة إلكترونية يسأل: كم عميلاً يجب أن نستطلع لنعرف الرضا الحقيقى بدقة ±3 نقاط؟")
    callout(d, "meaning",
            "Behind every dashboard, every A/B test, every survey is a statistical question. Statistics is the toolkit; business gives you the questions.",
            "خلف كل لوحة بيانات، وكل اختبار A/B، وكل استبيان، سؤال إحصائى. الإحصاء أداة، والأعمال هى التى تعطيك الأسئلة.")
L.section("practice", "3 · Statistics in the Real Business World",
          "3 · الإحصاء فى عالم الأعمال الحقيقى", s3)

L.add_memory_aid("Statistics = collect → summarise → analyse → decide.",
                 "الإحصاء = جمع ← تلخيص ← تحليل ← قرار.")
L.summary(
    en="This chapter zooms out before zooming in. Statistics is a four-step process — collect, summarise, analyse, decide — and the rest of the module gives you the language and tools to do each step well, with real business intuition baked in.",
    ar="هذا الفصل يأخذنا نحو الصورة الواسعة قبل التعمّق. الإحصاء عملية من أربع خطوات — جمع، تلخيص، تحليل، قرار — وبقية الموديول يمنحك اللغة والأدوات لتؤدّى كل خطوة جيداً، مع فهم عملى لأعمال الواقع."
)
L.render_memory()

L.quiz_q(qid="M02-L01-Q1", qtype="MCQ",
         q_en="Statistics is BEST defined as:",
         q_ar="أفضل تعريف للإحصاء هو:",
         opts_en=["A branch of mathematics about numbers only.",
                  "The process of collecting, summarising, and analysing data to make decisions.",
                  "A type of software like Excel.",
                  "Probability theory."],
         opts_ar=["فرع من الرياضيات يهتم بالأرقام فقط.",
                  "عملية جمع البيانات وتلخيصها وتحليلها لاتخاذ القرارات.",
                  "نوع من البرمجيات مثل Excel.",
                  "نظرية الاحتمالات."],
         ans="B",
         exp_en="Statistics is the end-to-end process, not just numbers, not just one program. Probability is a tool within it.",
         exp_ar="الإحصاء عملية متكاملة، ليس مجرد أرقام، ولا برنامج واحد. الاحتمالات أداة داخله.")

L.quiz_q(qid="M02-L01-Q2", qtype="MCQ",
         q_en="Which step is NOT a part of the statistics process?",
         q_ar="أى خطوة من هذه ليست جزءاً من عملية الإحصاء؟",
         opts_en=["Collecting data","Summarising data","Guessing without data","Making decisions"],
         opts_ar=["جمع البيانات","تلخيص البيانات","التخمين بدون بيانات","اتخاذ القرار"],
         ans="C",
         exp_en="Statistics is evidence-based. Guessing without data is the opposite of what statistics does.",
         exp_ar="الإحصاء قائم على الأدلة. التخمين دون بيانات هو عكس ما يفعله الإحصاء.")

L.quiz_q(qid="M02-L01-Q3", qtype="MCQ",
         q_en="A retailer wants to know whether last month's revenue was higher than the previous month. This is primarily:",
         q_ar="متجر يريد معرفة هل إيرادات الشهر الماضى أعلى من الشهر السابق. هذا أساساً:",
         opts_en=["Inferential","Descriptive","Machine learning","None of the above"],
         opts_ar=["استدلالى","وصفى","تعلم آلى","لا شىء مما سبق"],
         ans="B",
         exp_en="Describing what already happened — sums, comparisons of known totals — is descriptive statistics.",
         exp_ar="وصف ما حدث بالفعل — مجاميع، مقارنات لقيم معروفة — هو إحصاء وصفى.")

L.quiz_q(qid="M02-L01-Q4", qtype="TrueFalse",
         q_en="True or False: Statistics is only used in scientific research, not in business.",
         q_ar="صح أم خطأ: الإحصاء يُستخدم فى البحث العلمى فقط، لا فى الأعمال.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="Every modern business runs on statistics: marketing tests, finance models, ops dashboards, product analytics.",
         exp_ar="كل عمل حديث يقوم على الإحصاء: اختبارات تسويق، نماذج مالية، لوحات تشغيل، تحليلات منتجات.")

L.quiz_q(qid="M02-L01-Q5", qtype="MCQ",
         q_en="Which is the BEST way to think about this module?",
         q_ar="أى عبارة أفضل لفهم هذا الموديول؟",
         opts_en=["A list of formulas to memorise.",
                  "A map: where each tool fits in the data-decision pipeline.",
                  "An introduction to machine learning only.",
                  "A history of statistics."],
         opts_ar=["قائمة بصيغ للحفظ.",
                  "خريطة: أين تقع كل أداة فى مسار البيانات والقرار.",
                  "مقدمة فى التعلم الآلى فقط.",
                  "تاريخ الإحصاء."],
         ans="B",
         exp_en="The landscape module is a map. Later modules drill into each tool.",
         exp_ar="موديول المشهد هو خريطة. الموديولات اللاحقة تتعمق فى كل أداة.")
L.render_quiz(); L.save()

# ============================================================
# L02-02 — The Anatomy of a Data Set
# ============================================================
L = Lesson(
    lid="M02-L02", module=2,
    title_en="The Anatomy of a Data Set",
    title_ar="تشريح مجموعة البيانات",
    objectives_en=[
        "Distinguish population, sample, and dataset.",
        "Identify rows, columns, variables, target, and unique ID.",
        "Recognise when a dataset is a sample vs the full population.",
    ],
    objectives_ar=[
        "التمييز بين المجتمع والعينة ومجموعة البيانات.",
        "تحديد الصفوف والأعمدة والمتغيرات والهدف والمعرف الفريد.",
        "إدراك متى تكون البيانات عينة ومتى تكون المجتمع كاملاً.",
    ],
    sources=["02 The Statistical Landscape/The Anatomy of a Data Set.mp4",
             "Slides PDF: pages 5-8",
             "Translation: ترجمة الشريحة.docx pages 5-8"],
    out_dir=OUT)
L.header(); L.objectives()

def s1(d):
    ep(d, "A dataset is the collection of data we work with for our analysis. Visually, it is a table: rows of observations, columns of variables.")
    ap(d, "مجموعة البيانات هى البيانات التى نعمل عليها فى التحليل. بصرياً، هى جدول: صفوف من المشاهدات، أعمدة من المتغيرات.")
    callout(d, "key",
            "Row = one observation (one customer, one transaction, one tree). Column = one variable (age, price, mortality). The Target is the column we want to predict or explain.",
            "الصف = مشاهدة واحدة (عميل، عملية شراء، شجرة). العمود = متغير واحد (عمر، سعر، نفوق). الهدف هو العمود الذى نريد التنبؤ به أو شرحه.")
L.section("intro", "1 · The Picture of a Dataset", "1 · صورة مجموعة البيانات", s1)

def s2(d):
    kvtable(d,
        header=["Concept / المفهوم","Definition / التعريف","Business example / مثال أعمال"],
        rows=[
            ("Population\nالمجتمع",
             "Every member you care about.\nكل عضو يهمنا.",
             "All Egyptian online shoppers.\nكل المتسوقين عبر الإنترنت فى مصر."),
            ("Sample\nالعينة",
             "A subset selected for study.\nمجموعة فرعية مختارة للدراسة.",
             "1,200 surveyed shoppers.\n1,200 متسوق تم استطلاعهم."),
            ("Dataset\nمجموعة البيانات",
             "The structured table you actually have.\nالجدول الفعلى الذى تملكه.",
             "Excel file with 1,200 rows × 15 cols.\nملف Excel به 1,200 صف × 15 عمود."),
            ("Variable / Feature\nمتغير / خاصية",
             "A column.\nعمود.",
             "Customer age, total spend, channel.\nعمر العميل، إجمالى الإنفاق، القناة."),
            ("Target variable\nالمتغير الهدف",
             "Column we want to predict.\nالعمود المراد التنبؤ به.",
             "Did the customer churn? (Yes/No)\nهل غادر العميل؟ (نعم/لا)"),
            ("Unique ID\nالمعرف الفريد",
             "One value per row, never repeated.\nقيمة واحدة لكل صف، لا تتكرر.",
             "Customer_ID = 10043.\nمعرّف العميل = 10043."),
        ])
L.section("definitions", "2 · Core Vocabulary", "2 · المفردات الأساسية", s2)

def s3(d):
    ep(d, "In statistics we almost always work with a sample. The full population is usually too large, too expensive, or impossible to capture entirely. We then use the sample to make inferences about the population.")
    ap(d, "فى الإحصاء، نعمل دائماً تقريباً على عينة. المجتمع كاملاً غالباً ما يكون كبيراً جداً، أو مكلفاً، أو مستحيل الحصر. ثم نستخدم العينة لاستدلالات حول المجتمع.")
    callout(d, "example",
            "A bank with 5 million credit-card customers won't analyse every transaction. It pulls a random sample of 50,000 to estimate fraud rate, average spend, and demographics.",
            "بنك لديه 5 مليون عميل بطاقة ائتمانية لن يحلل كل المعاملات. سيسحب عينة عشوائية من 50,000 ليقدّر معدل الاحتيال ومتوسط الإنفاق والديموغرافيا.")
L.section("concept", "3 · Why We Use Samples", "3 · لماذا نستخدم العينات", s3)

def s4(d):
    callout(d, "example",
            "Imagine a coffee chain with 320 branches collects daily sales:\n• Rows = days (one per branch per day).\n• Columns = branch, region, weather, promotion_active, total_sales.\n• Target = total_sales (or churn, return rate, etc.).\n• Unique ID = (branch_id + date).",
            "تخيل سلسلة قهوة بـ 320 فرعاً تجمع المبيعات اليومية:\n• الصفوف = الأيام (يوم لكل فرع).\n• الأعمدة = الفرع، المنطقة، الطقس، عرض ترويجى، إجمالى المبيعات.\n• الهدف = إجمالى المبيعات (أو معدل المغادرة أو الإرجاع).\n• المعرف الفريد = (معرف الفرع + التاريخ).")
L.section("example", "4 · Worked Example - A Coffee Chain",
          "4 · مثال محلول - سلسلة قهوة", s4)

def s5(d):
    callout(d, "trap",
            "Treating a biased sample as if it represented the population. Surveying only your existing happy customers and concluding 'everyone loves us' is a textbook error.",
            "معاملة عينة متحيزة وكأنها تمثل المجتمع. استطلاع العملاء السعداء فقط ثم استنتاج أن الجميع يحبوننا هو خطأ كلاسيكى.")
    callout(d, "trap",
            "Forgetting the unique ID. Without it, you can't deduplicate, can't audit, can't roll back errors.",
            "نسيان المعرف الفريد. بدونه لا يمكنك إزالة المكررات ولا التدقيق ولا التراجع عن الأخطاء.")
L.section("traps", "5 · Common Traps", "5 · أخطاء شائعة", s5)

def s6(d):
    ep(d, "Before any analysis: open the dataset and answer four questions.")
    ap(d, "قبل أى تحليل: افتح البيانات وأجب عن أربعة أسئلة.")
    en_(d, "How many rows? How many columns?")
    en_(d, "What is the unique ID? Is it truly unique?")
    en_(d, "Which column is the target?")
    en_(d, "Is this a sample of something bigger, or the full population?")
    an_(d, "كم عدد الصفوف؟ كم عدد الأعمدة؟")
    an_(d, "ما المعرف الفريد؟ هل هو فريد فعلاً؟")
    an_(d, "أى عمود هو الهدف؟")
    an_(d, "هل هذه عينة من شىء أكبر، أم المجتمع كاملاً؟")
L.section("practice", "6 · Your Pre-Analysis Checklist",
          "6 · قائمتك قبل بدء التحليل", s6)

L.add_memory_aid("Row = observation. Column = variable. Target = what you predict. Unique ID = one value per row.",
                 "صف = مشاهدة. عمود = متغير. الهدف = ما تتنبأ به. المعرف الفريد = قيمة لكل صف.")
L.add_memory_aid("Sample ⊂ Population. We study the sample to learn about the population.",
                 "العينة جزء من المجتمع. ندرس العينة لنعرف المجتمع.")
L.summary(
    en="A dataset is your table of analysis. Rows are observations, columns are variables, one column is usually the target, and a unique ID keeps the rows distinct. Most real datasets are samples, not full populations, which sets up everything in inferential statistics.",
    ar="مجموعة البيانات هى جدول التحليل. الصفوف مشاهدات، الأعمدة متغيرات، عمود واحد هو الهدف غالباً، والمعرف الفريد يميّز الصفوف. معظم البيانات الحقيقية عينات لا مجتمعات كاملة، وعلى هذا يُبنى الإحصاء الاستدلالى لاحقاً."
)
L.render_memory()

L.quiz_q(qid="M02-L02-Q1", qtype="MCQ",
         q_en="In a customer dataset, each row represents:",
         q_ar="فى بيانات العملاء، كل صف يمثّل:",
         opts_en=["A column","A single observation (one customer)","All customers combined","A variable"],
         opts_ar=["عمود","مشاهدة واحدة (عميل واحد)","كل العملاء مجمّعين","متغير"],
         ans="B",
         exp_en="A row is one observation. A column is a variable.",
         exp_ar="الصف مشاهدة واحدة. العمود متغير.")

L.quiz_q(qid="M02-L02-Q2", qtype="MCQ",
         q_en="An e-commerce team analyses 1,500 random customer orders out of 850,000. The 1,500 orders are:",
         q_ar="فريق تجارة إلكترونية يحلل 1,500 طلب عشوائى من بين 850,000. الـ 1,500 هذه:",
         opts_en=["The population","A sample of the population","Unique IDs","Target variables"],
         opts_ar=["المجتمع","عينة من المجتمع","معرفات فريدة","متغيرات هدف"],
         ans="B",
         exp_en="A subset selected for study from a larger group is a sample.",
         exp_ar="مجموعة فرعية مختارة من مجموعة أكبر للدراسة هى عينة.")

L.quiz_q(qid="M02-L02-Q3", qtype="MCQ",
         q_en="Which column is the most likely Target in a churn-prediction project?",
         q_ar="أى عمود هو الأرجح ليكون 'الهدف' فى مشروع توقع مغادرة العملاء؟",
         opts_en=["Customer_ID","Signup_Date","Plan_Type","Did_Churn (Yes/No)"],
         opts_ar=["معرّف العميل","تاريخ التسجيل","نوع الباقة","هل غادر (نعم/لا)"],
         ans="D",
         exp_en="The target is what you want to predict — here, churn itself.",
         exp_ar="الهدف هو ما تريد التنبؤ به — هنا، المغادرة نفسها.")

L.quiz_q(qid="M02-L02-Q4", qtype="TrueFalse",
         q_en="True or False: A dataset and a population are always the same thing.",
         q_ar="صح أم خطأ: مجموعة البيانات والمجتمع شىء واحد دائماً.",
         opts_en=["True","False"], opts_ar=["صح","خطأ"], ans="B",
         exp_en="A dataset is usually a sample of the population, not the whole population.",
         exp_ar="مجموعة البيانات غالباً عينة من المجتمع، لا المجتمع كله.")

L.quiz_q(qid="M02-L02-Q5", qtype="MCQ",
         q_en="A unique ID column should:",
         q_ar="عمود المعرف الفريد يجب أن:",
         opts_en=["Contain duplicates for backup",
                  "Have one unique value per row, never repeating",
                  "Always be a name",
                  "Be optional in business datasets"],
         opts_ar=["يحتوى على مكررات للنسخ الاحتياطى",
                  "يحوى قيمة فريدة لكل صف، لا تتكرر",
                  "يكون اسماً دائماً",
                  "اختيارى فى بيانات الأعمال"],
         ans="B",
         exp_en="By definition, a unique ID is unique per row. Otherwise it isn't doing its job.",
         exp_ar="بحكم تعريفه، المعرف الفريد لا يتكرر. وإلا لا يؤدى دوره.")
L.render_quiz(); L.save()

print("L02-01, L02-02 done.")
