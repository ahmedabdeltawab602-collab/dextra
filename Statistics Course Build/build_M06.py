import sys, os, json
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY)

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 06 - Conclusion"
os.makedirs(OUT, exist_ok=True)

# ───────── Cover ─────────
d = setup(); fc(d)
titlep(d, module_no=6,
       module_en="Conclusion",
       module_ar="الخلاصة",
       subtitle_en="Course Summary · The Statistical Toolbox · What's Next",
       subtitle_ar="ملخص الدورة · صندوق الأدوات الإحصائية · ما التالى")
end_m(d, os.path.join(OUT, "M06 - Cover.docx"))

# ============================================================
# L06-01 — Course Summary & The Statistical Toolbox
# ============================================================
L = Lesson(
    lid="M06-L01", module=6,
    title_en="Course Summary and the Statistical Toolbox",
    title_ar="ملخص الدورة وصندوق الأدوات الإحصائية",
    objectives_en=[
        "Review the full arc of the course in one lesson.",
        "Match each business question to the right tool from the toolbox.",
        "Take an honest look at the frequentist approach's strengths and weaknesses.",
        "Know your next steps after this course.",
    ],
    objectives_ar=[
        "مراجعة الدورة كاملة فى درس واحد.",
        "ربط كل سؤال أعمال بالأداة المناسبة من الصندوق.",
        "نظرة أمينة على نقاط قوة وضعف المنهج التكرارى.",
        "معرفة الخطوات التالية بعد الدورة.",
    ],
    sources=["06 Conclusion/Course Summary.mp4",
             "Slides PDF: page 80 (Chapter Review)"], out_dir=OUT)
L.header(); L.objectives()

# §1 — The arc of the course
def s1(d):
    ep(d, "You have just travelled the full statistical landscape. Take a moment to see how far you've come.")
    ap(d, "لقد جلت للتو المشهد الإحصائى كاملاً. خذ لحظة لترى كم قطعت.")
    kvtable(d, header=["Module / الموديول","What you can now do / ما يمكنك فعله الآن"],
        rows=[
            ("01 — Introduction\nالمقدمة",
             "Speak the language of statistics — describe vs infer.\nالتحدث بلغة الإحصاء — وصف مقابل استدلال."),
            ("02 — Statistical Landscape\nالمشهد الإحصائى",
             "Map any business question to the right family of tools.\nربط أى سؤال أعمال بعائلة الأدوات الصحيحة."),
            ("03 — Descriptive Statistics\nالإحصاء الوصفى",
             "Summarise any dataset with centre, spread, and shape; explore relationships with correlation and regression.\nتلخيص أى بيانات بالمركز والتشتت والشكل؛ استكشاف العلاقات بالارتباط والانحدار."),
            ("04 — Inferential Statistics\nالاستدلالى",
             "Move from sample to population using p-values, alpha, CLT, LLN; size samples; control bias.\nالانتقال من العينة للمجتمع بقيم p، ألفا، CLT، LLN؛ تحديد العينات؛ التحكم فى التحيز."),
            ("05 — Hypothesis Testing\nاختبار الفرضيات",
             "Run formal A/B tests start-to-finish: state, set, select, compute, decide. Report responsibly.\nتشغيل اختبارات A/B رسمية من البداية للنهاية: صِغ، حدّد، اختر، احسب، قرّر. أبلغ بمسؤولية."),
        ])
L.section("intro", "1 · The Arc You've Travelled",
          "1 · الرحلة التى قطعتها", s1)

# §2 — The Statistical Toolbox decision flow
def s2(d):
    ep(d, "When facing a new business problem, walk this decision tree:")
    ap(d, "أمام أى مشكلة أعمال جديدة، اتبع شجرة القرار هذه:")
    en_(d, "Step A: Do I understand my data yet? If NO → start with descriptive stats and visualisation (Module 03).")
    en_(d, "Step B: Am I just describing what is, or making a decision? If just describing → KPIs, dashboards, summary report.")
    en_(d, "Step C: If making ONE BIG decision → inferential statistics + hypothesis test (Modules 04–05).")
    en_(d, "Step D: If making MANY automated decisions → machine learning.")
    an_(d, "الخطوة A: هل أفهم بياناتى؟ إن لا → ابدأ بالوصفى والتصور (الموديول 03).")
    an_(d, "الخطوة B: هل أصف فقط أم أتخذ قراراً؟ إن وصف فقط → مؤشرات، لوحات، تقرير.")
    an_(d, "الخطوة C: إن قرار كبير واحد → استدلالى + اختبار فرضيات (الموديولان 04 و 05).")
    an_(d, "الخطوة D: إن قرارات آلية كثيرة → تعلم آلى.")
L.section("concept", "2 · The Statistical Toolbox Decision Flow",
          "2 · مسار قرار صندوق الأدوات", s2)

# §3 — The toolbox by business question
def s3(d):
    kvtable(d, header=["Business question / سؤال أعمال","Tool / الأداة","Module / الموديول"],
        rows=[
            ("What is our average revenue?\nما متوسط إيراداتنا؟",
             "Mean (descriptive).\nالمتوسط (وصفى).","M03-L02"),
            ("How risky is this product line?\nكم مخاطرة هذا الخط؟",
             "Standard deviation, IQR.\nالانحراف، الربيعى.","M03-L03/L04"),
            ("What share of customers fall into each segment?\nما حصة كل قطاع؟",
             "Frequency distribution / histogram.\nالتوزيع التكرارى.","M03-L05/L07"),
            ("Are ad spend and sales related?\nهل ينفع الإعلان المبيعات؟",
             "Correlation; regression.\nالارتباط؛ الانحدار.","M03-L10/L11"),
            ("Did the new policy actually raise NPS?\nهل رفعت السياسة الجديدة NPS؟",
             "Hypothesis test (two-sample).\nاختبار فرضيات لعينتَين.","M05-L06"),
            ("Is this drug better than placebo?\nهل الدواء أفضل من الوهمى؟",
             "One-tailed hypothesis test.\nاختبار ذيل واحد.","M05-L07/L08"),
            ("How many to survey for ±3% precision?\nكم نستطلع لدقة ±3%؟",
             "Sample-size formula.\nصيغة حجم العينة.","M04-L07"),
            ("Can I trust this small sample?\nهل أثق فى هذه العينة الصغيرة؟",
             "CLT + standard error.\nالنهاية المركزية + الخطأ المعيارى.","M04-L09"),
            ("Did we miss a real effect?\nهل فاتنا تأثير حقيقى؟",
             "Power analysis (Type II error).\nتحليل القوة.","M05-L09"),
        ])
L.section("example", "3 · The Toolbox by Business Question",
          "3 · صندوق الأدوات حسب السؤال", s3)

# §4 — Strengths of the frequentist approach
def s4(d):
    h3(d, "Strengths", "نقاط القوة")
    eb(d, "Objective: relies only on the observed data, not on subjective priors.")
    eb(d, "Well-understood: a century of theory and practice behind it.")
    eb(d, "Built into every major statistical tool (Excel, Python, R, SAS).")
    eb(d, "Excellent fit for controlled experiments with clear design.")
    ab(d, "موضوعى: يعتمد فقط على البيانات المرصودة، لا على معرفة مسبقة ذاتية.")
    ab(d, "مفهوم جيداً: قرن من النظرية والتطبيق.")
    ab(d, "مدمج فى كل أداة إحصائية رئيسية (Excel، Python، R، SAS).")
    ab(d, "مناسب جداً للتجارب المضبوطة بتصميم واضح.")
    h3(d, "Weaknesses", "نقاط الضعف")
    eb(d, "Hungry for data — small samples often give inconclusive results.")
    eb(d, "Relies on tightly controlled experimental conditions.")
    eb(d, "Language is not intuitive (p-value is a constant source of confusion).")
    eb(d, "Doesn't easily incorporate prior knowledge.")
    eb(d, "Same data + different framing can yield different decisions across analysts.")
    ab(d, "نهم للبيانات — عينات صغيرة غالباً تعطى نتائج غير حاسمة.")
    ab(d, "يتطلب تجارب مضبوطة بعناية.")
    ab(d, "اللغة غير بديهية (قيمة p مصدر مستمر للحيرة).")
    ab(d, "لا يدمج المعرفة المسبقة بسهولة.")
    ab(d, "نفس البيانات + صياغة مختلفة قد تعطى قرارات مختلفة عبر المحللين.")
L.section("interpret", "4 · Honest Strengths and Weaknesses of the Frequentist Approach",
          "4 · نقاط قوة وضعف المنهج التكرارى", s4)

# §5 — The Bayesian alternative & where to learn more
def s5(d):
    ep(d, "An honest course mentions what it leaves out. Two big alternatives or extensions exist:")
    ap(d, "دورة أمينة تذكر ما لا تغطيه. توجد بديلان كبيران:")
    kvtable(d, header=["Approach / المنهج","What it adds / ما يضيفه","When to use / متى تستخدمه"],
        rows=[
            ("Bayesian statistics\nالإحصاء البيزى",
             "Incorporates prior knowledge; gives probability of hypotheses directly.\nيدمج المعرفة المسبقة؛ يعطى احتمال الفرضيات مباشرة.",
             "Small samples; rich prior info; sequential decisions.\nعينات صغيرة؛ معرفة مسبقة غنية؛ قرارات متتابعة."),
            ("Machine learning\nالتعلم الآلى",
             "Pattern detection at scale; automated predictions.\nاكتشاف أنماط على نطاق واسع؛ تنبؤات آلية.",
             "Many automated decisions; complex non-linear data.\nقرارات آلية كثيرة؛ بيانات معقدة غير خطية."),
            ("Non-parametric tests\nاختبارات لاباراميترية",
             "No normality assumption; rank-based.\nلا تفترض الطبيعى؛ مبنية على الرتب.",
             "Very small or very skewed samples.\nعينات صغيرة جداً أو منحرفة بشدة."),
        ])
    callout(d, "tip",
            "Suggested next steps: (1) a course on probability distributions and the t/F/chi-square tests; (2) a basics-of-Python or basics-of-R course to scale these techniques; (3) Bayesian Methods for Hackers (free online) as a friendly entry to Bayesian thinking.",
            "خطوات تالية مقترحة: (1) دورة فى التوزيعات الاحتمالية واختبارات t/F/chi-square؛ (2) أساسيات Python أو R لتوسيع التطبيق؛ (3) Bayesian Methods for Hackers (مجانى على الإنترنت) كمدخل سهل للتفكير البيزى.")
L.section("practice", "5 · What This Course Did NOT Cover (and Where to Go Next)",
          "5 · ما لم تغطه الدورة (والوجهة التالية)", s5)

# §6 — A philosophy for the working analyst
def s6(d):
    h3(d, "Five habits that will outlast any specific test", "خمس عادات تبقى بعد أى اختبار محدد")
    en_(d, "ALWAYS look at the data before computing anything. Histograms, scatter plots, box plots.")
    en_(d, "Be specific about WHAT you're claiming and WHO the population is.")
    en_(d, "Pre-register your hypothesis, alpha, and test type before seeing the data.")
    en_(d, "Report effect size + uncertainty alongside p-values. NEVER p alone.")
    en_(d, "Translate the result into business language for the decision-maker. Statistics serves the decision, not the other way around.")
    an_(d, "انظر دائماً للبيانات قبل أى حساب. رسوم تكرار، تشتت، صندوقية.")
    an_(d, "كن محدداً فيما تدّعى ومن هو المجتمع.")
    an_(d, "سجّل فرضيتك وألفا ونوع الاختبار قبل البيانات.")
    an_(d, "اذكر حجم التأثير + عدم اليقين مع قيمة p. لا p وحدها أبداً.")
    an_(d, "ترجم النتيجة للغة أعمال لصانع القرار. الإحصاء يخدم القرار، لا العكس.")
L.section("traps", "6 · A Working Analyst's Philosophy",
          "6 · فلسفة المحلل العامل", s6)

# Memory aids
L.add_memory_aid("Describe → Infer → Decide. That's all statistics, in three words.",
                 "صف ← استدل ← قرّر. هذا كل الإحصاء فى ثلاث كلمات.")
L.add_memory_aid("The right tool depends on the question — not on what you happen to know.",
                 "الأداة الصحيحة تعتمد على السؤال — لا على ما تعرفه صدفة.")
L.add_memory_aid("p-value never proves anything. It just measures how surprised H0 should be.",
                 "قيمة p لا تثبت شيئاً. تقيس فقط كم يجب أن تتفاجأ H0.")
L.add_memory_aid("Effect size matters more than p. Always report both.",
                 "حجم التأثير أهم من p. اذكرهما معاً دائماً.")
L.add_memory_aid("Random > big. A small random sample beats a huge biased one.",
                 "العشوائية > الحجم. عينة صغيرة عشوائية تفوز على ضخمة متحيزة.")

# Summary
L.summary(
    en="You've walked the full statistical landscape: from describing data, to inferring from samples, to running formal hypothesis tests. Match the question to the tool: descriptive for KPIs, inferential for single big decisions, machine learning for automated decisions at scale. The frequentist approach you mastered is the industry default — objective, well-supported, hungry for data. It is not the only approach (Bayesian and non-parametric methods extend it), but it is the foundation everything else builds on. The hardest part of becoming a real analyst isn't the math — it's the discipline of pre-registering, reporting effect size, and resisting the urge to manipulate data until it says what you want.",
    ar="لقد جلت المشهد الإحصائى كاملاً: من وصف البيانات، إلى الاستدلال من العينات، إلى تشغيل اختبارات فرضيات رسمية. اربط السؤال بالأداة: وصفى للمؤشرات، استدلالى للقرارات الكبيرة الفردية، تعلم آلى للقرارات الآلية على نطاق واسع. المنهج التكرارى الذى أتقنته هو افتراضى الصناعة — موضوعى، مدعوم بقوة، نهم للبيانات. ليس المنهج الوحيد (البيزى واللاباراميترى يمدّانه)، لكنه الأساس الذى يبنى عليه كل ما عداه. أصعب جزء فى أن تصبح محللاً حقيقياً ليس الرياضيات — بل انضباط التسجيل المسبق، وذكر حجم التأثير، ومقاومة إغراء التلاعب بالبيانات."
)
L.render_memory()

# Closing message
def s_close(d):
    callout(d, "key",
            "Congratulations. The 41 lessons you've completed are the same foundation that supports a multi-trillion-dollar industry of A/B testing, clinical trials, quality control, polling, and modern data science. The math hasn't changed in a century, but the questions you can now answer have.",
            "تهانينا. الـ 41 درساً التى أكملتها هى نفس الأساس الذى تقوم عليه صناعة بتريليونات الدولارات من اختبارات A/B والتجارب السريرية وضبط الجودة والاستطلاعات وعلم البيانات الحديث. الرياضيات لم تتغير منذ قرن، لكن الأسئلة التى يمكنك إجابتها الآن قد تغيّرت.")
L.section("summary", "7 · A Final Word",
          "7 · كلمة أخيرة", s_close)

# Comprehensive review quiz (5 questions touching multiple modules)
L.quiz_q(qid="M06-L01-Q1", qtype="MCQ",
         q_en="Which is descriptive, NOT inferential?",
         q_ar="أى مما يلى وصفى، لا استدلالى؟",
         opts_en=["Computing last month's average revenue from full records",
                  "Estimating population mean from a sample",
                  "Running an A/B test","Computing a p-value"],
         opts_ar=["حساب متوسط إيراد الشهر الماضى من السجلات الكاملة",
                  "تقدير متوسط المجتمع من عينة","تشغيل اختبار A/B","حساب قيمة p"],
         ans="A", exp_en="Computing a value from full data on hand is descriptive.",
         exp_ar="حساب قيمة من البيانات الموجودة وصفى.")

L.quiz_q(qid="M06-L01-Q2", qtype="MCQ",
         q_en="A right-skewed business metric (like revenue per customer) is best summarised with:",
         q_ar="مقياس أعمال منحرف يميناً (مثل الإيراد لكل عميل) يلخّص أفضل بـ:",
         opts_en=["Mean only","Median (and mention mean for context)",
                  "Range only","Standard deviation only"],
         opts_ar=["المتوسط فقط","الوسيط (مع ذكر المتوسط)","المدى فقط","الانحراف المعيارى فقط"],
         ans="B", exp_en="Median resists the long right tail. Mean alone misleads.",
         exp_ar="الوسيط مقاوم للذيل الأيمن. المتوسط وحده مضلّل.")

L.quiz_q(qid="M06-L01-Q3", qtype="MCQ",
         q_en="Power of a test is increased most reliably by:",
         q_ar="قوة الاختبار تُزاد بشكل موثوق بـ:",
         opts_en=["Lowering alpha","Increasing sample size",
                  "Changing the alternative","Picking nicer colours"],
         opts_ar=["خفض ألفا","زيادة n","تغيير البديلة","ألوان أجمل"], ans="B",
         exp_en="Bigger n is the universal lever for power.",
         exp_ar="n أكبر هى الذراع العالمية للقوة.")

L.quiz_q(qid="M06-L01-Q4", qtype="MCQ",
         q_en="An A/B test on millions of users shows p = 0.0001 with a +0.05% conversion lift. The honest report says:",
         q_ar="اختبار A/B على ملايين المستخدمين p=0.0001 لكن +0.05% فقط. التقرير الأمين يقول:",
         opts_en=["Big effect, roll out","Statistically significant but practically tiny — decide based on cost/benefit",
                  "Re-run the test","Discard the result"],
         opts_ar=["تأثير كبير، أطلق","ذو دلالة إحصائية لكن ضئيل عملياً — قرّر بحسب التكلفة/الفائدة",
                  "أعد التشغيل","تجاهل"], ans="B",
         exp_en="Statistical significance ≠ business value.", exp_ar="الدلالة الإحصائية ≠ القيمة التجارية.")

L.quiz_q(qid="M06-L01-Q5", qtype="MCQ",
         q_en="A non-statistical decision-maker asks: 'Should we launch?' Your job as analyst is to:",
         q_ar="صانع قرار غير متخصص يسأل: 'هل نطلق؟' دورك كمحلل:",
         opts_en=["Hand them a p-value and walk away",
                  "Translate the test result into a clear business recommendation with uncertainty",
                  "Refuse to answer until they learn statistics",
                  "Always recommend launch"],
         opts_ar=["أعطه قيمة p وامش","ترجم النتيجة لتوصية أعمال واضحة مع عدم اليقين",
                  "ارفض حتى يتعلم الإحصاء","انصح بالإطلاق دائماً"], ans="B",
         exp_en="Statistics serves business decisions, not the other way around.",
         exp_ar="الإحصاء يخدم قرارات الأعمال، لا العكس.")
L.render_quiz()
L.save()

print("L06-01 done.")

# Module 06 index
module_index = {
    "id":"M06","module":6,"type":"module",
    "title_en":"Conclusion","title_ar":"الخلاصة",
    "lessons":[
        {"id":"M06-L01","title_en":"Course Summary and the Statistical Toolbox","title_ar":"ملخص الدورة وصندوق الأدوات"},
    ],
    "sources":["06 Conclusion/Course Summary.mp4","Slides PDF page 80"]
}
with open(os.path.join(OUT,"M06 - Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(module_index, f, ensure_ascii=False, indent=2)
print("Module 06 index written.")
