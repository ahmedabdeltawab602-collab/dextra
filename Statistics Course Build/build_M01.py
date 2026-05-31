import sys, os
sys.path.insert(0, "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build")
from lessonkit import Lesson
from helpers import (h2, h3, ep, ap, el, eb, ab, en_, an_, eq, div, callout, kvtable, pgbrk,
                     setup, titlep, fc, end_m, PRIMARY, SECONDARY, RED, YELLOW, GREEN, ORANGE)
import json

OUT = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build/Module 01 - Introduction"
os.makedirs(OUT, exist_ok=True)

# ───────── Cover page (module title) ─────────
d = setup(); fc(d)
titlep(d, module_no=1,
       module_en="Introduction",
       module_ar="المقدمة",
       subtitle_en="The Course at a Glance · Your Roadmap to Statistics",
       subtitle_ar="نظرة عامة على الدورة · خريطة الطريق إلى علم الإحصاء")
end_m(d, os.path.join(OUT, "M01 — Cover.docx"))

# ───────── L01-01: Course Introduction & Roadmap ─────────
L = Lesson(
    lid="M01-L01", module=1,
    title_en="Course Introduction & Roadmap",
    title_ar="مقدمة الدورة وخريطة الطريق",
    objectives_en=[
        "Understand what Statistics actually is — and what it's not.",
        "Recognise how Statistics powers everyday business decisions.",
        "See the journey: from describing data to testing hypotheses.",
        "Get familiar with the six modules ahead of you.",
    ],
    objectives_ar=[
        "أن تفهم ما هو الإحصاء حقاً — وما ليس كذلك.",
        "أن تدرك كيف يقود الإحصاء قرارات الأعمال اليومية.",
        "أن ترى الرحلة: من وصف البيانات إلى اختبار الفرضيات.",
        "أن تتعرف على الموديولات الستة القادمة.",
    ],
    sources=[
        "Folder: 01 Introduction/Course Introduction.mp4",
        "Slides PDF: pages 1–4",
        "Translation: ترجمة الشريحة.docx pages 1–4",
        "OpenStax: Chapter 1.1 — Definitions of Statistics",
    ],
    out_dir=OUT
)
L.header()
L.objectives()

# §1 — Intro
def s1(d):
    ep(d, "Statistics is the science of collecting, summarising, analysing, and interpreting data — so that you can make decisions or draw inferences with confidence, even when the truth is hidden in noise.")
    ap(d, "الإحصاء هو علم جمع البيانات وتلخيصها وتحليلها وتفسيرها — حتى تتمكن من اتخاذ القرارات أو استخلاص الاستدلالات بثقة، حتى عندما تكون الحقيقة مختبئة وسط الضوضاء.")
    ep(d, "Think of it as a pair of glasses. Without them, raw data looks like a blurry crowd. With them, patterns, averages, and surprises snap into focus.")
    ap(d, "تخيل أن الإحصاء نظارة. بدونها، تبدو البيانات الخام كحشد ضبابى. ومعها، تتضح الأنماط والمتوسطات والمفاجآت.")
    callout(d, "key",
            "Statistics turns data into decisions. Every dashboard, every A/B test, every forecast you've ever seen is statistics in action.",
            "الإحصاء يحوّل البيانات إلى قرارات. كل لوحة بيانات وكل اختبار A/B وكل توقّع رأيته فى حياتك هو إحصاء فى صورته العملية.")
L.section("intro", "1 · What Is Statistics?", "1 · ما هو الإحصاء؟", s1)

# §2 — Three branches
def s2(d):
    ep(d, "The course is built around three foundational ideas you will master step by step:")
    ap(d, "الدورة مبنية حول ثلاث أفكار أساسية ستتقنها خطوة بخطوة:")
    kvtable(d,
        header=["Branch / الفرع", "Question it answers / السؤال الذى يجيب عنه", "Tool example / مثال على الأداة"],
        rows=[
            ("Descriptive Statistics\n(الإحصاء الوصفى)",
             "What does the data look like?\nكيف تبدو البيانات؟",
             "Mean, Median, Std Dev\nالمتوسط، الوسيط، الانحراف المعيارى"),
            ("Inferential Statistics\n(الإحصاء الاستدلالى)",
             "What can we say about the whole from a sample?\nماذا نستنتج عن الكل من جزء؟",
             "Confidence intervals, p-values\nفترات الثقة، قيم p"),
            ("Hypothesis Testing\n(اختبار الفرضيات)",
             "Is the change real, or just noise?\nهل الفرق حقيقى أم مجرد ضوضاء؟",
             "Two-tail / one-tail tests\nاختبارات ثنائية وأحادية الذيل"),
        ])
L.section("definitions", "2 · The Three Big Branches", "2 · الفروع الثلاثة الكبرى", s2)

# §3 — Why every analyst needs it
def s3(d):
    ep(d, "Whatever role you target — data analyst, BI developer, marketer, financial analyst — Statistics is your common language. It tells you when a result is real and when it's just luck.")
    ap(d, "أياً كان الدور الذى تستهدفه — محلل بيانات، مطور BI، مسوّق، محلل مالى — فالإحصاء هو لغتك المشتركة. هو الذى يخبرك متى تكون النتيجة حقيقية ومتى تكون مجرد حظ.")
    h3(d, "Real business questions Statistics answers", "أسئلة عمل حقيقية يجيب عنها الإحصاء")
    eb(d, "Did our new homepage really lift conversions, or did Tuesday's traffic just behave differently?")
    eb(d, "How many customers must we survey to estimate the true Net Promoter Score within ±3 points?")
    eb(d, "Are sales of Product A and Product B genuinely different, or is the gap within normal week-to-week variance?")
    eb(d, "Is the new price point cannibalising loyal customers, or only repelling bargain hunters?")
    ab(d, "هل أدّت صفحتنا الرئيسية الجديدة فعلاً إلى رفع معدلات التحويل، أم أن زوّار الثلاثاء تصرفوا بشكل مختلف فحسب؟")
    ab(d, "كم عميلاً يجب أن نستطلع رأيه لتقدير معدل NPS الحقيقى بدقة ±3 نقاط؟")
    ab(d, "هل مبيعات المنتجَين A وB مختلفة فعلاً، أم أن الفجوة ضمن التباين الأسبوعى الطبيعى؟")
    ab(d, "هل السعر الجديد يهرّب العملاء الأوفياء، أم يصدّ الباحثين عن العروض فقط؟")
L.section("concept", "3 · Why Statistics Matters in Data Analysis",
          "3 · لماذا يهمّ الإحصاء فى تحليل البيانات", s3)

# §4 — Course roadmap
def s4(d):
    ep(d, "Here is the journey we'll take together, six modules deep:")
    ap(d, "هذه هى الرحلة التى سنخوضها معاً عبر ستة موديولات:")
    kvtable(d,
        header=["#", "Module / الموديول", "What you'll learn / ماذا ستتعلم"],
        rows=[
            ("01","Introduction\nالمقدمة","Big picture · the language of data\nالصورة الكبيرة · لغة البيانات"),
            ("02","The Statistical Landscape\nالمشهد الإحصائى","Population vs sample · descriptive vs inferential\nالمجتمع والعينة · الوصفى والاستدلالى"),
            ("03","Descriptive Statistics\nالإحصاء الوصفى","Mean, median, std, normal, correlation, regression\nالمتوسط، الوسيط، الانحراف، الطبيعى، الارتباط، الانحدار"),
            ("04","Inferential Statistics\nالإحصاء الاستدلالى","p-values, alpha, CLT, LLN, bias control\nقيم p، ألفا، CLT، LLN، التحيز"),
            ("05","Hypothesis Testing\nاختبار الفرضيات","5 steps · one-tail vs two-tail · Type I/II errors\n٥ خطوات · أحادى/ثنائى الذيل · أخطاء النوعَين"),
            ("06","Conclusion\nالخلاصة","The Statistical Toolbox · putting it all together\nصندوق الأدوات · جمع كل ما تعلّمت"),
        ])
    callout(d, "tip",
            "Each lesson follows the same rhythm: definition → formula → worked example → interpretation → practical use → quiz. Once you internalise the rhythm, the math becomes the easy part.",
            "كل درس يتبع نفس الإيقاع: تعريف ← صيغة ← مثال محلول ← تفسير ← استخدام عملى ← كويز. بمجرد أن تستوعب الإيقاع، تصبح الرياضيات هى الجزء السهل.")
L.section("practice", "4 · The Roadmap Ahead", "4 · خريطة الطريق", s4)

# §5 — How to study
def s5(d):
    ep(d, "Four habits that will make this course stick:")
    ap(d, "أربع عادات ستجعل هذه الدورة ترسخ فى ذهنك:")
    en_(d, "Read the lesson once for the big picture, then re-read the example slowly with a calculator.")
    en_(d, "After each lesson, take the 5-question quiz before peeking at the answers — even if you fail one, the explanation cements the idea.")
    en_(d, "Re-state every formula in plain English (and Arabic). If you can teach it to a friend, you own it.")
    en_(d, "Apply each concept to your own data — a sales log, a customer survey, even your monthly spending.")
    an_(d, "اقرأ الدرس مرة للحصول على الصورة الكاملة، ثم أعد قراءة المثال ببطء مع آلة حاسبة.")
    an_(d, "بعد كل درس، حل كويز الخمسة أسئلة قبل أن ترى الإجابات — حتى لو أخطأت فى واحد، فالشرح يثبّت الفكرة.")
    an_(d, "أعد صياغة كل قانون بلغة بسيطة (إنجليزية وعربية). إن استطعت أن تشرحه لصديق، فأنت تملكه.")
    an_(d, "طبّق كل مفهوم على بياناتك الخاصة — سجل مبيعات، استطلاع عملاء، حتى مصروفاتك الشهرية.")
L.section("traps", "5 · How to Get the Most Out of This Course",
          "5 · كيف تستفيد أقصى استفادة من الدورة", s5)

# Memory aid
L.add_memory_aid(
    "Statistics = Description + Inference. Description tells you what is. Inference tells you what probably is true beyond what you see.",
    "الإحصاء = وصف + استدلال. الوصف يخبرك بما هو موجود. الاستدلال يخبرك بما يُرجَّح أنه صحيح فيما لا تراه."
)
L.add_memory_aid(
    "Without statistics, every business decision is a guess wearing a confident smile.",
    "بدون الإحصاء، كل قرار عمل هو مجرد تخمين يرتدى ابتسامة واثقة."
)

# Summary
L.summary(
    en="Statistics is the bridge between raw data and confident decisions. In the six modules ahead, you'll move from describing what you see (mean, std, distribution shape) to inferring what you can't see (population behavior, true effects) to formally testing whether a change is real. Each lesson is bilingual, example-driven, and tied back to real business problems — customers, sales, products, prices.",
    ar="الإحصاء هو الجسر بين البيانات الخام والقرارات الواثقة. عبر الموديولات الستة القادمة، ستنتقل من وصف ما تراه (المتوسط، الانحراف، شكل التوزيع) إلى الاستدلال عمّا لا تراه (سلوك المجتمع، التأثيرات الحقيقية) إلى الاختبار الرسمى لما إذا كان أى تغيير حقيقياً. كل درس ثنائى اللغة، مدعوم بالأمثلة، ومربوط بمشاكل أعمال حقيقية — عملاء، مبيعات، منتجات، أسعار."
)
L.render_memory()

# Quiz — 5 questions
L.quiz_q(qid="M01-L01-Q1", qtype="MCQ",
         q_en="Statistics is best described as the science of:",
         q_ar="يُوصَف الإحصاء بأنه علم:",
         opts_en=[
             "Memorising large datasets.",
             "Collecting, summarising, analysing and interpreting data to make decisions.",
             "Building dashboards in Excel.",
             "Forecasting stock prices."
         ],
         opts_ar=[
             "حفظ مجموعات البيانات الكبيرة.",
             "جمع البيانات وتلخيصها وتحليلها وتفسيرها لاتخاذ القرارات.",
             "بناء لوحات بيانات فى Excel.",
             "التنبؤ بأسعار الأسهم."
         ],
         ans="B",
         exp_en="Statistics is end-to-end: from collecting raw data to interpreting it for decisions. Dashboards and forecasts are downstream applications.",
         exp_ar="الإحصاء عملية متكاملة من جمع البيانات الخام إلى تفسيرها لاتخاذ القرارات. أما لوحات البيانات والتوقعات فهى تطبيقات لاحقة عليه.")

L.quiz_q(qid="M01-L01-Q2", qtype="MCQ",
         q_en="Which question is INFERENTIAL rather than descriptive?",
         q_ar="أى سؤال من هذه يُعدّ استدلالياً وليس وصفياً؟",
         opts_en=[
             "What was our average order value last month?",
             "How many customers bought Product A in Q3?",
             "Does our new ad increase clicks compared to the old one?",
             "What is the median age of our subscribers?"
         ],
         opts_ar=[
             "ما متوسط قيمة الطلب الشهر الماضى؟",
             "كم عميلاً اشترى المنتج A فى الربع الثالث؟",
             "هل يزيد إعلاننا الجديد من النقرات مقارنة بالقديم؟",
             "ما الوسيط العمرى للمشتركين لدينا؟"
         ],
         ans="C",
         exp_en="A, B and D describe what already happened. C goes beyond the observed data to ask whether the ad has a real effect — that is inference.",
         exp_ar="الخيارات A و B و D تصف ما حدث بالفعل. أما C فيتجاوز البيانات المرصودة ليسأل: هل للإعلان تأثير حقيقى؟ هذا استدلال.")

L.quiz_q(qid="M01-L01-Q3", qtype="TrueFalse",
         q_en="True or False: A descriptive statistic such as the mean can be calculated from any sample, no matter how biased the sample is.",
         q_ar="صح أم خطأ: يمكن حساب الإحصاء الوصفى مثل المتوسط من أى عينة مهما كانت متحيزة.",
         opts_en=["True","False"],
         opts_ar=["صح","خطأ"],
         ans="A",
         exp_en="You can always *compute* the mean of any list of numbers. The catch is interpretation: a biased sample's mean does not represent the population. Statistics is computable; inference is what gets tricky.",
         exp_ar="يمكنك دائماً حساب المتوسط لأى قائمة من الأرقام. لكن الإشكال فى التفسير: متوسط عينة متحيزة لا يمثل المجتمع. الحساب ممكن، أما الاستدلال فهو الجزء الصعب.")

L.quiz_q(qid="M01-L01-Q4", qtype="MCQ",
         q_en="A marketing team launches an A/B test on a sales page. The most natural module to study FIRST in this course is:",
         q_ar="فريق تسويق يجرى اختبار A/B على صفحة مبيعات. الموديول الأنسب للبدء به فى هذه الدورة هو:",
         opts_en=[
             "Module 05 — Hypothesis Testing directly.",
             "Module 03 — Descriptive Statistics, to first understand the data.",
             "Module 06 — Conclusion.",
             "Skip the course; go to ChatGPT."
         ],
         opts_ar=[
             "الموديول الخامس — اختبار الفرضيات مباشرة.",
             "الموديول الثالث — الإحصاء الوصفى، لفهم البيانات أولاً.",
             "الموديول السادس — الخلاصة.",
             "تجاهل الدورة واسأل ChatGPT."
         ],
         ans="B",
         exp_en="You can't test hypotheses on data you don't understand. Start with descriptive statistics: distribution shape, central tendency, spread. Only then design a meaningful hypothesis test.",
         exp_ar="لا يمكنك اختبار فرضية على بيانات لا تفهمها. ابدأ بالإحصاء الوصفى: شكل التوزيع، النزعة المركزية، التشتت. ثم صمّم اختبار فرضية مفيد.")

L.quiz_q(qid="M01-L01-Q5", qtype="MCQ",
         q_en="Which statement BEST captures the spirit of inferential statistics?",
         q_ar="أى عبارة تعبّر أفضل عن روح الإحصاء الاستدلالى؟",
         opts_en=[
             "I will study every customer who ever existed.",
             "From a sample, I can make defensible claims about a population.",
             "I trust my gut more than the numbers.",
             "I never make decisions without 100% certainty."
         ],
         opts_ar=[
             "سأدرس كل عميل وُجد فى التاريخ.",
             "من عينة، يمكنى تقديم ادعاءات مدعومة حول المجتمع كله.",
             "أثق بحدسى أكثر من الأرقام.",
             "لا أتخذ قراراً دون يقين 100%."
         ],
         ans="B",
         exp_en="The whole point of inference: drawing supportable conclusions about a population from a sample, while quantifying the uncertainty.",
         exp_ar="جوهر الاستدلال: استخلاص استنتاجات مدعومة عن المجتمع من عينة، مع قياس عدم اليقين فى الادعاء.")

L.render_quiz()

dx, jx = L.save()
print("DOCX:", dx)
print("JSON:", jx)

# Save a minimal cover JSON too (module-level metadata)
cover_json = {
    "id":"M01","module":1,"type":"module","title_en":"Introduction","title_ar":"المقدمة",
    "lessons":[{"id":"M01-L01","title_en":"Course Introduction & Roadmap","title_ar":"مقدمة الدورة وخريطة الطريق"}],
    "sources":["01 Introduction/Course Introduction.mp4","Stats Fundamentals Course Slides.pdf pages 1–4"]
}
with open(os.path.join(OUT,"M01 — Module Index.json"),"w",encoding="utf-8") as f:
    json.dump(cover_json, f, ensure_ascii=False, indent=2)
print("INDEX: M01 — Module Index.json")
