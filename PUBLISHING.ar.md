# دليل نشر مكتبة dextra على GitHub وPyPI

هذا الدليل باللغة العربية يشرح كل خطوة لنشر المشروع بدءاً من مجلد المشروع المحلّي ووصولاً إلى `pip install dextra` للعموم.

---

## 1) المتطلبات الأولية

ثبّت على جهازك ما يلي إذا لم يكن موجوداً:

- Python 3.9 أو أحدث
- Git
- حساب GitHub
- (اختياري للنشر على PyPI) حساب على https://pypi.org وحساب على https://test.pypi.org

---

## 2) التجهيز محلياً

ادخل إلى مجلّد المشروع في الطرفية، ثم نفّذ:

```bash
cd dextra-project

# إنشاء بيئة افتراضية (اختياري لكن مستحسن)
python -m venv .venv
source .venv/bin/activate          # على Windows: .venv\Scripts\activate

# تثبيت المكتبة مع أدوات التطوير
pip install -e ".[dev]"

# تشغيل الاختبارات للتأكد من سلامة كل شيء
pytest

# فحص الأسلوب
ruff check .
```

إذا مرَّت الاختبارات كلها بنجاح، فأنت جاهز للنشر.

---

## 3) تهيئة Git وأول commit

```bash
git init
git add .
git commit -m "initial commit: dextra 0.1.0"
```

---

## 4) إنشاء مستودع على GitHub ورفع الكود

### الطريقة الأولى: عبر الموقع ثم الطرفية

1. افتح https://github.com/new
2. اسم المستودع: `dextra`
3. اجعله **Public** (لأن PyPI لا يَقبل مستودعات خاصة كمصدر للعرض العام).
4. لا تضف README أو .gitignore من GitHub — لدينا نسخ جاهزة.
5. اضغط **Create repository**.

ثم في الطرفية:

```bash
git branch -M main
git remote add origin https://github.com/ahmedabdeltawab602-collab/dextra.git
git push -u origin main
```

### الطريقة الثانية: عبر GitHub CLI (`gh`)

```bash
gh repo create dextra --public --source=. --remote=origin --push
```

---

## 5) التحقق من أن المشروع يعمل على GitHub

ادخل صفحة المستودع. بفضل ملف `.github/workflows/tests.yml`، ستبدأ GitHub Actions تلقائياً بتشغيل الاختبارات على Python 3.9، 3.10، 3.11، 3.12 — تابع شارة **tests** في README.

---

## 6) (اختياري) النشر على TestPyPI أولاً

قبل النشر على PyPI الحقيقي، استخدم TestPyPI لضمان عدم وجود أخطاء.

### أ) إنشاء حساب وتوكن

1. سجّل في https://test.pypi.org/account/register/
2. من الإعدادات → API tokens، أنشئ توكناً باسم `dextra-test` وصلاحية *الحساب كله* (أوّل مرة فقط، لاحقاً ضيّقها للمكتبة).

### ب) بناء الملفات

```bash
pip install --upgrade build twine
python -m build
```

هذا يُنتج ملفين في مجلد `dist/`:
- `dextra-0.1.0.tar.gz` (sdist)
- `dextra-0.1.0-py3-none-any.whl` (wheel)

### ج) الرفع إلى TestPyPI

```bash
# استخدم glob الإصدار الصريح دائماً (استبدل X.Y.Z برقم الإصدار الفعلي).
# لا تستخدم dist/* العام أبداً: قد يرفع أرتيفاكتات قديمة متبقية في dist/.
twine upload --repository testpypi dist/dextra-X.Y.Z*
```

- **Username:** `__token__`
- **Password:** التوكن كاملاً (يبدأ بـ `pypi-`).

### د) التجربة

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dextra
python -c "import dextra; print(dextra.__version__)"
```

---

## 7) النشر على PyPI الحقيقي

### أ) توكن على pypi.org

من https://pypi.org/manage/account/token/ أنشئ توكناً (نفس خطوات TestPyPI).

### ب) حفظ التوكن (اختياري)

لتسهيل الأمر، أنشئ ملف `~/.pypirc` بالشكل:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEI...REAL_TOKEN...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEI...TEST_TOKEN...
```

ثم:

```bash
chmod 600 ~/.pypirc
```

### ج) الرفع

```bash
# تنظيف آثار البناء السابق
rm -rf dist/ build/ *.egg-info

# إعادة البناء
python -m build

# الرفع إلى PyPI — glob الإصدار الصريح دائماً، لا dist/* العام (قد يرفع أرتيفاكتات قديمة)
twine upload dist/dextra-X.Y.Z*
```

### د) التحقق

```bash
pip install dextra
python -c "import dextra as dx; print(dx.__version__)"
```

---

## 8) الإصدارات اللاحقة

عند أي تحديث:

1. حدِّث `__version__` في `src/dextra/_version.py` (مثلاً من `0.1.0` إلى `0.1.1`).
2. حدِّث `CHANGELOG.md` بملخّص التغييرات.
3. نفّذ الاختبارات: `pytest`.
4. أنشئ commit وtag:

```bash
git add .
git commit -m "release 0.1.1: <summary>"
git tag v0.1.1
git push origin main --tags
```

5. على GitHub: **Releases → Draft a new release → Choose tag v0.1.1 → Publish release**.
6. بفضل ملف `.github/workflows/publish.yml`، سيتم بناء المكتبة ورفعها إلى PyPI تلقائياً.

> ملاحظة: لتفعيل النشر التلقائي أول مرة، يجب إعداد **Trusted Publishing** على PyPI:  
> https://pypi.org/manage/account/publishing/ → Add a new pending publisher — ربط اسم المستودع واسم Workflow `publish.yml` وبيئة `pypi`.

---

## 9) قائمة فحص نهائية قبل الإصدار

- [ ] حدَّثت `_version.py` و `CHANGELOG.md`.
- [ ] `pytest` تنجح محلياً.
- [ ] `ruff check .` بدون أخطاء.
- [ ] `python -m build` تنشئ `dist/*.tar.gz` و `dist/*.whl` بدون أخطاء.
- [ ] `twine check dist/*` تُرجع `PASSED`.
- [ ] جرّبت `pip install` من TestPyPI.
- [ ] رفعت التاج إلى GitHub.

---

## 10) نصائح مفيدة

**الترخيص:** تم اختيار MIT. إذا غيّرته لاحقاً، حدّث:
- ملف `LICENSE`.
- حقل `license` في `pyproject.toml`.
- مصنّف `License ::` داخل `classifiers`.

**الأمان:** لا تضع التوكنات في الكود أبداً. استخدم `~/.pypirc` أو GitHub Secrets.

**الأسماء:** إذا كان اسم `dextra` محجوزاً على PyPI، ستحتاج لاسم بديل. غيّر `name` في `pyproject.toml` (مثلاً `dextra-tools`) ثم أعد النشر.

**التوثيق:** فكّر في إضافة موقع بسيط عبر [mkdocs](https://www.mkdocs.org/) أو [Sphinx](https://www.sphinx-doc.org/) عندما يكبر المشروع.
