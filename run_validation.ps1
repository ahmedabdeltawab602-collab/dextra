# ============================================================================
#  run_validation.ps1  --  dextra local validation gate (Phase 6.1)
#  Runs: lint (ruff) + full test suite (pytest + coverage), then a summary.
#  Writes a full log to validation_log.txt that you can paste back.
#
#  Usage (from anywhere):
#     powershell -ExecutionPolicy Bypass -File "D:\06 PythonProjects\dextra-project\run_validation.ps1"
#
#  Optional: pass  -Phase6Only  to run only the regress tests.
# ============================================================================
param(
    [switch]$Phase6Only
)

$ErrorActionPreference = "Continue"   # run every step; report at the end
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$project = "D:\06 PythonProjects\dextra-project"
$log     = Join-Path $project "validation_log.txt"
Set-Location $project

# Start a transcript so the whole run is captured to a file.
Start-Transcript -Path $log -Force | Out-Null

function Section($t) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "  $t" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
}

Section "0. Environment"
& ".\.venv\Scripts\Activate.ps1"
python --version
Write-Host "Ensuring dev + ml dependencies (ruff, pytest, scikit-learn)..." -ForegroundColor DarkGray
python -m pip install -e ".[dev,ml]" --quiet
$pyver = (python -c "import sys,pandas,numpy; print(sys.version.split()[0], '| pandas', pandas.__version__, '| numpy', numpy.__version__)")
try { $skl = (python -c "import sklearn; print(sklearn.__version__)") } catch { $skl = "MISSING" }
Write-Host "python $pyver | scikit-learn $skl"

# ---------------------------------------------------------------------------
Section "1. Lint  --  ruff auto-fix then strict check"
Write-Host "1a) ruff check . --fix  (auto-fixes import order, unused imports, f-strings)" -ForegroundColor DarkGray
ruff check . --fix
Write-Host "`n1b) ruff check .  (strict gate -- must be clean)" -ForegroundColor DarkGray
ruff check .
$lintCode = $LASTEXITCODE

# ---------------------------------------------------------------------------
Section "2. Tests  --  pytest + coverage"
if ($Phase6Only) {
    Write-Host "Running ONLY the Phase 6 regress suite..." -ForegroundColor DarkGray
    pytest tests/test_phase6_stage1.py -v --cov=dextra.modeling --cov-report=term-missing
} else {
    Write-Host "Running the FULL suite (Phases 1-6 + consolidation + legacy)..." -ForegroundColor DarkGray
    pytest --cov=dextra --cov-report=term-missing --cov-fail-under=60
}
$testCode = $LASTEXITCODE

# ---------------------------------------------------------------------------
Section "3. SUMMARY"
$lintOK = ($lintCode -eq 0)
$testOK = ($testCode -eq 0)
if ($lintOK) { Write-Host "  LINT  : PASS  (ruff clean)" -ForegroundColor Green }
else         { Write-Host "  LINT  : FAIL  (ruff exit $lintCode -- see section 1b above)" -ForegroundColor Red }
if ($testOK) { Write-Host "  TESTS : PASS" -ForegroundColor Green }
else         { Write-Host "  TESTS : FAIL  (pytest exit $testCode -- see section 2 above)" -ForegroundColor Red }

Write-Host ""
if ($lintOK -and $testOK) {
    Write-Host "  ALL GREEN -- ready to commit & push (CI gate should pass)." -ForegroundColor Green
    Write-Host "  Next:" -ForegroundColor Green
    Write-Host '     git add -A; git commit -m "Phase 6.1 + lint clean"; git push --set-upstream origin main' -ForegroundColor Gray
} else {
    Write-Host "  NOT GREEN -- paste validation_log.txt (or the red section) back to Claude." -ForegroundColor Yellow
}

Stop-Transcript | Out-Null
Write-Host "`nFull log written to: $log" -ForegroundColor DarkGray
