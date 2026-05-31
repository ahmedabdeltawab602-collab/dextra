"""
validate.py — OOXML sanity check for produced docx files.
Validates: opens, no null bytes, all required parts present.
"""
import zipfile, sys, os, re
from lxml import etree

REQUIRED_PARTS = ["[Content_Types].xml", "word/document.xml"]

def validate(path):
    errs = []
    if not os.path.exists(path):
        return [f"FILE_MISSING: {path}"]
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for r in REQUIRED_PARTS:
                if r not in names: errs.append(f"MISSING_PART: {r}")
            # parse document.xml
            with z.open("word/document.xml") as f:
                data = f.read()
                if b"\x00" in data:
                    errs.append("NULL_BYTES in document.xml")
                try:
                    etree.fromstring(data)
                except Exception as e:
                    errs.append(f"XML_PARSE_ERR: {e}")
    except zipfile.BadZipFile:
        errs.append("BAD_ZIP")
    return errs

if __name__ == "__main__":
    targets = sys.argv[1:] or []
    if not targets:
        # walk default dir
        root = "/sessions/bold-great-lamport/mnt/02 Statistics Fundamentals/Statistics Course Build"
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.endswith(".docx"):
                    targets.append(os.path.join(dirpath, fn))
    ok = bad = 0
    for t in targets:
        errs = validate(t)
        if errs:
            print(f"FAIL: {t}")
            for e in errs: print("   -", e)
            bad += 1
        else:
            print(f"OK  : {os.path.basename(t)}")
            ok += 1
    print(f"\n{ok} OK / {bad} FAIL")
    sys.exit(0 if bad == 0 else 1)
