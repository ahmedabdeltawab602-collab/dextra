# The vendored suites are named test_*.py but are NOT collected directly; they
# are imported and executed by tests/test_legacy_suites.py (which turns their
# main()/assert checks into pytest results). Collecting them twice would run
# every suite a second time, so ignore them here.
collect_ignore_glob = ["test_*.py"]
