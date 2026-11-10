"""Offline preparation of the Cosmetics event cache from the frozen split CSVs.

These modules rebuild the prepared cache and the event cache from ``data/source/``. They are standalone
(imported by ``scripts/prepare_data.py`` with this directory on ``sys.path``) and are independent of the
runtime package in ``src/mupsr``.
"""
