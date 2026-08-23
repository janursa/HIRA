"""Checks that raw dataset files + PRIOR_DIR haven't silently changed, and that
this manifest itself isn't so old it can no longer be trusted.

Regenerate the manifest after an intentional raw-data update:
    python tests/preprocessing/fixtures/make_raw_manifest.py
"""
import json
import os
import time

import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'raw_manifest.json')
MANIFEST_MAX_AGE_DAYS = 365


@pytest.fixture(scope='module')
def manifest():
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def test_manifest_not_stale():
    age_days = (time.time() - os.path.getmtime(MANIFEST_PATH)) / 86400
    assert age_days < MANIFEST_MAX_AGE_DAYS, (
        f'raw_manifest.json is {age_days:.0f} days old (>{MANIFEST_MAX_AGE_DAYS}) — '
        'regenerate it with fixtures/make_raw_manifest.py after confirming raw data is still correct'
    )


def test_raw_files_unchanged(manifest):
    changed, missing = [], []
    for key, entry in manifest.items():
        path = key.split(':', 2)[-1]
        if not os.path.exists(path):
            missing.append(path)
            continue
        st = os.stat(path)
        if st.st_size != entry['size'] or st.st_mtime != entry['mtime']:
            changed.append(path)
    assert not missing, f'raw files went missing since manifest was recorded: {missing}'
    assert not changed, (
        f'raw files changed since manifest was recorded (size/mtime mismatch): {changed}\n'
        'If this is intentional, regenerate the manifest with fixtures/make_raw_manifest.py.'
    )
