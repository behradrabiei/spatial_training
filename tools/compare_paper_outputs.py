"""Compare rebuilt numerical outputs and figures, ignoring only container metadata."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image


def outputs(root):
    return {p.relative_to(root): p for p in (root / 'paper_results').rglob('*')
            if p.is_file() and ('results' in p.parts or 'figures' in p.parts)}


def compare(a, b, roots):
    suffix = a.suffix
    if suffix == '.parquet':
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
    elif suffix == '.png':
        np.testing.assert_array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b)))
    elif suffix == '.pdf':
        # Matplotlib records export time; page objects and drawing commands must match.
        clean = lambda p: re.sub(rb'/(CreationDate|ModDate)\s*\([^)]*\)', b'', p.read_bytes())
        assert clean(a) == clean(b), 'PDF content differs'
    else:
        def content(p):
            data = gzip.decompress(p.read_bytes()) if suffix == '.gz' else p.read_bytes()
            for root in roots:
                data = data.replace(str(root).encode(), b'<ROOT>')
            return data
        assert content(a) == content(b), 'content differs'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--require-rebuilt', action='store_true', help='Reject outputs merely copied from this checkout')
    args = ap.parse_args()
    roots = [args.baseline.resolve(), args.candidate.resolve()]
    left, right = map(outputs, roots)
    failures = []
    counts = Counter()
    for rel in sorted(left.keys() | right.keys()):
        try:
            original = Path(__file__).resolve().parents[1] / rel
            if args.require_rebuilt and original.exists():
                for candidate in (left[rel], right[rel]):
                    assert candidate.stat().st_mtime_ns != original.stat().st_mtime_ns, 'Output was copied but not rebuilt'
            compare(left[rel], right[rel], roots)
            counts[rel.suffix] += 1
        except (AssertionError, KeyError, ValueError) as exc:
            failures.append({'path': str(rel), 'reason': str(exc)[:1000]})
    report = {'matched_by_format': dict(counts), 'failures': failures}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
