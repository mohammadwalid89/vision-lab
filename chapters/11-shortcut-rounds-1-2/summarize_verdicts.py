"""Collect the 21 engine verdicts (DEV 3, Round 1 9, Round 2 9) into one small JSON.

Inputs are the verdict files the runs wrote; hashes, run ids and timestamps are
dropped. This produced `results/summary.json`.

    python summarize_verdicts.py --dev DEV_VERDICTS.json --r1 r1/VERDICTS.json \
        --r2-dir revision2/receipts --out results/summary.json
"""

import argparse
import json
import re
from pathlib import Path

from arms import PRIMARY_ARM, R2_KS


def _pick(verdict):
    return dict(
        label=verdict['label'],
        gates=verdict['bars'],
        failed=sorted(k for k, v in verdict['bars'].items() if v == 'FAIL'),
        worst_seed_values=verdict['metrics'],
    )


def summarize(dev, r1, r2_verdicts):
    rows = []
    for family in ('V1', 'V2', 'OWN'):
        rows.append(
            dict(
                stage='DEV',
                family=family,
                plan=dev[family]['id'],
                primary_arm=PRIMARY_ARM[family],
                **_pick(dev[family]),
            )
        )
    for item in r1:
        v = item['verdict']
        rows.append(
            dict(
                stage='R1',
                family=item['experiment'],
                iteration=item['iteration'],
                plan=item['plan_id'],
                primary_arm=PRIMARY_ARM[item['experiment']],
                **_pick(v),
            )
        )
    for v in r2_verdicts:
        family, it = v['id'].split('-')[3:5]
        family = family.upper()
        iteration = int(it[1:])
        rows.append(
            dict(
                stage='R2',
                family=family,
                iteration=iteration,
                k=R2_KS[iteration],
                plan=v['id'],
                primary_arm=('candidate@1' if family == 'V1' else 'candidate@0.5'),
                **_pick(v),
            )
        )
    return dict(
        n_attempts=len(rows),
        n_passing_all_gates=sum(not r['failed'] for r in rows),
        attempts=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--r2-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = json.loads(a.dev.read_text())
    r1 = json.loads(a.r1.read_text())
    files = sorted(a.r2_dir.glob('vision-wsp-r2-*_verdict.json'))
    r2 = [json.loads(f.read_text()) for f in files]
    if not all(re.fullmatch(r'vision-wsp-r2-\w+-i\d-\d+', v['id']) for v in r2):
        raise ValueError('unexpected Round 2 plan id')
    result = summarize(dev, r1, r2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + '\n')
    print(
        f'{result["n_attempts"]} attempts, {result["n_passing_all_gates"]} passing all gates'
    )


if __name__ == '__main__':
    main()
