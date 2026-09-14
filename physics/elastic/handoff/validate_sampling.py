"""Compute-node checks of the weighted proposal and low-energy termination."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from physics.elastic.handoff.study import prepare, DEFAULT_CONFIG
from physics.elastic.handoff import run


def check_task(task):
    case, kind, chunk = task
    run.initialize(case, 1e-5, 64, 1., 0.)
    if kind == 'terminal':
        # Replay the original failed history, including its original draw order.
        rng = np.random.default_rng(np.random.SeedSequence([case['seed'], 1]))
        position = rng.random(3) @ run._TRANSPORT.structure.lattice_angstrom
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        track = run._TRANSPORT.trace('C', 1000, position, direction, 100., rng=rng)
        row = run.summarize_history(track, 1., run._TRANSPORT.kernels)
        report = run.statistics([{'rows': [row]}], .05)
        assert report['terminated_histories'] == 1
        assert 0 <= row[7] < 1
        return case['phase'], kind, report
    rng = np.random.default_rng(np.random.SeedSequence([78211, chunk]))
    weights = []
    for _ in range(64):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        _, proposal = run._TRANSPORT.sample_collision_tube_mixture(
            'C', 1e8, direction, 100., .8, rng)
        weights.append(proposal.target_over_proposal_weight)
    return case['phase'], kind, weights


def main():
    manifest = prepare(json.loads(DEFAULT_CONFIG.read_text()))
    tasks = []
    for phase in manifest['configuration']['phases']:
        case = next(c for c in manifest['cohorts'] if c['phase'] == phase and c['stage'] == 'pilot'
                    and c['direction'] == 'isotropic' and c['energy_ev'] == 1000 and c['boundary_potential_ev'] == 30)
        tasks.append((case, 'terminal', 0))
        tasks.extend((case, 'normalization', chunk) for chunk in range(4))
    weights = {phase: [] for phase in manifest['configuration']['phases']}
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=8) as pool:
        for phase, kind, result in tqdm(pool.map(check_task, tasks), total=len(tasks), unit='checks'):
            if kind == 'terminal':
                print(phase, 'original failed track retained', result, flush=True)
            else:
                weights[phase].extend(result)
    for phase, values in weights.items():
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        assert min(values) > 0 and max(values) <= 5 * (1 + 1e-12)
        assert abs(np.mean(values) - 1) < 5 * se
        print(phase, 'proposal mean weight', np.mean(values), 'SE', se, flush=True)
    print('PASS: original low-energy failures and defensive proposal normalization for both phases.')


if __name__ == '__main__':
    main()
