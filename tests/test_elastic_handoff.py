import json
import pytest
from physics.elastic.handoff.study import DEFAULT_CONFIG, prepare, compare_independent


def test_both_phases_and_worker_independent_ids():
    config = json.loads(DEFAULT_CONFIG.read_text())
    a = prepare(config)
    assert a == prepare(config)
    assert len(a['binary_tasks']) == 12
    assert len(a['cohorts']) == 240
    assert {x['phase'] for x in a['cohorts']} == set(config['phases'])
    assert sum(x['stage']=='pilot' for x in a['cohorts']) == 96
    assert len({x['id'] for x in a['cohorts']}) == 240
    assert not a['handoff_qualified']
    assert a['status'] == 'prepared'
    config['boundary_potential_ev']=[15,30]
    with pytest.raises(ValueError, match='30 eV'):
        prepare(config)


def test_comparison_does_not_pass_unresolved_or_invalid_uncertainty():
    a={'mean':1.,'standard_error':.01}
    assert compare_independent(a,{'mean':1.01,'standard_error':.01})['status']=='screen_pass'
    assert compare_independent(a,{'mean':1.01,'standard_error':.2})['status']=='unresolved'
    for mean,se in [(0,.01),(1,0),(float('nan'),.01)]:
        assert compare_independent(a,{'mean':mean,'standard_error':se})['status']=='unresolved'
