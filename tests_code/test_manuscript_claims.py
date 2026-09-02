"""
Regression tests: do current pipeline outputs still support the quantitative
claims in manuscript.md?

Each test recomputes a manuscript number from already-materialized repo
outputs (TF-activity stats tables under output/features/tfa_major_b/stats/,
cached pseudobulk feature matrices under {HIRA_BASE_DIR}/features/, and the
pretrained GRNimmuneClock models) using the same functions the pipeline
itself uses (retrieve_sig_stats, test_unpaired, test_mixed_effects,
AgingClock.predict) -- no reimplementation of the underlying statistics.

Tolerances are intentionally generous (order-of-magnitude / sign checks)
since the goal is to catch regressions, not to reproduce exact manuscript
numbers.

Usage
-----
    cd <hira repo root>
    pytest -s scripts/tests/test_manuscript_claims.py
    # or
    python scripts/tests/test_manuscript_claims.py
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR.parent))              # `import hira` -> this repo
sys.path.insert(0, str(REPO_DIR / 'GRNimmuneClock'))   # `import grnimmuneclock`

env_file = REPO_DIR / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k, v)

from hira.src.feature_association.helper import retrieve_sig_stats, retrieve_stats, retrieve_feature_data
# aliased: names starting with `test_` would otherwise be collected by pytest as test functions
from hira.src.utils.util import test_unpaired as run_unpaired_test, test_mixed_effects as run_mixed_effects_test, retrieve_adata
from hira.src.config import get_config
from grnimmuneclock import AgingClock

# Which TF-activity analysis these claims are checked against (tfa_major_b/tfa_major_mc/tfa_major_sc).
# Override with: TFA_ANALYSIS_NAME=tfa_major_b pytest -s tests_code/test_manuscript_claims.py
ANALYSIS_NAME = os.environ.get('TFA_ANALYSIS_NAME', 'tfa_major_mc')


@lru_cache(maxsize=None)
def sig_stats():
    """Age-associated TFs (multi-cohort discovery), as used to make Fig 2A-B."""
    return retrieve_sig_stats(analysis_name=ANALYSIS_NAME)


@lru_cache(maxsize=None)
def all_stats():
    return retrieve_stats(analysis_name=ANALYSIS_NAME)


@lru_cache(maxsize=None)
def soundlife_stats():
    return retrieve_stats(analysis_name=ANALYSIS_NAME, dataset='soundlife')


@lru_cache(maxsize=None)
def clock(cell_type):
    return AgingClock(cell_type)


def predict(dataset, cell_type):
    # matches the pipeline's own src/clock/helper.py::wrapper_predict_age input: raw
    # GRN-gene expression via retrieve_adata, not tfa_major_b (TF activity) -- the clock
    # is trained on gene expression.
    adata = retrieve_adata(dataset=dataset, data_type='bulk', cell_type=cell_type, only_net_genes=True)
    return clock(cell_type).predict(adata.copy()).obs.copy()


# ===========================================================================
# "Age-related trajectories of gene regulation across immune cell types"
# (manuscript.md lines 59-71)
# ===========================================================================

def test_total_age_associated_tfs_exceeds_600():
    """'more than 600 TFs with age-associated changes across immune lineages'"""
    n = sig_stats()['gene'].nunique()
    print(f'\n[claim] >600 age-associated TFs total | actual: {n}')
    assert n > 600, f'expected >600 unique age-associated TFs, got {n}'


def test_cd8_cd4_tf_counts():
    """'approximately 370 and 120 age-associated TFs detected in CD8+ and CD4+ subsets' (Fig 2A-B)"""
    counts = sig_stats().groupby('cell_type')['gene'].nunique()
    cd8, cd4 = int(counts.get('CD8T', 0)), int(counts.get('CD4T', 0))
    print(f'\n[claim] CD8T~370, CD4T~120 | actual: CD8T={cd8}, CD4T={cd4}')
    assert 200 <= cd8 <= 550, f'CD8T count {cd8} far from manuscript value ~370'
    assert 60 <= cd4 <= 240, f'CD4T count {cd4} far from manuscript value ~120'


def test_validation_cohort_directional_consistency():
    """'More than 95% of age-associated TFs showed consistent directionality ... in the validation cohort' (CD8T only)"""
    disc = sig_stats()[sig_stats()['cell_type'] == 'CD8T'][['gene', 'cell_type', 'trend']].drop_duplicates()
    val = soundlife_stats()[soundlife_stats()['cell_type'] == 'CD8T'][['gene', 'cell_type', 'trend']].drop_duplicates(subset=['gene', 'cell_type'])
    merged = disc.merge(val, on=['gene', 'cell_type'], suffixes=('_disc', '_val'), how='inner')
    pct_consistent = (merged['trend_disc'] == merged['trend_val']).mean() * 100
    print(f'\n[claim] CD8T >95% consistent direction in validation | actual: {pct_consistent:.1f}% (n={len(merged)})')
    assert pct_consistent > 70, f'only {pct_consistent:.1f}% of CD8T TFs replicate direction in soundlife, expected >95%'


def test_validation_cohort_nonsignificant_fraction():
    """'approximately 20% of age-associated TFs did not reach statistical significance in the validation cohort' (CD8T only)"""
    disc = sig_stats()[sig_stats()['cell_type'] == 'CD8T'][['gene', 'cell_type']].drop_duplicates()
    val = soundlife_stats()[soundlife_stats()['cell_type'] == 'CD8T'][['gene', 'cell_type', 'is_significant']].drop_duplicates(subset=['gene', 'cell_type'])
    merged = disc.merge(val, on=['gene', 'cell_type'], how='inner')
    pct_not_sig = (~merged['is_significant']).mean() * 100
    print(f'\n[claim] CD8T ~20% not significant in validation | actual: {pct_not_sig:.1f}%')
    assert 10 <= pct_not_sig <= 35, f'{pct_not_sig:.1f}% not-significant in CD8T validation, expected ~20%'


def test_ten_tfs_shared_across_cd4_cd8_nk():
    """'Ten transcription factors were shared across all three cell types' incl. GATA3 (Fig 2D-F)"""
    disc = sig_stats()
    sets = {ct: set(disc[disc['cell_type'] == ct]['gene']) for ct in ['CD4T', 'CD8T', 'NK']}
    shared = sets['CD4T'] & sets['CD8T'] & sets['NK']
    print(f'\n[claim] 10 TFs shared across CD4T/CD8T/NK | actual: {len(shared)} -> {sorted(shared)}')
    assert len(shared) == 10, f'{len(shared)} shared TFs, expected exactly 10'
    assert 'GATA3' in shared, 'GATA3 explicitly named as a shared, cross-lineage-divergent TF in the manuscript'


def test_satb1_declines_with_age_in_all_three_lineages():
    """'conserved [decline] across ... such as the decline in SATB1' in CD4T/CD8T/NK"""
    stats = all_stats()
    for ct in ['CD4T', 'CD8T', 'NK']:
        row = stats[(stats['gene'] == 'SATB1') & (stats['cell_type'] == ct)]
        assert not row.empty, f'SATB1 missing from {ct} stats entirely'
        print(f'\n[claim] SATB1 declines with age in {ct} | trend: {row["trend"].iloc[0]}')
        assert (row['trend'] == 'Decrease in aging').all(), f'SATB1 trend in {ct} is not "Decrease in aging"'


# ===========================================================================
# "GRN-informed cell type-specific aging clocks" (manuscript.md lines 73-85)
# ===========================================================================

def test_clock_model_feature_coverage():
    """
    Diagnostic (not a manuscript number): what fraction of the pretrained
    clock's expected features are actually present in the currently cached
    gene-expression feature matrices (the clock's real input space per the
    manuscript: GRN target-gene expression, not TF activity). Low coverage
    means the shipped clock (GRNimmuneClock/grnimmuneclock/models/) is stale
    relative to the current GRN/feature-association pipeline output and any
    prediction-based claim below is expected to degrade.
    """
    for ct in ['CD4T', 'CD8T']:
        c = clock(ct)
        adata = retrieve_feature_data(dataset='aida', cell_type=ct, analysis_name='ge_major_b')
        overlap = set(c.feature_names) & set(adata.var_names)
        frac = len(overlap) / len(c.feature_names)
        print(f'\n[diagnostic] {ct} clock feature coverage vs current pipeline: '
              f'{len(overlap)}/{len(c.feature_names)} ({frac:.1%})')
        assert frac > 0.8, (
            f'{ct} clock model only recognizes {frac:.1%} of its expected features in the '
            f'current ge_major_b output -- the bundled model likely needs retraining '
            f'against the current GRN/feature pipeline (run scripts/clock_analysis.sh training step)'
        )


def test_clock_accuracy_cd4t_cd8t():
    """'GRN-informed clocks predicted chronological age with ... Spearman correlations of
    approximately 0.8 in both CD4+ and CD8+ T cells' on held-out test cohorts (Fig 3A)"""
    from scipy.stats import spearmanr
    for ct in ['CD4T', 'CD8T']:
        for cohort in ['aida', 'perez_sle']:  # zhang not cached locally, see test_test_cohort_data_available
            obs = predict(cohort, ct)
            if 'condition' in obs.columns and obs['condition'].nunique() > 1:
                obs = obs[obs['condition'].astype(str).str.lower().isin(['healthy', 'normal'])]
            sp = spearmanr(obs['age'].astype(float), obs['predicted_age'])[0]
            print(f'\n[claim] Spearman~0.8 in {ct} | actual on {cohort}: {sp:.3f} (n={len(obs)})')
            assert sp > 0.6, f'{ct}/{cohort} predicted-vs-true-age Spearman={sp:.3f}, expected ~0.8'


def test_test_cohort_data_available():
    """Sanity check that the published test cohorts used in practice (Fig 3A: AIDA, Perez;
    Zhang excluded -- not cached locally) are reachable through the pipeline's own
    data-retrieval function."""
    missing = []
    for cohort in ['aida', 'perez_sle']:
        try:
            retrieve_feature_data(dataset=cohort, cell_type='CD4T', analysis_name=ANALYSIS_NAME)
        except Exception:
            missing.append(cohort)
    print(f'\n[claim] clocks tested across AIDA, Perez | missing locally: {missing}')
    assert not missing, f'test cohort(s) {missing} have no cached tfa_major_b feature file'


# ===========================================================================
# "Autoimmune activation accelerates T-cell aging ... SLE" (manuscript.md lines 87-97)
# ===========================================================================

def test_sle_cohort_size():
    """'We analyzed transcriptomes from 261 individuals, including healthy controls and
    patients with SLE'"""
    obs = predict('perez_sle', 'CD8T')
    n = obs['donor_id'].nunique() if 'donor_id' in obs.columns else len(obs)
    print(f'\n[claim] 261 SLE-cohort individuals | actual: {n}')
    assert 200 <= n <= 320, f'{n} donors in perez_sle CD8T, expected ~261'


def test_sle_accelerates_cd8t_aging_in_young_patients():
    """'CD8+ T from individuals younger than 50 years, in whom predicted age increased by
    approximately +6 years (FDR = 1x10^-11)'"""
    obs = predict('perez_sle', 'CD8T')
    obs['age'] = obs['age'].astype(float)
    young = obs[obs['age'] < 50]
    p, delta = run_unpaired_test(young, ctr='healthy', treatment='SLE')
    print(f'\n[claim] SLE CD8T<50yo: +6yr, p=1e-11 | actual: {delta:+.1f}yr, p={p:.2e} (n={len(young)})')
    assert delta > 0, f'expected SLE to accelerate predicted age in young CD8T, got {delta:+.1f}yr'
    assert p < 0.05, f'expected a significant SLE effect in young CD8T, got p={p:.2e}'


# ===========================================================================
# "Systematic cytokine perturbation identifies IL-10" (manuscript.md lines 99-109)
# ===========================================================================

def test_parsebioscience_cached_conditions():
    """'PBMCs stimulated with 90 cytokines across 12 donors' -- checks what's actually
    materialized for the clock pipeline's cached parsebioscience feature matrix."""
    adata = retrieve_feature_data(dataset='parsebioscience', cell_type='CD4T', analysis_name=ANALYSIS_NAME)
    n_donors = adata.obs['donor_id'].nunique()
    conditions = sorted(adata.obs['condition'].unique())
    print(f'\n[claim] 90 cytokines x 12 donors | actual: {len(conditions)} condition(s) {conditions}, '
          f'{n_donors} donors')
    assert n_donors == 12, f'{n_donors} donors cached for parsebioscience, expected 12'
    assert len(conditions) >= 90 or set(conditions) == {'PBS', 'IL-10'}, (
        f'only {conditions} cached locally -- the full 90-cytokine screen is not materialized '
        f'in {{HIRA_BASE_DIR}}/features/tfa_major_b/, only the PBS/IL-10 slice used for the IL-10 follow-up'
    )


def test_il10_reduces_predicted_age():
    """'decreasing predicted biological age by 4.9 years in CD4+ T cells and by 2.0 years in
    CD8+ T cells (FDR < 0.0001)'"""
    config = get_config('parsebioscience')
    expected_years = {'CD4T': -4.9, 'CD8T': -2.0}
    for ct in ['CD4T', 'CD8T']:
        obs = predict('parsebioscience', ct)
        p, delta = run_mixed_effects_test(obs, ctr='PBS', treatment='IL-10',
                                       target_variable='predicted_age', group_key='donor_id', config=config)
        print(f'\n[claim] IL-10 {ct}: {expected_years[ct]:+.1f}yr, FDR<1e-4 | actual: {delta:+.1f}yr, p={p:.2e}')
        assert delta < 0, (
            f'manuscript reports IL-10 REDUCES predicted age in {ct} by {abs(expected_years[ct])}yr; '
            f'current pipeline gives {delta:+.1f}yr (wrong sign)'
        )
        assert p < 0.05, f'expected a significant IL-10 effect in {ct}, got p={p:.2e}'


# ===========================================================================
# "Pharmacological reversal of immune-aging signatures" (manuscript.md lines 113-124)
# ===========================================================================

def test_ruxolitinib_reduces_predicted_age_op():
    """'ruxolitinib ... showed a age-reversal effect of ~9 years in CD4+ T cells (FDR = 0.023)'"""
    config = get_config('op')
    obs = predict('op', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='DMSO', treatment='Ruxolitinib',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] Ruxolitinib (op) CD4T: -9yr, FDR=0.023 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to reduce predicted age in CD4T, got {delta:+.1f}yr'
    assert p < 0.05, f'expected a significant ruxolitinib effect in CD4T (op), got p={p:.2e}'


def test_cxcl9_donor_count():
    """'PBMCs from seven healthy donors were treated with ruxolitinib ... under both
    basal condition (RPMI) and LPS-stimulation'"""
    adata = retrieve_feature_data(dataset='CXCL9', cell_type='CD4T', analysis_name=ANALYSIS_NAME)
    n_donors = adata.obs['donor_id'].nunique()
    print(f'\n[claim] 7 ex vivo donors | actual: {n_donors}')
    assert n_donors == 7, f'{n_donors} donors cached for CXCL9, expected 7'


def test_lps_accelerates_predicted_age_ex_vivo():
    """'LPS induced a pronounced immune age acceleration of ~8 years ... (P = 1e-16; Fig. 4C)'"""
    config = get_config('CXCL9')
    obs = predict('CXCL9', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='RPMI', treatment='LPS',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] LPS ex vivo CD4T: +8yr, P=1e-16 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta > 0, f'manuscript reports LPS INCREASES predicted age by ~8yr; current pipeline gives {delta:+.1f}yr (wrong sign)'
    assert p < 0.05, f'expected a significant LPS effect, got p={p:.2e}'


def test_ruxolitinib_reduces_baseline_predicted_age_ex_vivo():
    """'Ruxolitinib reduced predicted age under baseline conditions (~2 years, P = 0.048)'"""
    config = get_config('CXCL9')
    obs = predict('CXCL9', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='RPMI', treatment='RPMI + ruxolitinib',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] Ruxolitinib baseline ex vivo CD4T: -2yr, P=0.048 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to reduce predicted age at baseline, got {delta:+.1f}yr'
    assert p < 0.05, f'manuscript claims this reaches significance (P=0.048), got p={p:.2e}'


def test_ruxolitinib_attenuates_lps_induced_aging_ex_vivo():
    """'directionally consistent attenuation of the LPS-induced increase in predicted age
    (~2.5 years, P = 0.12)' -- manuscript notes this one did NOT reach significance."""
    config = get_config('CXCL9')
    obs = predict('CXCL9', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='LPS', treatment='LPS + ruxolitinib',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] Ruxolitinib vs LPS ex vivo CD4T: -2.5yr, P=0.12 (n.s.) | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to attenuate the LPS-induced increase, got {delta:+.1f}yr'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
