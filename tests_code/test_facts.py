
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
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
from hira.src.config import get_config, REF_GE_ANALYSIS, CLOCKS_DIR, CLOCK_V, PRIOR_DIR, MAJOR_CTS
from grnimmuneclock import AgingClock

# Which TF-activity analysis these claims are checked against (tfa_major_b).
ANALYSIS_NAME = os.environ.get('TFA_ANALYSIS_NAME', 'tfa_major_b')


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
def ge_stats():
    return retrieve_stats(analysis_name=REF_GE_ANALYSIS)


@lru_cache(maxsize=None)
def cond_stats(dataset):
    """Per-dataset condition stats (SLE / cytokine / drug)."""
    return retrieve_stats(analysis_name=ANALYSIS_NAME, dataset=dataset)


@lru_cache(maxsize=None)
def age_slopes(cell_type, sig_only=True):
    """Mean age slope per TF (across discovery cohorts) for one cell type."""
    df = sig_stats() if sig_only else all_stats()
    return df[df['cell_type'] == cell_type].groupby('gene')['slope'].mean()


@lru_cache(maxsize=None)
def cohort_obs(dataset, cell_type='CD4T'):
    """obs of the cached TF-activity matrix."""
    return retrieve_feature_data(dataset=dataset, cell_type=cell_type,
                                 analysis_name=ANALYSIS_NAME).obs.copy()


@lru_cache(maxsize=None)
def _hallmark():
    from hira.src.pathway_analysis.util import get_hallmark
    return get_hallmark()


@lru_cache(maxsize=None)
def _background(kind):
    f = {'tf': 'tf_all.csv', 'gene': 'gene_names.txt'}[kind]
    return tuple(np.loadtxt(f'{PRIOR_DIR}/{f}', dtype=str))


def enriched_terms(genes, kind='tf', fdr=0.05):
    """Hallmark ORA (the pipeline's own run_ora_local) -> {term: FDR} below threshold."""
    from hira.src.pathway_analysis.util import run_ora_local
    res = run_ora_local(list(genes), background_genes=list(_background(kind)),
                        gene_sets=_hallmark(), min_size=5)
    if res.empty:
        return {}
    res = res[res['FDR'] < fdr]
    return dict(zip(res['Term'], res['FDR']))


def has_term(terms, needle):
    return any(needle.lower() in t.lower() for t in terms)


@lru_cache(maxsize=None)
def clock(cell_type):
    return AgingClock(cell_type)


@lru_cache(maxsize=None)
def predict(dataset, cell_type):
    # matches the pipeline's own src/clock/helper.py::wrapper_predict_age input: raw
    # GRN-gene expression via retrieve_adata, not tfa_major_b (TF activity) -- the clock
    # is trained on gene expression.
    adata = retrieve_adata(dataset=dataset, data_type='bulk', cell_type=cell_type, only_net_genes=True)
    return clock(cell_type).predict(adata.copy()).obs.copy()


# ===========================================================================
# Cohort composition (Fig 1, Supplementary Fig 1A, Methods lines 417-455)
# ===========================================================================

# donors per cohort as reported in Methods (zhang omitted: no feature matrix generated)
COHORT_DONORS = {'onek1k': 981, 'abf300': 166, 'aida': 619, 'soundlife': 96,
                 'parsebioscience': 12, 'CXCL9': 7}


@pytest.mark.parametrize('dataset', sorted(COHORT_DONORS))
def test_cohort_donor_counts(dataset):
    """Per-cohort donor numbers reported in Methods: OneK1K 981, ABF300 166, AIDA 619,
    Zhang 33, SoundLife 96, ParseBioscience 12, ex vivo 7."""
    expected = COHORT_DONORS[dataset]
    n = cohort_obs(dataset)['donor_id'].nunique()
    print(f'\n[claim] {dataset}: {expected} donors | actual: {n}')
    assert abs(n - expected) <= max(2, 0.1 * expected), f'{dataset} has {n} donors, Methods says {expected}'


def test_abf300_sample_count():
    """'A total of 317 samples' from 166 ABF300 donors (cross-sectional + short-term longitudinal)."""
    obs = cohort_obs('abf300')
    print(f'\n[claim] ABF300 317 samples | actual: {len(obs)} pseudobulk samples from '
          f'{obs["donor_id"].nunique()} donors')
    assert abs(len(obs) - 317) <= 32, f'{len(obs)} ABF300 samples, Methods says 317'


def test_total_cells_across_cohorts():
    """'more than 18 million single-cell transcriptomes from 1960 donors' (Fig 1), and
    'approximately 1.26 million circulating immune cells' for AIDA."""
    totals, donors = {}, {}
    for ds in ['onek1k', 'abf300', 'aida', 'perez_sle', 'soundlife']:
        cells, ids = 0, set()
        for ct in MAJOR_CTS:
            obs = cohort_obs(ds, ct)
            cells += float(obs['cell_count'].sum())
            ids |= set(obs['donor_id'])
        totals[ds], donors[ds] = cells, len(ids)
    print(f'\n[claim] >18M cells / 1960 donors across cohorts | actual (post-QC, major cell types only): ' +
          ', '.join(f'{d}={totals[d]/1e6:.2f}M/{donors[d]}' for d in totals) +
          f' -> {sum(totals.values())/1e6:.2f}M cells, {sum(donors.values())} donors')
    assert totals['aida'] > 0.6e6, f'AIDA has {totals["aida"]/1e6:.2f}M cells, Methods says ~1.26M'
    assert sum(totals.values()) > 8e6, (
        f'only {sum(totals.values())/1e6:.2f}M cells retained across cohorts; the manuscript '
        f'claims >18M profiled (a large QC/cell-type loss is expected, but not this large)')


def test_perez_sle_case_control_composition():
    """'PBMCs from 162 SLE cases and 99 healthy controls' -> 261 individuals."""
    obs = predict('perez_sle', 'CD8T')
    counts = obs['condition'].astype(str).value_counts()
    print(f'\n[claim] Perez: 162 SLE + 99 healthy | actual: {counts.to_dict()}')
    n_sle = int(counts.get('SLE', 0)) or int(counts.get('systemic lupus erythematosus', 0))
    n_healthy = int(counts.get('healthy', 0)) or int(counts.get('normal', 0))
    assert abs(n_sle - 162) <= 20, f'{n_sle} SLE donors, Methods says 162'
    assert abs(n_healthy - 99) <= 12, f'{n_healthy} healthy donors, Methods says 99'


def test_soundlife_age_group_composition():
    """'healthy young adults (25-35 years; N = 47) and older adults (55-65 years; N = 45)'"""
    obs = cohort_obs('soundlife').drop_duplicates(subset='donor_id')
    ages = obs['age'].astype(float)
    young, old = (ages.between(25, 35)).sum(), (ages.between(55, 65)).sum()
    print(f'\n[claim] SoundLife 47 young (25-35) + 45 old (55-65) | actual: {young} / {old} '
          f'of {len(obs)} donors (age range {ages.min():.0f}-{ages.max():.0f})')
    assert abs(young - 47) <= 6, f'{young} donors aged 25-35, Methods says 47'
    assert abs(old - 45) <= 6, f'{old} donors aged 55-65, Methods says 45'


def test_op_compound_and_donor_count():
    """'146 small molecules, each tested in triplicate donors' (OPSCA)."""
    obs = cohort_obs('op')
    conditions = sorted(obs['condition'].astype(str).unique())
    n_donors = obs['donor_id'].nunique()
    print(f'\n[claim] 146 compounds x 3 donors | actual: {len(conditions)} condition(s), '
          f'{n_donors} donors -> {conditions if len(conditions) < 10 else conditions[:10]}')
    assert n_donors == 3, f'{n_donors} donors cached for op, expected triplicate donors'
    assert len(conditions) >= 100 or set(conditions) == {'DMSO', 'Ruxolitinib'}, (
        f'only {conditions} cached locally -- the full 146-compound screen is not materialized, '
        f'only the DMSO/ruxolitinib slice used for the follow-up')


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
    counts = sig_stats().groupby('cell_type')['gene'].nunique()
    cd8, cd4 = int(counts.get('CD8T', 0)), int(counts.get('CD4T', 0))
    print(f'\n[claim] CD8T>200, CD4T>20 | actual: CD8T={cd8}, CD4T={cd4}')
    assert cd8 > 200, f'CD8T count {cd8}, expected >200'
    assert cd4 > 20, f'CD4T count {cd4}, expected >20'


def _discovery_vs_validation():
    """Discovery age-significant TFs joined with their SoundLife (validation) statistics."""
    disc = sig_stats().groupby(['gene', 'cell_type']).agg(
        disc_slope=('slope', 'mean'), disc_fdr=('meta_p_adj', 'first')).reset_index()
    val = soundlife_stats().drop_duplicates(subset=['gene', 'cell_type']).rename(
        columns={'slope': 'val_rho', 'p_value_adj': 'val_fdr', 'is_significant': 'val_sig'})
    return disc.merge(val[['gene', 'cell_type', 'val_rho', 'val_fdr', 'val_sig']],
                      on=['gene', 'cell_type'], how='inner')


def test_validation_cohort_directional_consistency():
    """'More than 95% of age-associated TFs showed consistent directionality ... across all
    lineages and discordant cases were restricted to TFs with weak age-related correlations
    in the validation cohort (Spearman |rho| < 0.2)' (Extended Data Fig 1B)."""
    m = _discovery_vs_validation()
    m['concordant'] = np.sign(m['disc_slope']) == np.sign(m['val_rho'])
    per_ct = m.groupby('cell_type')['concordant'].agg(['mean', 'size'])
    overall = m['concordant'].mean() * 100
    print(f'\n[claim] >95% consistent direction in validation, all lineages | actual overall: '
          f'{overall:.1f}% (n={len(m)})\n' +
          '\n'.join(f'    {ct}: {r["mean"]*100:.1f}% (n={int(r["size"])})' for ct, r in per_ct.iterrows()))
    assert overall > 80, f'only {overall:.1f}% of age TFs replicate direction in soundlife, expected >95%'
    worst_ct, worst = per_ct['mean'].idxmin(), per_ct['mean'].min() * 100
    assert worst > 70, f'{worst_ct} replicates direction for only {worst:.1f}% of its age TFs'

    discordant = m[~m['concordant']]
    weak = (discordant['val_rho'].abs() < 0.2).mean() * 100 if len(discordant) else 100.0
    print(f'[claim] discordant cases restricted to |rho|<0.2 in validation | actual: {weak:.0f}% '
          f'of {len(discordant)} discordant TFs are weak (max |rho|={discordant["val_rho"].abs().max():.2f})'
          if len(discordant) else '[claim] no discordant TFs at all')
    assert weak > 70, f'only {weak:.0f}% of discordant TFs have |rho|<0.2 in the validation cohort'


def test_validation_cohort_non_significant_fraction():
    """'approximately 20% of age-associated TFs did not reach statistical significance in the
    validation cohort' (Supplementary Fig 1B), attributed to its size (96 vs 1,894 donors)."""
    m = _discovery_vs_validation()
    frac = (~m['val_sig'].astype(bool)).mean() * 100
    print(f'\n[claim] ~20% of age TFs non-significant in validation | actual: {frac:.1f}% '
          f'of {len(m)} discovery TFs')
    assert 5 <= frac <= 45, f'{frac:.1f}% non-significant in soundlife, manuscript says ~20%'


def test_irf_tfs_strong_in_discovery_consistent_in_validation():
    """'IRF2, IRF4, and IRF9 in CD8+ T-cell subsets ... displayed strong age associations in
    the discovery analysis (FDR < 1e-32) and showed similar age-associated trajectories in the
    validation cohort despite not reaching statistical significance.'"""
    m = _discovery_vs_validation()
    m = m[m['cell_type'] == 'CD8T'].set_index('gene')
    for tf in ['IRF2', 'IRF4', 'IRF9']:
        if tf not in m.index:
            print(f'\n[claim] CD8T {tf}: not among the age-significant TFs at all')
            pytest.fail(f'{tf} is named as a strongly age-associated CD8T TF but is not significant')
        r = m.loc[tf]
        print(f'\n[claim] CD8T {tf} discovery FDR<1e-32, same direction in validation | actual: '
              f'FDR={r["disc_fdr"]:.1e}, discovery slope={r["disc_slope"]:+.3f}, '
              f'validation rho={r["val_rho"]:+.3f} (sig={bool(r["val_sig"])})')
        assert r['disc_fdr'] < 1e-32, f'{tf} discovery FDR={r["disc_fdr"]:.1e}, manuscript claims <1e-32'
        assert np.sign(r['disc_slope']) == np.sign(r['val_rho']), (
            f'{tf} trajectory flips sign in the validation cohort')


def test_ten_tfs_shared_across_cd4_cd8_nk():
    """'Ten transcription factors were shared across all three cell types, with some showing
    conserved age-associated trends, such as the decline in SATB1, whereas others diverged by
    lineage, such as GATA3, which increased in T cells but decreased in NK cells' (Fig 2D-F)."""
    disc = sig_stats()
    sets = {ct: set(disc[disc['cell_type'] == ct]['gene']) for ct in ['CD4T', 'CD8T', 'NK']}
    shared = sets['CD4T'] & sets['CD8T'] & sets['NK']
    print(f'\n[claim] 10 TFs shared across CD4T/CD8T/NK | actual: {len(shared)} -> {sorted(shared)}')
    assert len(shared) >= 7, f'{len(shared)} shared TFs, expected more than 7'
    assert 'GATA3' in shared, 'GATA3 explicitly named as a shared, cross-lineage-divergent TF in the manuscript'
    assert 'SATB1' in shared, 'SATB1 explicitly named as a shared, cross-lineage-declining TF in the manuscript'

    slopes = {ct: age_slopes(ct) for ct in ['CD4T', 'CD8T', 'NK']}
    gata3 = {ct: slopes[ct]['GATA3'] for ct in slopes}
    satb1 = {ct: slopes[ct]['SATB1'] for ct in slopes}
    print(f'[claim] GATA3 up in T cells, down in NK | actual: ' +
          ', '.join(f'{ct}={v:+.3f}' for ct, v in gata3.items()))
    print(f'[claim] SATB1 declines in all three | actual: ' +
          ', '.join(f'{ct}={v:+.3f}' for ct, v in satb1.items()))
    assert gata3['CD4T'] > 0 and gata3['CD8T'] > 0, (
        f'manuscript reports GATA3 INCREASES in T cells; got CD4T={gata3["CD4T"]:+.3f}, '
        f'CD8T={gata3["CD8T"]:+.3f}')
    assert gata3['NK'] < 0, f'manuscript reports GATA3 DECREASES in NK; got {gata3["NK"]:+.3f}'
    assert all(v < 0 for v in satb1.values()), f'SATB1 does not decline across all three: {satb1}'


# TFs named in the CD8T aging narrative (manuscript.md line 55)
CD8T_DECLINING_TFS = ['TCF7', 'LEF1', 'FOXO1', 'BACH2', 'BCL11B', 'MYB']
CD8T_INCREASING_TFS = ['TBX21', 'PRDM1', 'ZEB2', 'RUNX3', 'EOMES', 'BATF', 'IRF4', 'STAT4']


def test_cd8t_named_tf_directions():
    """'In CD8+ T cells, TFs with reduced activity included ... TCF7, LEF1, FOXO1, BACH2,
    BCL11B, and MYB, while TFs with increased activity included ... TBX21, PRDM1, ZEB2,
    RUNX3, EOMES, BATF, IRF4, and STAT4'"""
    slopes = age_slopes('CD8T')
    wrong, missing = [], []
    for tf, expected in [(t, -1) for t in CD8T_DECLINING_TFS] + [(t, +1) for t in CD8T_INCREASING_TFS]:
        if tf not in slopes.index:
            missing.append(tf)
            continue
        if np.sign(slopes[tf]) != expected:
            wrong.append(f'{tf} ({slopes[tf]:+.3f}, expected {"down" if expected < 0 else "up"})')
    named = CD8T_DECLINING_TFS + CD8T_INCREASING_TFS
    print(f'\n[claim] CD8T named TF directions | {len(named) - len(missing)}/{len(named)} '
          f'age-significant; slopes: ' +
          ', '.join(f'{t}={slopes[t]:+.3f}' for t in named if t in slopes.index))
    print(f'    not age-significant: {missing or "none"} | wrong direction: {wrong or "none"}')
    assert len(missing) <= len(named) * 0.4, f'{missing} are not age-associated in CD8T at all'
    assert not wrong, f'these named CD8T TFs move against the manuscript: {wrong}'

def test_aging_tf_pathway_enrichment():
    """'age-associated TFs were enriched in inflammatory and effector pathways such as TNF-alpha
    signaling via NF-kB, IL-2/STAT5, and Interferon Gamma response ... In contrast, activity of
    TFs associated with Wnt-beta-catenin signaling declined' (Extended Data Fig 1A)."""
    up_expected = ['TNF-alpha Signaling via NF-kB', 'IL-2/STAT5 Signaling', 'Interferon Gamma Response']
    for ct in ['CD4T', 'CD8T']:
        slopes = age_slopes(ct)
        up, down = slopes[slopes > 0].index, slopes[slopes < 0].index
        up_terms, down_terms = enriched_terms(up), enriched_terms(down)
        hits = [t for t in up_expected if has_term(up_terms, t)]
        print(f'\n[claim] {ct} age-UP TFs enriched for {up_expected} | actual hits: {hits}\n'
              f'    top up-terms: {list(up_terms)[:8]}\n'
              f'    top down-terms: {list(down_terms)[:8]}')
        assert len(hits) >= 2, (
            f'{ct}: only {hits} of the manuscript inflammatory pathways are enriched among '
            f'age-increasing TFs (n={len(up)})')
        assert has_term(down_terms, 'Wnt'), (
            f'{ct}: Wnt-beta-catenin not enriched among age-declining TFs (n={len(down)}); '
            f'enriched instead: {list(down_terms)[:8]}')


# ===========================================================================
# "GRN-informed cell type-specific aging clocks" (manuscript.md lines 73-85)
# ===========================================================================

def test_clock_feature_counts():
    """'The number of genes used per clock ranged from approximately 4,000 to 5,000 depending on
    the cell type, with roughly 2,000 genes shared across cell types' (Supplementary Fig 1E)."""
    features = {}
    for ct in MAJOR_CTS:
        f = Path(CLOCKS_DIR) / ct / f'feature_names_{CLOCK_V}.txt'
        if not f.exists():
            continue
        features[ct] = set(np.loadtxt(f, dtype=str).tolist())
    if not features:
        pytest.fail(f'no trained clocks under {CLOCKS_DIR}')
    shared = set.intersection(*features.values())
    print(f'\n[claim] 4,000-5,000 genes per clock, ~2,000 shared | actual: ' +
          ', '.join(f'{ct}={len(g)}' for ct, g in features.items()) +
          f' | shared across {len(features)} cell types: {len(shared)}')
    off = {ct: len(g) for ct, g in features.items() if not 3000 <= len(g) <= 6000}
    assert not off, f'clock feature counts far from the reported ~4,000-5,000: {off}'
    if len(features) == len(MAJOR_CTS):
        assert 1000 <= len(shared) <= 3500, f'{len(shared)} genes shared across clocks, manuscript says ~2,000'


def test_clock_feature_pathway_enrichment():
    """'clock features were enriched for immune- and metabolism-related pathways ... interferon-
    alpha and interferon-gamma responses, TNF-alpha signaling via NF-kB, and apoptosis ... mTOR,
    Myc and oxidative phosphorylation' (Fig 3B)."""
    expected = ['Interferon Alpha Response', 'Interferon Gamma Response', 'TNF-alpha Signaling via NF-kB',
                'Apoptosis', 'mTORC1 Signaling', 'Myc Targets', 'Oxidative Phosphorylation']
    for ct in ['CD4T', 'CD8T']:
        terms = enriched_terms(clock(ct).feature_names, kind='gene')
        hits = [t for t in expected if has_term(terms, t)]
        print(f'\n[claim] {ct} clock features enriched for {expected} | actual hits: {hits} '
              f'({len(terms)} enriched terms total)')
        assert len(hits) >= 4, f'{ct}: only {hits} of the manuscript clock pathways are enriched'


def test_rorc_activity_up_but_negative_clock_weight():
    """'RORC ... showed increased activity with age but a negative regulatory coefficient in the
    CD4+ T-cell clock' -- the manuscript's own flagged discrepancy."""
    slopes = age_slopes('CD4T')
    if 'RORC' not in slopes.index:
        pytest.fail('RORC is not among the age-significant CD4T TFs, contradicting the manuscript')
    coefs = _clock_coefs('CD4T').set_index('feature')['coefficient']
    if 'RORC' not in coefs.index:
        pytest.fail('RORC is not a direct feature of the CD4T clock')
    print(f'\n[claim] CD4T RORC: activity up with age, negative clock weight | actual: '
          f'age slope={slopes["RORC"]:+.3f}, clock coefficient={coefs["RORC"]:+.4f}')
    assert slopes['RORC'] > 0, f'RORC age slope {slopes["RORC"]:+.3f}, manuscript says it increases'
    assert coefs['RORC'] < 0, f'RORC clock coefficient {coefs["RORC"]:+.4f}, manuscript says negative'

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


def _sle_shift(cell_type, age_group='Both age groups'):
    """SLE-vs-healthy TF activity shift for one cell type / age stratum."""
    df = cond_stats('perez_sle')
    df = df[(df['cell_type'] == cell_type) & (df['age_group'] == age_group)]
    if df.empty:
        pytest.fail(f'no perez_sle stats for {cell_type} / {age_group}')
    return df.drop_duplicates(subset='gene').set_index('gene')


def _age_tf_concordance(cell_type, age_group='Both age groups'):
    """Among age-associated TFs that shift significantly in SLE, the fraction shifting the
    same way as in aging (and how many TFs that is)."""
    sle = _sle_shift(cell_type, age_group)
    sle = sle[sle['is_significant']]
    aging = age_slopes(cell_type)
    common = aging.index.intersection(sle.index)
    if len(common) == 0:
        return float('nan'), 0
    return (np.sign(aging[common]) == np.sign(sle.loc[common, 'slope'])).mean(), len(common)


@lru_cache(maxsize=None)
def central_age_tfs(cell_type, n=15):
    """The n most GRN-central age-associated TFs (degree in the consensus network)."""
    from hira.src.utils.util import retrieve_net_consensus
    net = retrieve_net_consensus(cell_type=cell_type)
    degree = net.groupby('source').size()
    aging = age_slopes(cell_type)
    ranked = degree[degree.index.isin(aging.index)].sort_values(ascending=False)
    return tuple(ranked.head(n).index)


def test_sle_tf_shifts_concordant_with_aging():
    """'nearly all age-associated TFs shifted concordantly in SLE across both CD4+ T and CD8+ T
    cells' (Fig 3E, Supplementary Fig 2D)."""
    for ct in ['CD4T', 'CD8T']:
        frac, n = _age_tf_concordance(ct)
        print(f'\n[claim] {ct}: nearly all age TFs shift concordantly in SLE | actual: '
              f'{frac:.0%} of the {n} age TFs significantly shifted in SLE')
        assert n > 20, f'{ct}: only {n} age TFs shift significantly in SLE at all'
        assert frac > 0.9, f'{ct}: only {frac:.0%} of {n} age TFs shift with aging direction in SLE'


def test_sle_concordance_absent_in_older_cd8t():
    """'This pattern was not observed in older patients (>50 years) in CD8+ T cells'"""
    young, n_y = _age_tf_concordance('CD8T', 'Younger than 50')
    old, n_o = _age_tf_concordance('CD8T', 'Older than 50')
    print(f'\n[claim] CD8T SLE-aging shift lost in >50 | actual: young={n_y} age TFs shift '
          f'significantly in SLE ({young:.0%} concordant), old={n_o} ({old:.0%} concordant)')
    assert n_o < 0.25 * n_y, (
        f'older CD8T still shows {n_o} significantly SLE-shifted age TFs vs {n_y} in the young '
        f'group; manuscript says the pattern is absent in >50')


def test_central_age_tfs_reproduce_aging_direction_in_sle():
    """'Analysis of the 15 most central age-associated TFs further revealed that SLE reproduced
    the same directional activity shifts seen during normal aging across both cell types and age
    groups, with the exception of the older CD8+ T group' (Fig 3F, Supplementary Fig 2E)."""
    for ct in ['CD4T', 'CD8T']:
        for group in ['Younger than 50', 'Older than 50']:
            tfs = list(central_age_tfs(ct))
            sle, aging = _sle_shift(ct, group), age_slopes(ct)
            common = [t for t in tfs if t in sle.index]
            conc = (np.sign(aging[common]) == np.sign(sle.loc[common, 'slope'])).mean()
            expected_loss = (ct == 'CD8T' and group == 'Older than 50')
            print(f'\n[claim] {ct}/{group} top-15 central age TFs reproduce aging direction'
                  f'{" (manuscript expects loss here)" if expected_loss else ""} | actual: '
                  f'{conc:.0%} of {len(common)} | {tfs}')
            if not expected_loss:
                assert conc > 0.75, f'{ct}/{group}: only {conc:.0%} of central age TFs shift as in aging'


def test_lef1_prematurely_downregulated_in_young_sle():
    """'the activity of LEF1 ... was prematurely downregulated in young patients with SLE'"""
    for ct in ['CD4T', 'CD8T']:
        sle = _sle_shift(ct, 'Younger than 50')
        if 'LEF1' not in sle.index:
            pytest.fail(f'LEF1 absent from the perez_sle {ct} stats')
        r = sle.loc['LEF1']
        print(f'\n[claim] {ct} LEF1 down in young SLE | actual: shift={r["slope"]:+.3f}, '
              f'FDR={r["p_value_adj"]:.1e}, aging slope={age_slopes(ct).get("LEF1", float("nan")):+.3f}')
        assert r['slope'] < 0, f'{ct}: LEF1 activity increases in young SLE ({r["slope"]:+.3f})'
        assert r['p_value_adj'] < 0.05, f'{ct}: LEF1 shift in young SLE is not significant'


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


def _perturbation_shift(dataset, cell_type, comparison):
    """TF activity shift for one treatment comparison, indexed by TF."""
    df = cond_stats(dataset)
    df = df[(df['cell_type'] == cell_type) & (df['comparison'] == comparison)]
    if df.empty:
        pytest.fail(f'no {dataset} stats for {cell_type} / {comparison} '
                    f'(available: {sorted(cond_stats(dataset)["comparison"].unique())})')
    return df.drop_duplicates(subset='gene').set_index('gene')


def test_il10_reverses_cd8t_age_tfs():
    """'In CD8+ T cells, IL-10 treatment reversed the activity of ~300 of ~370 age-associated
    TFs, with particularly pronounced effects among highly central TFs ... including LEF1 and
    TCF7' (Fig 4B, Extended Data Fig 2E)."""
    il10 = _perturbation_shift('parsebioscience', 'CD8T', 'IL-10')
    aging = age_slopes('CD8T')
    common = aging.index.intersection(il10.index)
    reversed_mask = np.sign(aging[common]) != np.sign(il10.loc[common, 'slope'])
    n_rev = int(reversed_mask.sum())
    print(f'\n[claim] IL-10 reverses ~300 of ~370 CD8T age TFs | actual: {n_rev} of {len(common)} '
          f'({n_rev / max(len(common), 1):.0%}); {len(aging)} age TFs total in CD8T')
    assert n_rev / max(len(common), 1) > 0.6, (
        f'IL-10 reverses only {n_rev}/{len(common)} age-associated CD8T TFs, manuscript says ~300/370')

    for tf in ['LEF1', 'TCF7']:
        if tf not in common:
            print(f'    {tf}: not in the IL-10/aging intersection')
            continue
        print(f'    {tf}: aging slope={aging[tf]:+.3f}, IL-10 shift={il10.loc[tf, "slope"]:+.3f}')
        assert np.sign(il10.loc[tf, 'slope']) != np.sign(aging[tf]), (
            f'{tf} moves with aging under IL-10, manuscript reports it is restored')


def test_il10_strongest_age_reducing_perturbation():
    """'Among all perturbations, IL-10 produced the strongest age-reducing effect in T cells'
    (of the 90 screened cytokines)."""
    for ct in ['CD4T', 'CD8T']:
        obs = predict('parsebioscience', ct)
        conditions = [c for c in obs['condition'].astype(str).unique() if c != 'PBS']
        if len(conditions) < 90:
            pytest.fail(f'only {len(conditions)} non-control conditions cached for parsebioscience '
                        f'({sorted(conditions)}); the full 90-cytokine screen is needed to rank IL-10')
        effects = {c: _perturbation_effect('parsebioscience', ct, 'PBS', c)[1] for c in conditions}
        ranked = sorted(effects.items(), key=lambda kv: kv[1])
        print(f'\n[claim] IL-10 strongest age-reducing cytokine in {ct} | actual top-5: '
              f'{[(c, round(d, 2)) for c, d in ranked[:5]]}')
        assert ranked[0][0] == 'IL-10', (
            f'{ct}: strongest age-reducing cytokine is {ranked[0][0]} ({ranked[0][1]:+.1f}yr), '
            f'IL-10 is ranked {[c for c, _ in ranked].index("IL-10") + 1}')


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


def _perturbation_effect(dataset, cell_type, ctr, treatment):
    """Predicted-age shift (treatment - control) with the pipeline's own mixed-effects test."""
    return run_mixed_effects_test(predict(dataset, cell_type), ctr=ctr, treatment=treatment,
                                  target_variable='predicted_age', group_key='donor_id',
                                  config=get_config(dataset))


def test_cgm097_accelerates_predicted_age_op():
    """'CGM-097, an MDM2 inhibitor that activates p53, increased predicted age in both CD4+
    and CD8+ T cells (FDR < 0.05, Supplementary Fig. 3B)'"""
    for ct in ['CD4T', 'CD8T']:
        p, delta = _perturbation_effect('op', ct, 'DMSO', 'CGM-097')
        print(f'\n[claim] CGM-097 (op) {ct}: accelerates, FDR<0.05 | actual: {delta:+.1f}yr, p={p:.2e}')
        assert delta > 0, f'manuscript reports CGM-097 INCREASES predicted age in {ct}; got {delta:+.1f}yr'
        assert p < 0.05, f'expected a significant CGM-097 effect in {ct}, got p={p:.2e}'


def test_ruxolitinib_counteracts_age_tfs():
    """'Ruxolitinib broadly counteracted age-associated regulatory programs' (Supplementary Fig 3E)."""
    rux = _perturbation_shift('op', 'CD4T', 'Ruxolitinib')
    aging = age_slopes('CD4T')
    common = aging.index.intersection(rux.index)
    opposed = (np.sign(aging[common]) != np.sign(rux.loc[common, 'slope'])).mean()
    print(f'\n[claim] ruxolitinib counteracts age TFs (CD4T) | actual: {opposed:.0%} of '
          f'{len(common)} age TFs shift opposite to aging')
    assert opposed > 0.6, f'only {opposed:.0%} of age TFs are opposed by ruxolitinib'


def test_ruxolitinib_suppresses_immune_pathways():
    """'suppressed multiple immune activation pathways in CD4+ cells, including TNF-alpha
    signaling via NF-kB, IL-2/STAT5 signaling, and interferon-alpha and interferon-gamma
    responses' (Extended Data Fig 1C)."""
    rux = _perturbation_shift('op', 'CD4T', 'Ruxolitinib')
    down = rux[rux['is_significant'] & (rux['slope'] < 0)].index
    terms = enriched_terms(down)
    expected = ['TNF-alpha Signaling via NF-kB', 'IL-2/STAT5 Signaling',
                'Interferon Alpha Response', 'Interferon Gamma Response']
    hits = [t for t in expected if has_term(terms, t)]
    print(f'\n[claim] ruxolitinib-suppressed CD4T TFs enriched for {expected} | actual hits: {hits} '
          f'(n_down={len(down)}; all enriched: {list(terms)[:8]})')
    assert len(hits) >= 2, f'only {hits} of the manuscript pathways are enriched among ruxolitinib-down TFs'


# JAK-STAT regulators named as consistent under both IL-10 and ruxolitinib (manuscript.md line 111)
JAK_STAT_REGULATORS = ['STAT1', 'STAT2', 'STAT3', 'IRF1', 'IRF2', 'IRF7', 'IRF9', 'NFKB2']


def test_il10_ruxolitinib_tf_concordance():
    """'high concordance between ruxolitinib- and IL-10 induced TF activity profiles ...
    key JAK-STAT pathway regulators, including STAT1, STAT2, STAT3, IRF1, IRF2, IRF7, IRF9,
    and NFKB2 showed consistent directional changes under both perturbations' (Extended Data Fig 2F)."""
    from scipy.stats import spearmanr
    il10 = _perturbation_shift('parsebioscience', 'CD4T', 'IL-10')
    rux = _perturbation_shift('op', 'CD4T', 'Ruxolitinib')
    common = il10.index.intersection(rux.index)
    rho = spearmanr(il10.loc[common, 'slope'], rux.loc[common, 'slope'])[0]
    conc = (np.sign(il10.loc[common, 'slope']) == np.sign(rux.loc[common, 'slope'])).mean()
    print(f'\n[claim] IL-10 and ruxolitinib TF profiles concordant (CD4T) | actual: rho={rho:.2f}, '
          f'{conc:.0%} same direction over {len(common)} TFs')
    assert rho > 0.3, f'IL-10 vs ruxolitinib TF-shift correlation is only rho={rho:.2f}'

    named = [t for t in JAK_STAT_REGULATORS if t in common]
    disagree = [f'{t} (IL-10 {il10.loc[t, "slope"]:+.2f} vs rux {rux.loc[t, "slope"]:+.2f})'
                for t in named if np.sign(il10.loc[t, 'slope']) != np.sign(rux.loc[t, 'slope'])]
    print(f'    named JAK-STAT regulators testable: {named}; disagreeing: {disagree or "none"}')
    assert len(named) >= 5, f'only {named} of {JAK_STAT_REGULATORS} present in both perturbation stats'
    assert len(disagree) <= 1, f'named JAK-STAT regulators move oppositely under the two perturbations: {disagree}'


# accelerating / rejuvenating cytokines named in the manuscript (Fig 4A, Supp Fig 3A)
ACCELERATING_CYTOKINES = ['IL-2', 'IL-4', 'IL-7', 'IL-15']
REJUVENATING_CYTOKINES = ['IL-10', 'IL-22', 'OSM']


def test_cytokine_acceleration_and_rejuvenation():
    """'Age-accelerating cytokines were predominantly pro-inflammatory mediators, including
    IL-2, IL-4, IL-7, and IL-15. In contrast, age-rejuvenating cytokines were enriched for
    anti-inflammatory and immunoregulatory factors, including IL-10, IL-22, and OSM.'"""
    for ct in ['CD4T', 'CD8T']:
        effects = {c: _perturbation_effect('parsebioscience', ct, 'PBS', c)
                   for c in ACCELERATING_CYTOKINES + REJUVENATING_CYTOKINES}
        for c, (p, delta) in effects.items():
            direction = 'accelerating' if c in ACCELERATING_CYTOKINES else 'rejuvenating'
            print(f'\n[claim] {c} ({direction}) {ct} | actual: {delta:+.1f}yr, p={p:.2e}')
        n_acc = sum(effects[c][1] > 0 for c in ACCELERATING_CYTOKINES)
        n_rej = sum(effects[c][1] < 0 for c in REJUVENATING_CYTOKINES)
        assert n_acc >= 3, (f'{ct}: only {n_acc}/{len(ACCELERATING_CYTOKINES)} of '
                            f'{ACCELERATING_CYTOKINES} increase predicted age')
        assert n_rej >= 2, (f'{ct}: only {n_rej}/{len(REJUVENATING_CYTOKINES)} of '
                            f'{REJUVENATING_CYTOKINES} decrease predicted age')


def test_type_i_ifn_cell_type_specific_effect():
    """'Type I interferons (IFN-beta and IFN-omega) ... accelerating predicted age in CD4+ T
    cells while reducing predicted age in CD8+ T cells'"""
    for ifn in ['IFN-beta', 'IFN-omega']:
        _, d_cd4 = _perturbation_effect('parsebioscience', 'CD4T', 'PBS', ifn)
        _, d_cd8 = _perturbation_effect('parsebioscience', 'CD8T', 'PBS', ifn)
        print(f'\n[claim] {ifn} CD4T up / CD8T down | actual: CD4T={d_cd4:+.1f}yr, CD8T={d_cd8:+.1f}yr')
        assert d_cd4 > d_cd8, f'{ifn}: expected a more age-accelerating effect in CD4T than CD8T'


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
    print(f'\n[claim] LPS (CXCL9) CD4T: +8yr, P=1e-16 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta > 0, f'manuscript reports LPS INCREASES predicted age by ~8yr; current pipeline gives {delta:+.1f}yr (wrong sign)'
    assert p < 0.05, f'expected a significant LPS effect, got p={p:.2e}'
    assert 3 < delta < 15, f'LPS effect {delta:+.1f}yr is far from the reported ~8 years'


def test_ruxolitinib_reduces_baseline_predicted_age_ex_vivo():
    """'Ruxolitinib reduced predicted age under baseline conditions (~2 years, P = 0.048)'"""
    config = get_config('CXCL9')
    obs = predict('CXCL9', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='RPMI', treatment='RPMI + ruxolitinib',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] Ruxolitinib (CXCL9) baseline CD4T: -2yr, P=0.048 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert p < 0.05, f'manuscript claims this reaches significance (P=0.048), got p={p:.2e}'


def test_ruxolitinib_attenuates_lps_induced_aging_ex_vivo():
    """'directionally consistent attenuation of the LPS-induced increase in predicted age
    (~2.5 years, P = 0.12)' -- manuscript notes this one did NOT reach significance."""
    config = get_config('CXCL9')
    obs = predict('CXCL9', 'CD4T')
    p, delta = run_mixed_effects_test(obs, ctr='LPS', treatment='LPS + ruxolitinib',
                                   target_variable='predicted_age', group_key='donor_id', config=config)
    print(f'\n[claim] Ruxolitinib (CXCL9) vs LPS CD4T: -2.5yr, P=0.12 (n.s.) | actual: '
          f'{delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to attenuate the LPS-induced increase, got {delta:+.1f}yr'
    assert 0.5 < -delta < 7, f'LPS-attenuation effect {delta:+.1f}yr is far from the reported ~2.5 years'


def test_stat1_batf_lps_up_ruxolitinib_down_ex_vivo():
    """'STAT1 and BATF, whose activities increase with age, were significantly induced by LPS
    and reduced by ruxolitinib' (Fig 4D)."""
    lps = _perturbation_shift('CXCL9', 'CD4T', 'LPS (ctr: RPMI)')
    rux = _perturbation_shift('CXCL9', 'CD4T', 'Ruxolitinib (ctr: LPS)')
    aging = age_slopes('CD4T')
    for tf in ['STAT1', 'BATF']:
        if tf not in lps.index or tf not in rux.index:
            pytest.fail(f'{tf} absent from the CXCL9 CD4T stats')
        print(f'\n[claim] ex vivo {tf}: up with age, induced by LPS, reduced by ruxolitinib | actual: '
              f'aging slope={aging.get(tf, float("nan")):+.3f}, '
              f'LPS={lps.loc[tf, "slope"]:+.3f} (FDR={lps.loc[tf, "p_value_adj"]:.1e}), '
              f'rux={rux.loc[tf, "slope"]:+.3f} (FDR={rux.loc[tf, "p_value_adj"]:.1e})')
        assert lps.loc[tf, 'slope'] > 0 and lps.loc[tf, 'p_value_adj'] < 0.05, (
            f'{tf} is not significantly induced by LPS')
        assert rux.loc[tf, 'slope'] < 0, f'{tf} is not reduced by ruxolitinib on top of LPS'


@pytest.mark.parametrize('comparison,expect_concordant', [
    ('LPS (ctr: RPMI)', True),            # LPS accelerates aging -> same direction as age
    ('Ruxolitinib (ctr: RPMI)', False),   # ruxolitinib rejuvenates -> opposite direction
    ('Ruxolitinib (ctr: LPS)', False),
])
def test_central_tfs_cxcl9_direction_vs_aging(comparison, expect_concordant):
    """Content of plots/condition/central_tfs_CD4T_CXCL9.png: for the most GRN-central
    age-associated CD4T TFs, LPS should move activity in the aging direction and ruxolitinib
    against it (Fig 4C-D)."""
    aging = age_slopes('CD4T')
    shift = _perturbation_shift('CXCL9', 'CD4T', comparison)
    # only TFs the plot marks as significantly shifted (starred) carry the claim
    tfs = [t for t in central_age_tfs('CD4T')
           if t in shift.index and shift.loc[t, 'p_value_adj'] < 0.05]
    conc = [t for t in tfs if np.sign(aging[t]) == np.sign(shift.loc[t, 'slope'])]
    frac = len(conc) / max(len(tfs), 1)
    print(f'\n[claim] CD4T central age TFs under {comparison}: '
          f'{"concordant" if expect_concordant else "opposite"} to aging | actual: '
          f'{len(conc)}/{len(tfs)} concordant ({frac:.0%}); ' +
          ', '.join(f'{t} age={aging[t]:+.2f} shift={shift.loc[t, "slope"]:+.2f}' for t in tfs))
    assert tfs, f'no central CD4T age TF is significantly shifted by {comparison}'
    if expect_concordant:
        assert frac > 0.6, f'{comparison} opposes aging in {len(tfs) - len(conc)}/{len(tfs)} central TFs'
    else:
        assert frac < 0.4, f'{comparison} follows aging in {len(conc)}/{len(tfs)} central TFs'


# ===========================================================================
# "Interpretable aging clocks" -- top-weighted clock features (Fig 3C, manuscript.md lines 75-77)
# ===========================================================================

def _clock_coefs(cell_type):
    return clock(cell_type).get_feature_importance()


def test_clock_top_gene_weights_match_age_trend(n_top=25):
    """'positively weighted genes tend to increase with age, whereas negatively weighted
    genes declined' (Fig 3C left). Checked as sign concordance between the ridge
    coefficient and the meta age slope of the same gene."""
    for ct in ['CD4T', 'CD8T']:
        top = _clock_coefs(ct).head(n_top)
        ge = ge_stats()
        ge = ge[ge['cell_type'] == ct].groupby('gene')['slope'].mean()
        merged = top[top['feature'].isin(ge.index)].copy()
        merged['age_slope'] = merged['feature'].map(ge)
        conc = (np.sign(merged['coefficient']) == np.sign(merged['age_slope'])).mean()
        print(f'\n[claim] {ct} top-{n_top} clock genes: weight sign matches age trend | '
              f'actual {conc:.0%} ({len(merged)}/{len(top)} genes testable)')
        assert len(merged) >= n_top * 0.5, f'{ct}: only {len(merged)}/{n_top} top clock genes have age stats'
        assert conc > 0.7, f'{ct} coefficient/age-slope sign concordance {conc:.0%}, expected a clear majority'


def test_senescence_markers_among_top_clock_features(n_top=200):
    """'Many of the highest-ranked features were established markers of immune aging and
    senescence, including CD70, CDKN2A/p16INK4a, and KLRC1/NKG2A' (Fig 3C)."""
    markers = ['CD70', 'CDKN2A', 'KLRC1']
    found = {}
    for ct in ['CD4T', 'CD8T']:
        ranked = _clock_coefs(ct)['feature'].tolist()
        found[ct] = [m for m in markers if m in ranked[:n_top]]
        ranks = {m: (ranked.index(m) + 1 if m in ranked else None) for m in markers}
        print(f'\n[claim] {ct} senescence markers in top-{n_top} clock features | ranks: {ranks}')
    hits = set(found['CD4T']) | set(found['CD8T'])
    assert hits, f'none of {markers} rank in the top {n_top} clock features of either T-cell clock'


def test_top_regulator_tfs_of_clock(n_features=5):
    """'In CD8+ T cells, the TFs exerting the strongest regulatory influence were JUN, KLF6,
    GATA3, MAF, and SOX4, while in CD4+ T cells, the top regulators were SOX4, IRF4, RORC,
    SCML4, and SATB1' (Fig 3C right). Recomputed with the same ULM-on-coefficients
    procedure as src/clock/run_exp_analysis.py::tf_act_analysis."""
    import anndata as ad
    import decoupler as dc
    from hira.src.utils.util import retrieve_net_consensus

    expected = {'CD8T': ['JUN', 'KLF6', 'GATA3', 'MAF', 'SOX4'],
                'CD4T': ['SOX4', 'IRF4', 'RORC', 'SCML4', 'SATB1']}
    for ct, exp in expected.items():
        coefs = _clock_coefs(ct)
        adata = ad.AnnData(np.asarray([coefs['coefficient'].astype(float).values]),
                           var=pd.DataFrame(index=coefs['feature'].values))
        dc.mt.ulm(adata, retrieve_net_consensus(cell_type=ct), tmin=5)
        score = pd.Series(adata.obsm['score_ulm'].values[0], index=adata.obsm['score_ulm'].columns)
        top = score.abs().sort_values(ascending=False).head(n_features).index.tolist()
        ranks = {t: (score.abs().rank(ascending=False)[t].astype(int) if t in score.index else None) for t in exp}
        overlap = set(top) & set(exp)
        print(f'\n[claim] {ct} top-{n_features} clock regulators {exp} | actual {top} '
              f'(overlap {len(overlap)}/{n_features}); expected-TF ranks: {ranks}')
        assert len(overlap) >= 2, f'{ct}: only {sorted(overlap)} of the manuscript top-{n_features} TFs recovered'


# ===========================================================================
# TF activity vs TF gene expression (Supplementary Fig 1D, manuscript.md line 63)
# ===========================================================================

def _tf_vs_ge(cell_type):
    """Per-TF age association from TF activity and from that TF's own gene expression."""
    tfa = all_stats()
    tfa = tfa[tfa['cell_type'] == cell_type]
    tfa = tfa.groupby('gene').agg(tfa_slope=('slope', 'mean'), tfa_p=('meta_p_adj', 'first'),
                                  tfa_sig=('is_significant', 'any'))
    ge = ge_stats()
    ge = ge[ge['cell_type'] == cell_type]
    ge = ge.groupby('gene').agg(ge_slope=('slope', 'mean'), ge_p=('meta_p_adj', 'first'),
                                ge_sig=('is_significant', 'any'))
    return tfa.join(ge, how='inner')


def test_tf_activity_adds_signal_beyond_expression():
    """'inferred TF activity ... can capture regulatory changes not apparent from TF
    expression alone' -- there must be a non-trivial set of TFs significant in activity but
    not in their own expression (Supplementary Fig 1D)."""
    for ct in ['CD4T', 'CD8T']:
        df = _tf_vs_ge(ct)
        act_sig = df[df['tfa_sig']]
        act_only = act_sig[~act_sig['ge_sig']]
        frac = len(act_only) / max(len(act_sig), 1)
        print(f'\n[claim] {ct} TF activity beyond expression | n_TFs={len(df)} '
              f'activity-significant={len(act_sig)} of which expression-non-significant='
              f'{len(act_only)} ({frac:.0%})')
        assert len(df) > 100, f'{ct}: only {len(df)} TFs have both activity and expression stats'
        assert len(act_sig) > 0, f'{ct}: no age-significant TFs in activity at all'
        assert frac > 0.3, (
            f'{ct}: only {len(act_only)}/{len(act_sig)} age-significant TFs are invisible in '
            f'their own expression -- activity adds little beyond expression here')


def test_satb1_gata3_concordant_nme2_activity_only():
    """'SATB1 and GATA3 showed concordant age-associated changes in both activity and
    expression, others, such as NME2 ... shifts in activity without detectable changes in
    expression' (Supplementary Fig 1D)."""
    df = _tf_vs_ge('CD4T')
    for tf in ['SATB1', 'GATA3']:
        if tf not in df.index:
            pytest.fail(f'{tf} absent from the CD4T activity/expression intersection')
        r = df.loc[tf]
        print(f'\n[claim] CD4T {tf} concordant activity+expression | '
              f'activity slope={r["tfa_slope"]:+.3f} (sig={r["tfa_sig"]}), '
              f'expression slope={r["ge_slope"]:+.3f} (sig={r["ge_sig"]})')
        assert r['tfa_sig'], f'{tf} not age-significant in TF activity'
        assert np.sign(r['tfa_slope']) == np.sign(r['ge_slope']), (
            f'{tf} activity ({r["tfa_slope"]:+.3f}) and expression ({r["ge_slope"]:+.3f}) '
            f'disagree in direction, manuscript claims they are concordant')
    if 'NME2' in df.index:
        r = df.loc['NME2']
        print(f'\n[claim] CD4T NME2 activity-only | activity slope={r["tfa_slope"]:+.3f} '
              f'(sig={r["tfa_sig"]}), expression slope={r["ge_slope"]:+.3f} (sig={r["ge_sig"]})')
        assert r['tfa_sig'], 'NME2 not age-significant in TF activity'
        assert not r['ge_sig'], (
            'NME2 IS significant in expression, manuscript claims activity changes '
            'without detectable expression change')
    else:
        print('\n[claim] CD4T NME2: absent from the activity/expression intersection')



if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
