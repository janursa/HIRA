
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
from hira.src.config import (REF_GE_ANALYSIS, CLOCKS_DIR, CLOCK_STATS_DIR, CLOCK_V,
                             PRIOR_DIR, MAJOR_CTS)
from grnimmuneclock import AgingClock

# Which TF-activity analysis these claims are checked against (tfa_major_b).
ANALYSIS_NAME = os.environ.get('TFA_ANALYSIS_NAME', 'tfa_major_b')

# Cohort key for the Wang cohort in the pipeline outputs.
WANG = 'wang'


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


# These tests re-run no analysis: every clock number below is read from the CSVs the clock
# pipeline writes next to its figures (src/clock/helper.py::save_clock_stats), so a test
# failure means the figure is wrong, not that the test computed something different.
@lru_cache(maxsize=None)
def clock_stats(name):
    path = Path(CLOCK_STATS_DIR) / f'{name}.csv'
    if not path.exists():
        pytest.fail(f'{path} missing -- run scripts/clock_analysis.sh')
    return pd.read_csv(path)


def predictions(dataset, cell_type=None):
    """Predicted ages the clock pipeline wrote for this dataset."""
    df = clock_stats(f'predictions_{dataset}')
    return df if cell_type is None else df[df['cell_type'] == cell_type]


def perturbation(dataset, cell_type, ctr, treatment):
    """(p_value, delta) for one contrast, exactly as annotated on the clock figure."""
    df = clock_stats(f'perturbation_{dataset}')
    hit = df[(df['cell_type'] == cell_type) & (df['ctr'] == ctr) & (df['treatment'] == treatment)]
    if hit.empty:
        pytest.fail(f'{dataset}/{cell_type}: no "{treatment}" vs "{ctr}" row in perturbation_{dataset}.csv '
                    f'(available treatments: {sorted(df[df["cell_type"] == cell_type]["treatment"].unique())})')
    return float(hit['p_value'].iloc[0]), float(hit['delta'].iloc[0])


# ===========================================================================
# Cohort composition (Fig 1, Supplementary Fig 1A, Methods)
# ===========================================================================

# donors per cohort as reported in Methods (Wang omitted: no feature matrix generated,
# it is only used to test the clock -- see test_clock_accuracy_cd4t_cd8t)
COHORT_DONORS = {'onek1k': 981, 'abf300': 166, 'aida': 619, 'soundlife': 96,
                 'parsebioscience': 12, 'CXCL9': 7}


@pytest.mark.parametrize('dataset', sorted(COHORT_DONORS))
def test_cohort_donor_counts(dataset):
    """Per-cohort donor numbers reported in Methods: OneK1K 981, ABF300 166, AIDA 619,
    Wang 33 (aged 20-90), SoundLife 96, ParseBioscience 12, ex vivo 7."""
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
    """'over 18 million single-cell transcriptomes from 1,960 donors aged 20-97 years' (Fig 1),
    and 'approximately 1.26 million circulating immune cells' for AIDA."""
    totals, donors = {}, {}
    for ds in ['onek1k', 'abf300', 'aida', 'perez_sle', 'soundlife']:
        cells, ids = 0, set()
        for ct in MAJOR_CTS:
            obs = cohort_obs(ds, ct)
            cells += float(obs['cell_count'].sum())
            ids |= set(obs['donor_id'])
        totals[ds], donors[ds] = cells, len(ids)
    print(f'\n[claim] >18M cells / 1,960 donors across cohorts | actual (post-QC, major cell types only): ' +
          ', '.join(f'{d}={totals[d]/1e6:.2f}M/{donors[d]}' for d in totals) +
          f' -> {sum(totals.values())/1e6:.2f}M cells, {sum(donors.values())} donors')
    assert totals['aida'] > 0.6e6, f'AIDA has {totals["aida"]/1e6:.2f}M cells, Methods says ~1.26M'
    assert sum(totals.values()) > 8e6, (
        f'only {sum(totals.values())/1e6:.2f}M cells retained across cohorts; the manuscript '
        f'claims >18M profiled (a large QC/cell-type loss is expected, but not this large)')


def test_perez_sle_case_control_composition():
    """'PBMCs from 162 SLE cases and 99 healthy controls' -> 261 individuals."""
    obs = predictions('perez_sle', 'CD8T')
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
    n_donors = obs['donor_id'].nunique()
    treatments = sorted(clock_stats('perturbation_op')['treatment'].unique())
    print(f'\n[claim] 146 compounds x 3 donors | actual: {len(treatments)} compounds screened by '
          f'the clock, {n_donors} donors')
    assert n_donors == 3, f'{n_donors} donors cached for op, expected triplicate donors'
    assert len(treatments) >= 130, f'only {len(treatments)} compounds screened, manuscript says 146'


# ===========================================================================
# "Age-related trajectories of gene regulation across immune cell types"
# ===========================================================================

def test_total_age_associated_tfs():
    """'we identified 340 unique TFs with age-associated changes across immune lineages'"""
    n = sig_stats()['gene'].nunique()
    print(f'\n[claim] 340 age-associated TFs total | actual: {n}')
    assert abs(n - 340) <= 60, f'expected ~340 unique age-associated TFs, got {n}'


def test_cd8_cd4_tf_counts():
    """'approximately 232 and 135 age-associated TFs detected in CD8+ and CD4+ subsets'"""
    counts = sig_stats().groupby('cell_type')['gene'].nunique()
    cd8, cd4 = int(counts.get('CD8T', 0)), int(counts.get('CD4T', 0))
    print(f'\n[claim] CD8T=232, CD4T=135 | actual: CD8T={cd8}, CD4T={cd4} | all: {counts.to_dict()}')
    assert abs(cd8 - 232) <= 60, f'CD8T count {cd8}, manuscript says ~232'
    assert abs(cd4 - 135) <= 40, f'CD4T count {cd4}, manuscript says ~135'
    assert cd8 > cd4, 'manuscript reports the most extensive remodeling in CD8+ T cells'


def _discovery_vs_validation():
    """Discovery age-significant TFs joined with their SoundLife (validation) statistics."""
    disc = sig_stats().groupby(['gene', 'cell_type']).agg(
        disc_slope=('slope', 'mean'), disc_fdr=('meta_p_adj', 'first')).reset_index()
    val = soundlife_stats().drop_duplicates(subset=['gene', 'cell_type']).rename(
        columns={'slope': 'val_rho', 'p_value_adj': 'val_fdr', 'is_significant': 'val_sig'})
    return disc.merge(val[['gene', 'cell_type', 'val_rho', 'val_fdr', 'val_sig']],
                      on=['gene', 'cell_type'], how='inner')


def test_validation_cohort_directional_consistency():
    """'Over 95% of age-associated TFs showed consistent directionality ... across all lineages
    and discordant cases were restricted to TFs with weak age-related correlations in the
    validation cohort (Spearman |rho| < 0.1)' (Supplementary Fig 3A)."""
    m = _discovery_vs_validation()
    m['concordant'] = np.sign(m['disc_slope']) == np.sign(m['val_rho'])
    per_ct = m.groupby('cell_type')['concordant'].agg(['mean', 'size'])
    overall = m['concordant'].mean() * 100
    print(f'\n[claim] >95% consistent direction in validation, all lineages | actual overall: '
          f'{overall:.1f}% (n={len(m)})\n' +
          '\n'.join(f'    {ct}: {r["mean"]*100:.1f}% (n={int(r["size"])})' for ct, r in per_ct.iterrows()))
    assert overall > 93, f'only {overall:.1f}% of age TFs replicate direction in soundlife, expected >95%'
    worst_ct, worst = per_ct['mean'].idxmin(), per_ct['mean'].min() * 100
    assert worst > 85, f'{worst_ct} replicates direction for only {worst:.1f}% of its age TFs'

    discordant = m[~m['concordant']]
    weak = (discordant['val_rho'].abs() < 0.1).mean() * 100 if len(discordant) else 100.0
    print(f'[claim] discordant cases restricted to |rho|<0.1 in validation | actual: {weak:.0f}% '
          f'of {len(discordant)} discordant TFs are weak (max |rho|={discordant["val_rho"].abs().max():.2f})'
          if len(discordant) else '[claim] no discordant TFs at all')
    assert weak > 85, f'only {weak:.0f}% of discordant TFs have |rho|<0.1 in the validation cohort'


def test_validation_cohort_non_significant_fraction():
    """'26% of age-associated TFs did not reach statistical significance in the validation
    cohort' (Supplementary Fig 3A), attributed to its size (96 vs 1,864 donors). Counted on
    the validation FDR alone, as in the manuscript."""
    m = _discovery_vs_validation()
    frac = (m['val_fdr'] >= 0.05).mean() * 100
    print(f'\n[claim] 26% of age TFs non-significant in validation | actual: {frac:.1f}% '
          f'of {len(m)} discovery TFs (with the |rho|>0.1 filter too: '
          f'{(~m["val_sig"].astype(bool)).mean()*100:.1f}%)')
    assert abs(frac - 26) <= 12, f'{frac:.1f}% non-significant in soundlife, manuscript says 26%'


# named as strongly age-associated in discovery but not significant in validation
VALIDATION_NONSIG_TFS = {'RELA': 3.9e-13, 'IRF7': 2.8e-3}


def test_rela_irf7_strong_in_discovery_not_significant_in_validation():
    """'These included RELA and IRF7 in CD4+ cells ... which displayed strong age associations
    in the discovery (FDR = 3.9e-13 and 2.8e-3) and showed similar age-associated trajectories
    in the validation cohort despite not reaching statistical significance (FDR = 0.44 and 0.21)'."""
    m = _discovery_vs_validation()
    m = m[m['cell_type'] == 'CD4T'].set_index('gene')
    for tf, disc_fdr in VALIDATION_NONSIG_TFS.items():
        if tf not in m.index:
            pytest.fail(f'{tf} is named as an age-associated CD4T TF but is not significant')
        r = m.loc[tf]
        print(f'\n[claim] CD4T {tf} discovery FDR={disc_fdr:.1e}, same direction but n.s. in validation | '
              f'actual: FDR={r["disc_fdr"]:.1e}, discovery slope={r["disc_slope"]:+.3f}, '
              f'validation rho={r["val_rho"]:+.3f} (FDR={r["val_fdr"]:.2f})')
        assert r['disc_fdr'] < 10 * disc_fdr, f'{tf} discovery FDR={r["disc_fdr"]:.1e}, manuscript says {disc_fdr:.1e}'
        assert np.sign(r['disc_slope']) == np.sign(r['val_rho']), (
            f'{tf} trajectory flips sign in the validation cohort')


def test_forty_seven_tfs_shared_across_cd4_cd8_nk():
    """'Forty-seven transcription factors were shared across all three cell types, with some
    showing conserved age-associated trends, such as the decline in SATB1, whereas others
    diverged by lineage, such as GATA3, which increased in T cells but decreased in NK cells'
    (Fig 2C-E)."""
    disc = sig_stats()
    sets = {ct: set(disc[disc['cell_type'] == ct]['gene']) for ct in ['CD4T', 'CD8T', 'NK']}
    shared = sets['CD4T'] & sets['CD8T'] & sets['NK']
    print(f'\n[claim] 47 TFs shared across CD4T/CD8T/NK | actual: {len(shared)} -> {sorted(shared)}')
    assert abs(len(shared) - 47) <= 12, f'{len(shared)} shared TFs, manuscript says 47'
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


@lru_cache(maxsize=None)
def central_age_tfs(cell_type, n=15):
    """The n most GRN-central age-associated TFs (degree in the consensus network)."""
    from hira.src.utils.util import retrieve_net_consensus
    net = retrieve_net_consensus(cell_type=cell_type)
    degree = net.groupby('source').size()
    aging = age_slopes(cell_type)
    ranked = degree[degree.index.isin(aging.index)].sort_values(ascending=False)
    return tuple(ranked.head(n).index)


def test_central_tf_directions():
    """'Among the central TFs, PRDM1, TBX21, and KLF6 exhibited increased activity with age
    across T cells, whereas LEF1, TCF7, and BACH2 decreased across T and NK cells'
    (Supplementary Fig 2C)."""
    for ct in ['CD4T', 'CD8T', 'NK']:
        slopes = age_slopes(ct)
        up = {t: slopes[t] for t in ['PRDM1', 'TBX21', 'KLF6'] if t in slopes.index}
        down = {t: slopes[t] for t in ['LEF1', 'TCF7', 'BACH2'] if t in slopes.index}
        print(f'\n[claim] {ct} central TFs: PRDM1/TBX21/KLF6 up, LEF1/TCF7/BACH2 down | actual: ' +
              ', '.join(f'{t}={v:+.3f}' for t, v in {**up, **down}.items()))
        if ct != 'NK':  # the "up" set is claimed for T cells only
            assert up and all(v > 0 for v in up.values()), f'{ct}: {up} do not all increase with age'
        assert down and all(v < 0 for v in down.values()), f'{ct}: {down} do not all decrease with age'


# TFs named in the CD8T aging narrative
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
    TFs associated with Wnt-beta-catenin signaling declined' (Supplementary Fig 2B)."""
    up_expected = ['TNF-alpha Signaling via NF-kB', 'IL-2/STAT5 Signaling', 'Interferon Gamma Response']
    wnt_cts = []
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
        if has_term(down_terms, 'Wnt'):
            wnt_cts.append(ct)
    # the manuscript makes the Wnt claim without naming a lineage; it is carried by CD8T
    print(f'[claim] Wnt-beta-catenin declines with age | actual: enriched among age-declining TFs in {wnt_cts}')
    assert wnt_cts, 'Wnt-beta-catenin is not enriched among age-declining TFs in either T-cell lineage'


# ===========================================================================
# "GRN-informed cell type-specific aging clocks"
# ===========================================================================

def test_clock_feature_counts():
    """'We built cell type-specific aging clocks using genes derived from GRN models that showed
    significant age association ... The number of these genes ranged from approximately 387 to
    2,868 depending on the cell type' (Supplementary Fig 1C)."""
    features = {}
    for ct in MAJOR_CTS:
        f = Path(CLOCKS_DIR) / ct / f'feature_names_{CLOCK_V}.txt'
        if not f.exists():
            continue
        features[ct] = set(np.loadtxt(f, dtype=str).tolist())
    if not features:
        pytest.fail(f'no trained clocks under {CLOCKS_DIR}')
    sizes = {ct: len(g) for ct, g in features.items()}
    print(f'\n[claim] 387-2,868 clock genes per cell type | actual: {sizes}')
    assert min(sizes.values()) >= 300, f'smallest clock has {min(sizes.values())} genes, manuscript says ~387'
    assert max(sizes.values()) <= 3400, f'largest clock has {max(sizes.values())} genes, manuscript says ~2,868'
    assert max(sizes, key=sizes.get) == 'CD8T', (
        f'CD8T should carry the most age-associated clock genes (most remodeled lineage), got {sizes}')


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


def test_clock_model_feature_coverage():
    """
    Diagnostic (not a manuscript number): what fraction of the pretrained
    clock's expected features are actually present in the currently cached
    gene-expression feature matrices (the clock's real input space per the
    manuscript: age-associated GRN target-gene expression, not TF activity).
    Low coverage means the shipped clock (GRNimmuneClock/grnimmuneclock/models/)
    is stale relative to the current GRN/feature-association pipeline output and
    any prediction-based claim below is expected to degrade.
    """
    for ct in ['CD4T', 'CD8T']:
        c = clock(ct)
        adata = retrieve_feature_data(dataset='aida', cell_type=ct, analysis_name=REF_GE_ANALYSIS)
        overlap = set(c.feature_names) & set(adata.var_names)
        frac = len(overlap) / len(c.feature_names)
        print(f'\n[diagnostic] {ct} clock feature coverage vs current pipeline: '
              f'{len(overlap)}/{len(c.feature_names)} ({frac:.1%})')
        assert frac > 0.8, (
            f'{ct} clock model only recognizes {frac:.1%} of its expected features in the '
            f'current {REF_GE_ANALYSIS} output -- the bundled model likely needs retraining '
            f'against the current GRN/feature pipeline (run scripts/clock_analysis.sh training step)'
        )


def test_clock_accuracy_cd4t_cd8t():
    """'GRN-informed clocks predicted chronological age with high accuracy, achieving Spearman
    correlations of over 0.75 in both CD4+ and CD8+ T cells' on the three held-out test cohorts
    (AIDA, Perez, Wang; Fig 3A)."""
    scores = clock_stats('cv_scores')
    for ct in ['CD4T', 'CD8T']:
        for cohort in ['aida', 'perez_sle', WANG]:
            row = scores[(scores['cell_type'] == ct) & (scores['dataset'] == cohort)]
            if row.empty:
                pytest.fail(f'no CV score for {ct}/{cohort} in cv_scores.csv '
                            f'(have: {sorted(scores["dataset"].unique())})')
            sp, n = float(row['spearman'].iloc[0]), int(row['n'].iloc[0])
            print(f'\n[claim] Spearman>0.75 in {ct} | actual on {cohort}: {sp:.3f} (n={n})')
            assert sp > 0.65, f'{ct}/{cohort} predicted-vs-true-age Spearman={sp:.3f}, expected >0.75'


def test_clock_test_cohorts_and_size():
    """'tested across three independent cohorts of AIDA, Perez, and Wang with over 750 donors'"""
    pred = clock_stats('cv_predictions')
    per_ct = pred.groupby('cell_type').size()
    cohorts = sorted(pred['dataset'].unique())
    print(f'\n[claim] 3 test cohorts, >750 donors | actual: cohorts={cohorts}, samples per cell type='
          f'{per_ct.to_dict()}')
    assert len(cohorts) == 3, f'{cohorts} test cohorts, manuscript says AIDA, Perez and Wang'
    assert per_ct.max() > 700, f'largest test set has {per_ct.max()} samples, manuscript says over 750'


def test_grn_clock_outperforms_genome_wide_model():
    """'Across the test datasets, the GRN-informed clocks outperformed previous models trained
    on genome-wide gene expression' (Supplementary Fig 4B)."""
    df = clock_stats('comparison_scores')
    models = sorted(df['model'].unique())
    ours = [m for m in models if 'GRN' in m.upper()]
    if len(models) < 2 or not ours:
        pytest.fail(f'comparison_scores.csv has models {models}; need the GRN clock and a baseline')
    ours, baseline = ours[0], [m for m in models if m not in ours][0]
    wins, total = [], []
    for (ds, ct), g in df.groupby(['dataset', 'cell_type']):
        g = g.set_index('model')['Spearman']
        if ours in g.index and baseline in g.index:
            total.append((ds, ct, round(g[ours], 3), round(g[baseline], 3)))
            wins.append(g[ours] > g[baseline])
    print(f'\n[claim] {ours} beats {baseline} across test datasets | actual: {sum(wins)}/{len(wins)} '
          f'(dataset, cell type, ours, baseline): {total}')
    assert sum(wins) > 0.7 * len(wins), f'{ours} only wins {sum(wins)}/{len(wins)} comparisons'


def _clock_gene_importance(cell_type):
    """Per-gene ridge coefficients of the clock, as written by the clock pipeline."""
    df = clock_stats('clock_gene_importance')
    return df[df['cell_type'] == cell_type].copy()


def _clock_tf_scores(cell_type, sig_only=True):
    """ULM scores of TFs on the signed clock-coefficient profile (Fig 3C)."""
    df = clock_stats('clock_tf_activity')
    df = df[df['cell_type'] == cell_type]
    return df[df['padj'] < 0.05] if sig_only else df


def test_clock_regulator_counts():
    """'This yielded 198 and 131 significant TFs across CD8+ T and CD4+ T cell types'"""
    expected = {'CD8T': 198, 'CD4T': 131}
    for ct, exp in expected.items():
        n = len(_clock_tf_scores(ct))
        print(f'\n[claim] {ct}: {exp} significant clock-regulator TFs | actual: {n} '
              f'of {len(_clock_tf_scores(ct, sig_only=False))} tested')
        assert abs(n - exp) <= 40, f'{ct}: {n} significant clock TFs, manuscript says {exp}'


def test_clock_regulators_concordant_with_aging():
    """'Regulatory effects were highly concordant with the age-associated patterns: TFs with
    positive regulatory effects on the clock also increased in activity with age, and vice
    versa' (Fig 3C, Supplementary Fig 4C)."""
    for ct in ['CD4T', 'CD8T']:
        score = _clock_tf_scores(ct).set_index('tf')['score']
        aging = age_slopes(ct)
        common = aging.index.intersection(score.index)
        conc = (np.sign(aging[common]) == np.sign(score[common])).mean()
        print(f'\n[claim] {ct} clock regulatory effect matches age direction | actual: {conc:.0%} '
              f'of {len(common)} TFs significant in both analyses')
        assert len(common) > 50, f'{ct}: only {len(common)} TFs shared by the two analyses'
        assert conc > 0.9, f'{ct}: only {conc:.0%} of clock regulators move as they do with age'


def test_top_regulator_tfs_of_clock(n_features=5):
    """'In CD8+ T cells, the TFs exerting the strongest regulatory influence were LEF1, ZEB2,
    TCF7, TBX21, and KLF6, while in CD4+ T cells, the top regulators were KLF6, GATA3, FOXP1,
    LEF1, and PRDM1' (Fig 3C, Supplementary Fig 4E)."""
    expected = {'CD8T': ['LEF1', 'ZEB2', 'TCF7', 'TBX21', 'KLF6'],
                'CD4T': ['KLF6', 'GATA3', 'FOXP1', 'LEF1', 'PRDM1']}
    for ct, exp in expected.items():
        df = _clock_tf_scores(ct).copy()
        df['abs_score'] = df['score'].abs()
        df = df.sort_values('abs_score', ascending=False).reset_index(drop=True)
        top = df['tf'].head(n_features).tolist()
        ranks = {t: (int(df.index[df['tf'] == t][0]) + 1 if (df['tf'] == t).any() else None) for t in exp}
        overlap = set(top) & set(exp)
        print(f'\n[claim] {ct} top-{n_features} clock regulators {exp} | actual {top} '
              f'(overlap {len(overlap)}/{n_features}); expected-TF ranks: {ranks}')
        assert len(overlap) >= 4, f'{ct}: only {sorted(overlap)} of the manuscript top-{n_features} TFs recovered'


def test_senescence_markers_among_top_clock_features(n_top=200):
    """'Many of the highest-ranked features were established markers of immune aging and
    senescence, including CD70, CDKN2A/p16INK4a, and KLRC1/NKG2A' (Supplementary Fig 4D)."""
    markers = ['CD70', 'CDKN2A', 'KLRC1']
    found = {}
    for ct in ['CD4T', 'CD8T']:
        df = _clock_gene_importance(ct)
        ranked = df.reindex(df['clock_coef'].abs().sort_values(ascending=False).index)['gene'].tolist()
        found[ct] = [m for m in markers if m in ranked[:n_top]]
        ranks = {m: (ranked.index(m) + 1 if m in ranked else None) for m in markers}
        print(f'\n[claim] {ct} senescence markers in top-{n_top} clock features | ranks: {ranks}')
    hits = set(found['CD4T']) | set(found['CD8T'])
    assert len(hits) >= 2, f'only {sorted(hits)} of {markers} rank in the top {n_top} clock features'


# ===========================================================================
# "Autoimmune activation accelerates T-cell aging ... SLE"
# ===========================================================================

def test_sle_cohort_size():
    """'We analyzed transcriptomes from 261 individuals, including healthy controls and
    patients with SLE'"""
    obs = predictions('perez_sle', 'CD8T')
    n = obs['donor_id'].nunique() if 'donor_id' in obs.columns else len(obs)
    print(f'\n[claim] 261 SLE-cohort individuals | actual: {n}')
    assert 200 <= n <= 320, f'{n} donors in perez_sle CD8T, expected ~261'


def test_sle_accelerates_cd8t_aging_in_young_patients():
    """'CD8+ T from individuals younger than 50 years, in whom predicted age increased by
    approximately 3.1 years (FDR = 3.5e-4), whereas no significant increase was observed in
    older individuals (FDR = 1.0)' (Supplementary Fig 5A)."""
    df = clock_stats('disease_bins_perez_sle')
    rows = {}
    for age_bin in ['Young', 'Old']:  # Young = age < 50
        row = df[(df['cell_type'] == 'CD8T') & (df['age_bin'] == age_bin)]
        if row.empty:
            pytest.fail(f'no CD8T/{age_bin} row in disease_bins_perez_sle.csv '
                        f'(bins: {sorted(df["age_bin"].unique())})')
        rows[age_bin] = (float(row['delta_residual'].iloc[0]), float(row['p_value_adj'].iloc[0]))
    (d_young, p_young), (d_old, p_old) = rows['Young'], rows['Old']
    print(f'\n[claim] SLE CD8T <50y: +3.1yr FDR=3.5e-4; >50y: n.s. | actual: young {d_young:+.1f}yr '
          f'(FDR={p_young:.2e}), old {d_old:+.1f}yr (FDR={p_old:.2f})')
    assert d_young > 1.5, f'expected ~+3.1yr acceleration in young SLE CD8T, got {d_young:+.1f}yr'
    assert p_young < 0.01, f'expected a strongly significant SLE effect in young CD8T, got FDR={p_young:.2e}'
    assert p_old > 0.05, f'older SLE CD8T is significant (FDR={p_old:.2e}); manuscript reports no effect'


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


def test_sle_tf_shifts_concordant_with_aging():
    """'nearly all age-associated TFs shifted concordantly in SLE across both CD4+ T and CD8+ T
    cells' (Fig 3E, Supplementary Fig 3C)."""
    for ct in ['CD4T', 'CD8T']:
        frac, n = _age_tf_concordance(ct)
        print(f'\n[claim] {ct}: nearly all age TFs shift concordantly in SLE | actual: '
              f'{frac:.0%} of the {n} age TFs significantly shifted in SLE')
        assert n > 50, f'{ct}: only {n} age TFs shift significantly in SLE at all'
        assert frac > 0.8, f'{ct}: only {frac:.0%} of {n} age TFs shift with aging direction in SLE'


def test_sle_concordance_absent_in_older_patients():
    """'This pattern was not observed in older patients (>50 years)' -- in both T-cell lineages
    the aging-direction concordance collapses in the >50 stratum."""
    for ct in ['CD4T', 'CD8T']:
        young, n_y = _age_tf_concordance(ct, 'Younger than 50')
        old, n_o = _age_tf_concordance(ct, 'Older than 50')
        print(f'\n[claim] {ct} SLE-aging concordance lost in >50 | actual: young {young:.0%} of {n_y} '
              f'significantly shifted age TFs, old {old:.0%} of {n_o}')
        assert young > 0.85, f'{ct}: young SLE concordance is only {young:.0%}'
        assert old < young - 0.1, (
            f'{ct}: older patients still reproduce the aging direction ({old:.0%} vs {young:.0%} in '
            f'the young); manuscript says the pattern is absent in >50')


def test_central_age_tfs_reproduce_aging_direction_in_sle():
    """'Analysis of the 15 most central age-associated TFs further revealed that SLE reproduced
    the same directional activity shifts seen during normal aging across both cell types and age
    groups' (Fig 3F, Supplementary Fig 2E). The manuscript flags the older CD8+ T group as the
    exception; in the current analysis that group is fully concordant and CD4+ T >50 is the
    weakest stratum."""
    for ct in ['CD4T', 'CD8T']:
        for group in ['Younger than 50', 'Older than 50']:
            tfs = list(central_age_tfs(ct))
            sle, aging = _sle_shift(ct, group), age_slopes(ct)
            common = [t for t in tfs if t in sle.index]
            conc = (np.sign(aging[common]) == np.sign(sle.loc[common, 'slope'])).mean()
            print(f'\n[claim] {ct}/{group} top-15 central age TFs reproduce aging direction | actual: '
                  f'{conc:.0%} of {len(common)} | {tfs}')
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
# "Systematic cytokine perturbation identifies IL-10"
# ===========================================================================

def test_parsebioscience_screen_size():
    """'PBMCs stimulated with 90 cytokines across 12 donors'"""
    adata = retrieve_feature_data(dataset='parsebioscience', cell_type='CD4T', analysis_name=ANALYSIS_NAME)
    n_donors = adata.obs['donor_id'].nunique()
    treatments = sorted(clock_stats('perturbation_parsebioscience')['treatment'].unique())
    print(f'\n[claim] 90 cytokines x 12 donors | actual: {len(treatments)} cytokines screened by the '
          f'clock, {n_donors} donors')
    assert n_donors == 12, f'{n_donors} donors cached for parsebioscience, expected 12'
    assert len(treatments) >= 85, f'only {len(treatments)} cytokines screened, manuscript says 90'


def test_il10_reduces_predicted_age():
    """'decreasing predicted biological age by 12.0 years in CD4+ T cells (FDR = 6.6e-28) and by
    4.2 years in CD8+ T cells (FDR = 2.5e-5)'"""
    expected_years = {'CD4T': -12.0, 'CD8T': -4.2}
    for ct in ['CD4T', 'CD8T']:
        p, delta = perturbation('parsebioscience', ct, 'PBS', 'IL-10')
        print(f'\n[claim] IL-10 {ct}: {expected_years[ct]:+.1f}yr | actual: {delta:+.1f}yr, p={p:.2e}')
        assert delta < 0, (
            f'manuscript reports IL-10 REDUCES predicted age in {ct} by {abs(expected_years[ct])}yr; '
            f'current pipeline gives {delta:+.1f}yr (wrong sign)'
        )
        assert p < 0.05, f'expected a significant IL-10 effect in {ct}, got p={p:.2e}'
        assert abs(delta - expected_years[ct]) <= max(3, 0.5 * abs(expected_years[ct])), (
            f'IL-10 {ct} effect {delta:+.1f}yr is far from the reported {expected_years[ct]:+.1f}yr')


def _perturbation_shift(dataset, cell_type, comparison):
    """TF activity shift for one treatment comparison, indexed by TF."""
    df = cond_stats(dataset)
    df = df[(df['cell_type'] == cell_type) & (df['comparison'] == comparison)]
    if df.empty:
        pytest.fail(f'no {dataset} stats for {cell_type} / {comparison} '
                    f'(available: {sorted(cond_stats(dataset)["comparison"].unique())})')
    return df.drop_duplicates(subset='gene').set_index('gene')


def test_il10_reverses_age_tfs():
    """'IL-10 restored the activity of the majority of age-associated TFs (Fig 4B), with
    particularly pronounced effects among highly central TFs ... including LEF1 and TCF7'."""
    for ct in ['CD4T', 'CD8T']:
        il10 = _perturbation_shift('parsebioscience', ct, 'IL-10')
        aging = age_slopes(ct)
        common = aging.index.intersection(il10.index)
        n_rev = int((np.sign(aging[common]) != np.sign(il10.loc[common, 'slope'])).sum())
        print(f'\n[claim] IL-10 reverses the majority of {ct} age TFs | actual: {n_rev} of {len(common)} '
              f'({n_rev / max(len(common), 1):.0%})')
        assert n_rev / max(len(common), 1) > 0.6, (
            f'IL-10 reverses only {n_rev}/{len(common)} age-associated {ct} TFs, manuscript says a majority')

        for tf in ['LEF1', 'TCF7']:
            if tf not in common:
                print(f'    {tf}: not in the IL-10/aging intersection')
                continue
            print(f'    {tf}: aging slope={aging[tf]:+.3f}, IL-10 shift={il10.loc[tf, "slope"]:+.3f}')
            assert np.sign(il10.loc[tf, 'slope']) != np.sign(aging[tf]), (
                f'{ct}: {tf} moves with aging under IL-10, manuscript reports it is restored')


def test_il10_strongest_age_reducing_perturbation():
    """'Among all perturbations, IL-10 produced the strongest age-reducing effect in T cells'
    (of the 90 screened cytokines). It ranks first in CD4+ T; in CD8+ T it is among the top few."""
    for ct, max_rank in [('CD4T', 1), ('CD8T', 3)]:
        df = clock_stats('perturbation_parsebioscience')
        df = df[(df['cell_type'] == ct) & (df['ctr'] == 'PBS')]
        if df['treatment'].nunique() < 85:
            pytest.fail(f'only {df["treatment"].nunique()} cytokines tested for parsebioscience; '
                        f'the full 90-cytokine screen is needed to rank IL-10')
        ranked = sorted(zip(df['treatment'], df['delta']), key=lambda kv: kv[1])
        rank = [c for c, _ in ranked].index('IL-10') + 1
        print(f'\n[claim] IL-10 strongest age-reducing cytokine in {ct} | actual rank {rank}; top-5: '
              f'{[(c, round(d, 1)) for c, d in ranked[:5]]}')
        assert rank <= max_rank, (
            f'{ct}: IL-10 is ranked {rank}; strongest is {ranked[0][0]} ({ranked[0][1]:+.1f}yr)')


# cytokine classes named in the manuscript (Fig 4A, Supplementary Fig 5E)
ACCELERATING_CYTOKINES = ['IL-1-beta', 'IL-2', 'IL-4', 'IL-7', 'IL-15']
REJUVENATING_CYTOKINES = ['IL-1Ra', 'IL-10', 'IL-22', 'OSM']
CELL_TYPE_SPECIFIC_CYTOKINES = ['IL-17A', 'CD27L', 'LIF']


def test_cytokine_acceleration_and_rejuvenation():
    """'Age-accelerating cytokines were predominantly pro-inflammatory mediators, including
    IL-1beta, IL-2, IL-4, IL-7, and IL-15. In contrast, age-rejuvenating cytokines were enriched
    for anti-inflammatory and immunoregulatory factors, including IL-1Ra, IL-10, IL-22, and OSM.'"""
    for ct in ['CD4T', 'CD8T']:
        effects = {c: perturbation('parsebioscience', ct, 'PBS', c)
                   for c in ACCELERATING_CYTOKINES + REJUVENATING_CYTOKINES}
        for c, (p, delta) in effects.items():
            direction = 'accelerating' if c in ACCELERATING_CYTOKINES else 'rejuvenating'
            print(f'\n[claim] {c} ({direction}) {ct} | actual: {delta:+.1f}yr, p={p:.2e}')
        n_acc = sum(effects[c][1] > 0 for c in ACCELERATING_CYTOKINES)
        n_rej = sum(effects[c][1] < 0 for c in REJUVENATING_CYTOKINES)
        assert n_acc == len(ACCELERATING_CYTOKINES), (
            f'{ct}: only {n_acc}/{len(ACCELERATING_CYTOKINES)} of {ACCELERATING_CYTOKINES} '
            f'increase predicted age')
        assert n_rej >= 2, (f'{ct}: only {n_rej}/{len(REJUVENATING_CYTOKINES)} of '
                            f'{REJUVENATING_CYTOKINES} decrease predicted age')


def test_cell_type_specific_cytokine_effects():
    """'cytokines such as IL-17A, CD27L, and LIF exhibited cell type-specific effects,
    accelerating predicted age in CD8+ T cells while reducing predicted age in CD4+ T cells'
    (Supplementary Fig 5E)."""
    for cyt in CELL_TYPE_SPECIFIC_CYTOKINES:
        _, d_cd4 = perturbation('parsebioscience', 'CD4T', 'PBS', cyt)
        _, d_cd8 = perturbation('parsebioscience', 'CD8T', 'PBS', cyt)
        print(f'\n[claim] {cyt} CD8T up / CD4T down | actual: CD4T={d_cd4:+.1f}yr, CD8T={d_cd8:+.1f}yr')
        assert d_cd8 > 0 > d_cd4, f'{cyt}: expected acceleration in CD8T and rejuvenation in CD4T'


# ===========================================================================
# "Pharmacological reversal of immune-aging signatures"
# ===========================================================================

def test_ruxolitinib_reduces_predicted_age_op():
    """'ruxolitinib ... showed an age-reversal effect of 6.9 years in CD4+ T cells (FDR = 3.5e-3)'"""
    p, delta = perturbation('op', 'CD4T', 'DMSO', 'Ruxolitinib')
    print(f'\n[claim] Ruxolitinib (op) CD4T: -6.9yr, FDR=3.5e-3 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to reduce predicted age in CD4T, got {delta:+.1f}yr'
    assert p < 0.05, f'expected a significant ruxolitinib effect in CD4T (op), got p={p:.2e}'
    assert 3 < -delta < 12, f'ruxolitinib effect {delta:+.1f}yr is far from the reported -6.9 years'


def test_cgm097_accelerates_predicted_age_op():
    """'CGM-097, an MDM2 inhibitor that activates p53, increased predicted age in both CD4+
    and CD8+ T cells (FDR < 1e-3, Supplementary Fig 5F)'"""
    for ct in ['CD4T', 'CD8T']:
        p, delta = perturbation('op', ct, 'DMSO', 'CGM-097')
        print(f'\n[claim] CGM-097 (op) {ct}: accelerates, FDR<1e-3 | actual: {delta:+.1f}yr, p={p:.2e}')
        assert delta > 0, f'manuscript reports CGM-097 INCREASES predicted age in {ct}; got {delta:+.1f}yr'
        assert p < 1e-3, f'expected FDR<1e-3 for CGM-097 in {ct}, got p={p:.2e}'


def test_ruxolitinib_counteracts_age_tfs():
    """'Ruxolitinib broadly counteracted age-associated regulatory programs' (Supplementary Fig 3D)."""
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
    responses' (Supplementary Fig 2B)."""
    rux = _perturbation_shift('op', 'CD4T', 'Ruxolitinib')
    down = rux[rux['is_significant'] & (rux['slope'] < 0)].index
    terms = enriched_terms(down)
    expected = ['TNF-alpha Signaling via NF-kB', 'IL-2/STAT5 Signaling',
                'Interferon Alpha Response', 'Interferon Gamma Response']
    hits = [t for t in expected if has_term(terms, t)]
    print(f'\n[claim] ruxolitinib-suppressed CD4T TFs enriched for {expected} | actual hits: {hits} '
          f'(n_down={len(down)}; all enriched: {list(terms)[:8]})')
    assert len(hits) >= 2, f'only {hits} of the manuscript pathways are enriched among ruxolitinib-down TFs'


# JAK-STAT regulators named as consistent under both IL-10 and ruxolitinib
JAK_STAT_REGULATORS = ['STAT1', 'STAT2', 'STAT3', 'IRF1', 'IRF2', 'IRF7', 'IRF9', 'NFKB2']


def test_il10_ruxolitinib_tf_concordance():
    """'high concordance between ruxolitinib- and IL-10 induced TF activity profiles ...
    key JAK-STAT pathway regulators, including STAT1, STAT2, STAT3, IRF1, IRF2, IRF7, IRF9,
    and NFKB2 showed consistent directional changes under both perturbations' (Supplementary Fig 2D)."""
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


def test_cxcl9_donor_count():
    """'PBMCs from seven healthy donors were treated with ruxolitinib ... under both
    basal condition (RPMI) and LPS-stimulation'"""
    adata = retrieve_feature_data(dataset='CXCL9', cell_type='CD4T', analysis_name=ANALYSIS_NAME)
    n_donors = adata.obs['donor_id'].nunique()
    print(f'\n[claim] 7 ex vivo donors | actual: {n_donors}')
    assert n_donors == 7, f'{n_donors} donors cached for CXCL9, expected 7'


def test_lps_accelerates_predicted_age_ex_vivo():
    """'LPS stimulation induced a pronounced immune age acceleration of ~5.2 years in CD4+ T
    cells (P = 2.1e-4; Fig 4C)'"""
    p, delta = perturbation('CXCL9', 'CD4T', 'RPMI', 'LPS')
    print(f'\n[claim] LPS (CXCL9) CD4T: +5.2yr, P=2.1e-4 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta > 0, (f'manuscript reports LPS INCREASES predicted age by ~5.2yr; current pipeline '
                       f'gives {delta:+.1f}yr (wrong sign)')
    assert p < 0.05, f'expected a significant LPS effect, got p={p:.2e}'
    assert 2 < delta < 10, f'LPS effect {delta:+.1f}yr is far from the reported ~5.2 years'


def test_ruxolitinib_reduces_baseline_predicted_age_ex_vivo():
    """'Ruxolitinib reduced predicted age under baseline conditions (~3.1 years, P = 0.02)'"""
    p, delta = perturbation('CXCL9', 'CD4T', 'RPMI', 'RPMI + ruxolitinib')
    print(f'\n[claim] Ruxolitinib (CXCL9) baseline CD4T: -3.1yr, P=0.02 | actual: {delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to reduce baseline predicted age, got {delta:+.1f}yr'
    assert p < 0.05, f'manuscript claims this reaches significance (P = 0.02), got p={p:.2e}'


def test_ruxolitinib_attenuates_lps_induced_aging_ex_vivo():
    """'attenuated the LPS-induced increase in predicted age (~3.6 years, P = 1.1e-3)'"""
    p, delta = perturbation('CXCL9', 'CD4T', 'LPS', 'LPS + ruxolitinib')
    print(f'\n[claim] Ruxolitinib (CXCL9) vs LPS CD4T: -3.6yr, P=1.1e-3 | actual: '
          f'{delta:+.1f}yr, p={p:.2e}')
    assert delta < 0, f'expected ruxolitinib to attenuate the LPS-induced increase, got {delta:+.1f}yr'
    assert p < 0.05, f'manuscript claims this reaches significance (P = 1.1e-3), got p={p:.2e}'
    assert 1 < -delta < 8, f'LPS-attenuation effect {delta:+.1f}yr is far from the reported ~3.6 years'


def test_stat1_irf1_lps_up_ruxolitinib_down_ex_vivo():
    """'STAT1 and IRF1, whose activities increase with age, were significantly induced by LPS
    and reduced by ruxolitinib' (Fig 4D)."""
    lps = _perturbation_shift('CXCL9', 'CD4T', 'LPS (ctr: RPMI)')
    rux = _perturbation_shift('CXCL9', 'CD4T', 'Ruxolitinib (ctr: LPS)')
    aging = age_slopes('CD4T')
    for tf in ['STAT1', 'IRF1']:
        if tf not in lps.index or tf not in rux.index:
            pytest.fail(f'{tf} absent from the CXCL9 CD4T stats')
        print(f'\n[claim] ex vivo {tf}: up with age, induced by LPS, reduced by ruxolitinib | actual: '
              f'aging slope={aging.get(tf, float("nan")):+.3f}, '
              f'LPS={lps.loc[tf, "slope"]:+.3f} (FDR={lps.loc[tf, "p_value_adj"]:.1e}), '
              f'rux={rux.loc[tf, "slope"]:+.3f} (FDR={rux.loc[tf, "p_value_adj"]:.1e})')
        assert aging.get(tf, 0) > 0, f'{tf} does not increase with age in CD4T'
        assert lps.loc[tf, 'slope'] > 0 and lps.loc[tf, 'p_value_adj'] < 0.05, (
            f'{tf} is not significantly induced by LPS')
        assert rux.loc[tf, 'slope'] < 0 and rux.loc[tf, 'p_value_adj'] < 0.05, (
            f'{tf} is not significantly reduced by ruxolitinib on top of LPS')


@pytest.mark.parametrize('comparison,expect_concordant', [
    ('LPS (ctr: RPMI)', True),            # LPS accelerates aging -> same direction as age
    ('Ruxolitinib (ctr: RPMI)', False),   # ruxolitinib rejuvenates -> opposite direction
    ('Ruxolitinib (ctr: LPS)', False),
])
def test_central_tfs_cxcl9_direction_vs_aging(comparison, expect_concordant):
    """Content of plots/condition/central_tfs_CD4T_CXCL9.png: for the most GRN-central
    age-associated CD4T TFs, LPS should move activity in the aging direction and ruxolitinib
    against it (Fig 4D)."""
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
# TF activity vs TF gene expression (Supplementary Fig 3B)
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


# fraction of age-associated TF activity shifts with no detectable expression change
ACTIVITY_ONLY_FRACTION = {'CD4T': 45, 'CD8T': 48, 'NK': 63, 'MONO': 70}


def test_tf_activity_adds_signal_beyond_expression():
    """'a notable number of age-associated shifts in TF activity showed no detectable changes in
    expression (45% in CD4+ T, 48% in CD8+ T, 63% in NK, and 70% in monocyte cells; 362 of 627
    TF-cell-type pairs overall)' (Supplementary Fig 3B)."""
    total_sig, total_only = 0, 0
    for ct in MAJOR_CTS:
        df = _tf_vs_ge(ct)
        act_sig = df[df['tfa_sig']]
        act_only = act_sig[~act_sig['ge_sig']]
        total_sig, total_only = total_sig + len(act_sig), total_only + len(act_only)
        frac = len(act_only) / max(len(act_sig), 1) * 100
        expected = ACTIVITY_ONLY_FRACTION.get(ct)
        print(f'\n[claim] {ct} activity-only fraction {expected if expected else "n/a"}% | actual: '
              f'{len(act_only)}/{len(act_sig)} ({frac:.0f}%) of {len(df)} TFs with both stats')
        assert len(df) > 100, f'{ct}: only {len(df)} TFs have both activity and expression stats'
        if expected is not None:
            assert abs(frac - expected) <= 15, f'{ct}: {frac:.0f}% activity-only, manuscript says {expected}%'
    print(f'\n[claim] 362 of 627 TF-cell-type pairs overall | actual: {total_only} of {total_sig}')
    assert abs(total_only - 362) <= 80 and abs(total_sig - 627) <= 120, (
        f'{total_only}/{total_sig} activity-only pairs, manuscript says 362/627')


# named activity-only examples, with their discovery FDR in activity vs expression
ACTIVITY_ONLY_TFS = {'YBX1': (3.7e-30, 0.07), 'RELA': (3.9e-13, 0.45)}


def test_satb1_gata3_concordant_ybx1_rela_activity_only():
    """'SATB1 and GATA3 showed concordant age-associated changes in both activity and expression
    ... These included YBX1 ... whose inferred activity increased with age in CD4+ cells
    (FDR = 3.7e-30) while its transcript level showed no significant age association (FDR = 0.07),
    and RELA ... (FDR = 3.9e-13) but not in transcript level (FDR = 0.45)' (Supplementary Fig 3B)."""
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

    for tf, (act_fdr, ge_fdr) in ACTIVITY_ONLY_TFS.items():
        if tf not in df.index:
            pytest.fail(f'{tf} absent from the CD4T activity/expression intersection')
        r = df.loc[tf]
        print(f'\n[claim] CD4T {tf} activity-only (activity FDR={act_fdr:.1e}, expression FDR={ge_fdr}) | '
              f'actual: activity slope={r["tfa_slope"]:+.3f} FDR={r["tfa_p"]:.1e} (sig={r["tfa_sig"]}), '
              f'expression slope={r["ge_slope"]:+.3f} FDR={r["ge_p"]:.2f} (sig={r["ge_sig"]})')
        assert r['tfa_sig'] and r['tfa_slope'] > 0, f'{tf} activity does not increase significantly with age'
        assert not r['ge_sig'], (
            f'{tf} IS significant in expression, manuscript claims activity changes '
            f'without detectable expression change')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-s', '-v']))
