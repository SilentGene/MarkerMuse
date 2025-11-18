#!/usr/bin/env python3
"""
MarkerMuse: select best SingleM marker(s) for alpha-diversity calculation.

Usage examples:
    python markermuse.py --input_folder /path/to/otu_files --otu_suf _singlem.otu.tsv
    python markermuse.py --input_folder /path/to/otu_files --otu_suf _singlem.otu.tsv --taxprof _singlem.tax.tsv

Outputs (by default, written to folder `marker_muse_output`):
    - marker_scores.tsv         : per-marker raw metrics, normalized metrics, score (desc sorted), rank (first column)
    - marker_scores_heatmap.png : heatmap of normalized metrics across markers
    - marker_scores_topbar.png  : bar plot of top 20 marker scores

Dependencies: pandas, numpy, scipy
(install with `pip install pandas numpy scipy`)

Notes:
 - Input: folder containing per-sample SingleM OTU TSVs (tab-separated) whose filenames end with the provided suffix.
 - Each OTU TSV must contain columns: gene, sample, sequence, num_hits, coverage, taxonomy
 - In data-driven mode, the taxonomic-profile (taxprof) file produced by SingleM is required. That file must be tab-separated and contain columns: sample, coverage, taxonomy.
 - Optional: provide --metadata <meta.tsv> with columns at least: sample, group to enable group-discrimination metric (ANOVA F on Shannon).

Algorithm implemented (brief): parse all OTU files -> build per-marker sample x OTU count matrices -> compute per-sample alpha (richness, Shannon, Simpson) -> compute per-marker metrics (mean richness, presence_rate, singleton_rate, bootstrap CV of Shannon, rarefaction slope, discriminability if groups provided, and representativeness vs taxprof if provided) -> normalize metrics -> compute score via weighted sum (expert preset or data-driven weights fitted to predict taxprof representativeness) -> output ranked markers + plots

"""

import argparse
import os
import glob
import sys
import math
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, entropy
from scipy import stats


# ----------------------------- utility functions -----------------------------

def safe_read_tsv(path):
    try:
        return pd.read_csv(path, sep="\t", header=0, dtype=str)
    except Exception:
        return pd.read_csv(path, sep="\t", header=0, encoding='latin1')


def shannon_from_counts(counts):
    # counts: 1D numpy array of non-negative ints
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / float(total)
    return float(-np.sum(p * np.log(p)))


def simpson_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / float(total)
    return float(1.0 - np.sum(p * p))


# ----------------------------- parsing functions ------------------------------

def parse_otu_files(input_folder, otu_suf):
    """
    Read all files in input_folder matching *{otu_suf} and aggregate into data structures.

    Returns:
      markers: set of marker ids
      samples: sorted list of sample ids
      marker_sample_otu_counts: dict marker -> dict sample -> Counter(seq -> num_hits)
      seq_taxonomy: dict (marker, sequence) -> taxonomy string (first encountered)
    """
    pattern = os.path.join(input_folder, f"*{otu_suf}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found with pattern {pattern}")

    marker_sample_otu_counts = defaultdict(lambda: defaultdict(Counter))
    seq_taxonomy = {}
    sample_names = []

    for p in files:
        df = safe_read_tsv(p)
        cols = [c.strip() for c in df.columns]
        # normalize column names
        df.columns = cols
        # require columns
        required = {'gene', 'sample', 'sequence', 'num_hits', 'taxonomy'}
        if not required.issubset(set(cols)):
            raise ValueError(f"File {p} missing required columns. Found columns: {cols}")
        # get sample name: prefer value in 'sample' column (should be identical per file)
        sample_vals = df['sample'].unique()
        if len(sample_vals) != 1:
            # fallback: try to infer from filename
            sample = os.path.basename(p).replace(otu_suf, '')
        else:
            sample = sample_vals[0]
        sample_names.append(sample)

        # process rows
        for _, row in df.iterrows():
            marker = row['gene']
            seq = row['sequence']
            try:
                cnt = int(float(row['num_hits']))
            except Exception:
                try:
                    cnt = int(row['num_hits'])
                except Exception:
                    cnt = 0
            tax = row.get('taxonomy', '')
            if cnt > 0:
                marker_sample_otu_counts[marker][sample][seq] += cnt
                key = (marker, seq)
                if key not in seq_taxonomy:
                    seq_taxonomy[key] = tax
    samples = sorted(set(sample_names))
    markers = sorted(marker_sample_otu_counts.keys())
    return markers, samples, marker_sample_otu_counts, seq_taxonomy


# ------------------------- per-marker metric calculations --------------------

def build_marker_count_df(marker, samples, marker_sample_otu_counts):
    # build DataFrame: rows = samples, cols = OTU sequences, values = counts
    otus = set()
    for s in samples:
        otus.update(marker_sample_otu_counts[marker].get(s, {}).keys())
    otus = sorted(otus)
    if len(otus) == 0:
        # empty
        return pd.DataFrame(0, index=samples, columns=[]).astype(int)
    mat = np.zeros((len(samples), len(otus)), dtype=int)
    for i, s in enumerate(samples):
        counter = marker_sample_otu_counts[marker].get(s, {})
        for j, otu in enumerate(otus):
            mat[i, j] = counter.get(otu, 0)
    df = pd.DataFrame(mat, index=samples, columns=otus)
    return df


def compute_alpha_metrics(count_df):
    # count_df: samples x otus DataFrame
    samples = list(count_df.index)
    richness = np.array((count_df > 0).sum(axis=1)).astype(int)
    shannon = []
    simpson = []
    for i in range(count_df.shape[0]):
        counts = count_df.iloc[i, :].values.astype(int)
        shannon.append(shannon_from_counts(counts))
        simpson.append(simpson_from_counts(counts))
    shannon = np.array(shannon)
    simpson = np.array(simpson)
    return {'richness': richness, 'shannon': shannon, 'simpson': simpson}


def bootstrap_cv_alpha(counts_row, metric_fn, n_boot=100):
    # counts_row: 1D array of counts for a sample
    total = int(counts_row.sum())
    if total <= 0:
        return np.nan
    probs = counts_row / float(total)
    # if only one OTU non-zero => zero variation
    if np.count_nonzero(counts_row) <= 1:
        return 0.0
    vals = []
    for _ in range(n_boot):
        sampled = np.random.multinomial(total, probs)
        vals.append(metric_fn(sampled))
    vals = np.array(vals)
    mean = vals.mean()
    if mean == 0:
        return 0.0
    return float(vals.std() / mean)


def rarefaction_shannon(counts_row, depths, n_reps=5):
    # approximate rarefaction with multinomial sampling at different depths
    total = int(counts_row.sum())
    if total <= 0:
        return [0.0] * len(depths)
    probs = counts_row / float(total)
    res = []
    for d in depths:
        d_int = int(min(max(1, int(d)), total))
        rep_vals = []
        for _ in range(n_reps):
            sampled = np.random.multinomial(d_int, probs)
            rep_vals.append(shannon_from_counts(sampled))
        res.append(float(np.mean(rep_vals)))
    return res


# ----------------------------- taxonomic helpers -----------------------------

def extract_phylum(taxonomy_str):
    # taxonomy like: Root; d__Bacteria; p__Chloroflexota; ...
    if pd.isna(taxonomy_str):
        return 'unclassified'
    toks = [t.strip() for t in str(taxonomy_str).split(';')]
    for t in toks:
        if t.startswith('p__'):
            return t
    # fallback: domain-level
    for t in toks:
        if t.startswith('d__'):
            return t
    return 'unclassified'


def compute_marker_vs_taxprof_corr(marker, samples, count_df, seq_taxonomy, taxprof_df):
    """
    For each sample, compute phylum-level coverage vector for marker and for taxprof; compute Spearman correlation; return mean correlation across samples.
    If taxprof doesn't have any phylum rows for a sample, return nan.
    """
    per_sample_corrs = []
    # precompute marker phylum coverage per sample
    for s in samples:
        if count_df.shape[1] == 0:
            per_sample_corrs.append(np.nan)
            continue
        # marker phylum coverage: sum counts for OTUs mapped to each phylum
        otu_seqs = count_df.columns.tolist()
        counts = count_df.loc[s].values.astype(int)
        phylum_counts = defaultdict(int)
        for otu, cnt in zip(otu_seqs, counts):
            key = (count_df.columns.name, otu)  # not used
            # seq_taxonomy keys are (marker, seq); we need marker - but this function doesn't have marker name
            # so seq_taxonomy keys expected to be accessible externally; we will pass seq_taxonomy mapping where keys are (marker, seq)
            # To avoid complexity, we will assume seq_taxonomy provided maps (marker_id, seq)
            # However this function doesn't know marker id; the caller will pass an adjusted seq_taxonomy_lookup that maps seq->phylum for this marker
            pass
    # Note: this function will be rewritten in caller with simpler approach
    return np.nan


# ------------------------------ main scoring flow ---------------------------

def compute_metrics_for_marker(marker, samples, marker_sample_otu_counts, seq_taxonomy, taxprof_df=None, metadata_df=None, bootstrap_n=100, rarefaction_reps=5):
    # Build count matrix
    count_df = build_marker_count_df(marker, samples, marker_sample_otu_counts)
    # assign name to columns to help later (not necessary)
    count_df.columns.name = marker

    # alpha metrics
    alphas = compute_alpha_metrics(count_df)
    richness = alphas['richness']
    shannon = alphas['shannon']
    simpson = alphas['simpson']

    mean_richness = float(np.nanmean(richness)) if len(richness)>0 else 0.0
    presence_rate = float(np.mean(richness > 0)) if len(richness)>0 else 0.0

    # singleton rate: fraction of OTUs with total count == 1 across all samples
    total_counts_by_otu = count_df.sum(axis=0).values.astype(int) if count_df.shape[1] > 0 else np.array([])
    if total_counts_by_otu.size == 0:
        singleton_rate = np.nan
    else:
        singleton_rate = float(np.sum(total_counts_by_otu == 1) / float(total_counts_by_otu.size))

    # bootstrap CV for shannon: per-sample CV averaged
    cvs = []
    for i in range(count_df.shape[0]):
        row = count_df.iloc[i, :].values.astype(int)
        cv = bootstrap_cv_alpha(row, shannon_from_counts, n_boot=bootstrap_n)
        if not np.isnan(cv):
            cvs.append(cv)
    cv_shannon = float(np.nanmean(cvs)) if len(cvs)>0 else np.nan

    # rarefaction slope: for each sample compute shannon at depths and slope (linear regression of shannon ~ depth), then average slopes
    slopes = []
    for i in range(count_df.shape[0]):
        row = count_df.iloc[i, :].values.astype(int)
        total = int(row.sum())
        if total <= 1:
            continue
        depths = np.unique(np.round(np.linspace(max(1, int(total*0.1)), total, num=5))).astype(int)
        # ensure at least 2 depths
        if len(depths) < 2:
            continue
        vals = rarefaction_shannon(row, depths, n_reps=rarefaction_reps)
        # fit simple linear regression slope
        try:
            slope, intercept, r_val, p_val, std_err = stats.linregress(depths, vals)
            slopes.append(slope)
        except Exception:
            continue
    slope_rarefaction = float(np.nanmean(slopes)) if len(slopes)>0 else np.nan

    # discriminability across groups (if metadata provided)
    F_group = np.nan
    if metadata_df is not None and 'sample' in metadata_df.columns and 'group' in metadata_df.columns:
        # build mapping sample->group
        samp2group = dict(zip(metadata_df['sample'].astype(str), metadata_df['group'].astype(str)))
        # prepare lists
        groups = []
        vals = []
        for s, val in zip(count_df.index.tolist(), shannon):
            if s in samp2group:
                groups.append(samp2group[s])
                vals.append(val)
        if len(set(groups)) >= 2:
            # perform one-way ANOVA across groups
            group_vals = []
            for g in sorted(set(groups)):
                group_vals.append([v for (g2, v) in zip(groups, vals) if g2 == g])
            try:
                F_stat, p = stats.f_oneway(*group_vals)
                F_group = float(F_stat)
            except Exception:
                F_group = np.nan

    # representativeness vs taxprof (if provided) -- compute mean spearman correlation across samples at phylum level
    mean_corr = np.nan
    if taxprof_df is not None:
        # build taxprof phylum coverage table: rows samples, cols phylum
        taxprof = taxprof_df.copy()
        taxprof['phylum'] = taxprof['taxonomy'].apply(extract_phylum)
        # pivot
        tp_pivot = taxprof.pivot_table(index='sample', columns='phylum', values='coverage', aggfunc='sum', fill_value=0)
        # ensure samples in same order
        tp_pivot = tp_pivot.reindex(index=samples, fill_value=0)

        # build marker-derived phylum coverage table
        marker_phylum = pd.DataFrame(0.0, index=samples, columns=tp_pivot.columns)
        for s in samples:
            if s not in count_df.index:
                continue
            row = count_df.loc[s]
            # for each OTU (sequence), get taxonomy by (marker, seq)
            for seq, cnt in row.items():
                if cnt <= 0:
                    continue
                key = (marker, seq)
                tax = seq_taxonomy.get(key, '')
                ph = extract_phylum(tax)
                if ph not in marker_phylum.columns:
                    # add new column if element not in taxprof columns
                    marker_phylum.loc[:, ph] = 0.0
                marker_phylum.at[s, ph] += float(cnt)
        # convert to comparable "coverage" scale by percent (per-sample sum to 100)
        with np.errstate(divide='ignore', invalid='ignore'):
            mp_norm = marker_phylum.div(marker_phylum.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
            mp_norm = mp_norm.fillna(0.0)
        # compute spearman per sample between mp_norm and tp_pivot (select shared columns)
        shared_cols = sorted(set(mp_norm.columns).intersection(set(tp_pivot.columns)))
        if len(shared_cols) >= 1:
            corrs = []
            for s in samples:
                v1 = mp_norm.loc[s, shared_cols].values.astype(float)
                v2 = tp_pivot.loc[s, shared_cols].values.astype(float)
                # require some variance
                if np.nanstd(v1) == 0 and np.nanstd(v2) == 0:
                    continue
                try:
                    r, p = spearmanr(v1, v2)
                    if not np.isnan(r):
                        corrs.append(float(r))
                except Exception:
                    continue
            if len(corrs) > 0:
                mean_corr = float(np.mean(corrs))

    metrics = {
        'marker': marker,
        'n_samples': int(count_df.shape[0]),
        'n_otus': int(count_df.shape[1]),
        'mean_richness': mean_richness,
        'presence_rate': presence_rate,
        'singleton_rate': float(singleton_rate) if not np.isnan(singleton_rate) else np.nan,
        'cv_shannon': float(cv_shannon) if not np.isnan(cv_shannon) else np.nan,
        'slope_rarefaction': float(slope_rarefaction) if not np.isnan(slope_rarefaction) else np.nan,
        'F_group': float(F_group) if not np.isnan(F_group) else np.nan,
        'mean_corr_taxprof': float(mean_corr) if not np.isnan(mean_corr) else np.nan
    }
    return metrics


def normalize_series_minmax(s):
    s = pd.Series(s)
    if s.isnull().all():
        return s
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def compute_scores(metrics_df, mode='expert', expert_weights=None):
    # metrics_df: DataFrame with columns including the metric names
    # define which metrics we will use
    # metrics where higher is better: mean_corr_taxprof, F_group, presence_rate, mean_richness
    # metrics where lower is better: cv_shannon, slope_rarefaction, singleton_rate

    use_cols = ['mean_corr_taxprof', 'F_group', 'presence_rate', 'mean_richness', 'cv_shannon', 'slope_rarefaction', 'singleton_rate']
    for c in use_cols:
        if c not in metrics_df.columns:
            metrics_df[c] = np.nan

    norm = metrics_df.copy()
    for c in use_cols:
        norm[c] = normalize_series_minmax(metrics_df[c])

    # invert columns where lower is better
    for c in ['cv_shannon', 'slope_rarefaction', 'singleton_rate']:
        if c in norm.columns:
            norm[c] = 1.0 - norm[c]

    # default expert weights
    if expert_weights is None:
        expert_weights = {
            'mean_corr_taxprof': 0.30,
            'F_group': 0.20,
            'cv_shannon': 0.15,
            'mean_richness': 0.10,
            'slope_rarefaction': 0.10,
            'singleton_rate': 0.10,
            'presence_rate': 0.05
        }

    if mode == 'expert':
        # compute weighted sum
        score = np.zeros(len(norm))
        for k, w in expert_weights.items():
            if k in norm.columns:
                score += w * norm[k].fillna(0).values
        metrics_df['score'] = score
        metrics_df['rank'] = metrics_df['score'].rank(method='min', ascending=False).astype(int)
        # attach normalized columns
        for c in use_cols:
            metrics_df[c + '_norm'] = norm[c]
        return metrics_df

    # data-driven mode: we will attempt to fit a linear model predicting mean_corr_taxprof using other metrics as features
    # target: mean_corr_taxprof if available
    df = metrics_df.copy()
    features = ['F_group', 'cv_shannon', 'mean_richness', 'slope_rarefaction', 'singleton_rate', 'presence_rate']
    X = df[features].astype(float).fillna(df[features].astype(float).mean())
    y = df['mean_corr_taxprof'].astype(float)
    valid = ~y.isnull()
    if valid.sum() < 3:
        warnings.warn('Not enough markers with taxprof correlation to fit data-driven weights; falling back to expert weights')
        return compute_scores(metrics_df, mode='expert', expert_weights=expert_weights)
    # normalize X
    Xn = X.copy()
    for col in Xn.columns:
        Xn[col] = normalize_series_minmax(Xn[col])
    # fit least squares
    try:
        Xmat = np.asarray(Xn[valid])
        yvec = np.asarray(y[valid])
        # add intercept
        X_with_const = np.column_stack([np.ones(Xmat.shape[0]), Xmat])
        coef, *_ = np.linalg.lstsq(X_with_const, yvec, rcond=None)
        intercept = coef[0]
        coefs = coef[1:]
        # absolute value of coefficients as importance, normalized to sum 1
        w_raw = np.abs(coefs)
        if w_raw.sum() == 0:
            warnings.warn('Fitted coefficients are zero; falling back to expert weights')
            return compute_scores(metrics_df, mode='expert', expert_weights=expert_weights)
        w_norm = w_raw / float(w_raw.sum())
        # map weights back to metric names
        fitted_weights = dict(zip(features, w_norm))
        # we also include mean_corr_taxprof as target-correspondence weight to prefer markers with high correlation
        # give it some weight proportional to how well target is explained (R^2)
        # compute R^2
        y_pred = X_with_const.dot(coef)
        ss_res = float(((yvec - y_pred) ** 2).sum())
        ss_tot = float(((yvec - yvec.mean()) ** 2).sum())
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        # assign combined weights: features according to fitted, plus mean_corr_taxprof gets r2 fraction
        feature_total = 1.0 - r2
        final_weights = {}
        for k, v in fitted_weights.items():
            final_weights[k] = v * feature_total
        final_weights['mean_corr_taxprof'] = r2
        # ensure presence_rate included; if not assigned (it was in features) ok
        # now compute score using normalized metrics
        score = np.zeros(len(df))
        for k, w in final_weights.items():
            if k in norm.columns:
                score += w * norm[k].fillna(0).values
        df['score'] = score
        df['rank'] = df['score'].rank(method='min', ascending=False).astype(int)
        # attach normalized columns
        for c in use_cols:
            df[c + '_norm'] = norm[c]
        # also attach final_weights for reference
        df.attrs = getattr(df, 'attrs', {})
        df.attrs['fitted_weights'] = final_weights
        return df
    except Exception as e:
        warnings.warn(f'Failed fitting data-driven model: {e}; falling back to expert weights')
        return compute_scores(metrics_df, mode='expert', expert_weights=expert_weights)


# ------------------------------- plotting -----------------------------------

def make_plots(metrics_df, outprefix='marker_scores'):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # heatmap of normalized metrics (select _norm columns)
    norm_cols = [c for c in metrics_df.columns if c.endswith('_norm')]
    if len(norm_cols) > 0:
        arr = metrics_df[norm_cols].fillna(0).values
        fig, ax = plt.subplots(figsize=(max(6, 0.12 * arr.shape[0]), max(10, 0.6 * arr.shape[1])))
        cax = ax.imshow(arr, aspect='auto', interpolation='nearest')
        ax.set_yticks(np.arange(len(metrics_df)))
        ax.set_yticklabels(metrics_df['marker'])
        ax.set_xticks(np.arange(len(norm_cols)))
        ax.set_xticklabels(norm_cols, rotation=45, ha='right')
        fig.colorbar(cax, ax=ax)
        plt.tight_layout()
        fig.savefig(outprefix + '_heatmap.png', dpi=150)
        plt.close(fig)
    # barplot of top 20 markers
    topn = min(20, len(metrics_df))
    df_sorted = metrics_df.sort_values('score', ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * topn)))
    ax.barh(range(topn)[::-1], df_sorted['score'].values, align='center')
    ax.set_yticks(range(topn)[::-1])
    ax.set_yticklabels(df_sorted['marker'].values)
    ax.set_xlabel('Score')
    plt.tight_layout()
    fig.savefig(outprefix + '_topbar.png', dpi=150)
    plt.close(fig)


# --------------------------------- CLI --------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MarkerMuse: select best SingleM marker(s) for alpha diversity')
    parser.add_argument('-i', '--input_folder', required=True, help='folder with SingleM OTU files (per-sample), tab-separated')
    parser.add_argument('--otu_suf', required=True, help='suffix for OTU files (e.g. _singlem.otu.tsv)')
    parser.add_argument('--taxprof', default=None, help='if provided: suffix or path for SingleM taxonomic-profile (_singlem.tax.tsv) to enable data-driven scoring')
    parser.add_argument('--metadata', default=None, help='optional metadata TSV with columns sample and group for discriminability metric')
    parser.add_argument('--out_prefix', default='marker_scores', help='prefix for output files')
    parser.add_argument('-o', '--output_folder', default='marker_muse_output', help='folder to write outputs (created if missing)')
    parser.add_argument('--bootstrap_n', type=int, default=100, help='bootstrap replicates for CV estimation')
    parser.add_argument('--rarefaction_reps', type=int, default=5, help='reps per rarefaction depth')
    parser.add_argument('--min_markers_for_fit', type=int, default=4, help='min markers with taxprof to attempt data-driven fit')
    args = parser.parse_args()

    markers, samples, marker_sample_otu_counts, seq_taxonomy = parse_otu_files(args.input_folder, args.otu_suf)
    print(f'Found {len(markers)} markers and {len(samples)} samples')

    # auto-select mode based on presence of --taxprof
    mode = 'data-driven' if args.taxprof is not None else 'expert'
    taxprof_df = None
    if mode == 'data-driven':
        # taxprof can be a suffix pattern contained in input_folder, or a file path
        candidate = os.path.join(args.input_folder, f"*{args.taxprof}")
        matched = sorted(glob.glob(candidate))
        if len(matched) == 1:
            taxprof_df = safe_read_tsv(matched[0])
        elif len(matched) > 1:
            dfs = [safe_read_tsv(p) for p in matched]
            taxprof_df = pd.concat(dfs, axis=0, ignore_index=True)
        elif os.path.exists(args.taxprof):
            taxprof_df = safe_read_tsv(args.taxprof)
        else:
            raise FileNotFoundError('Taxprof file not found; tried: ' + candidate + ' and ' + str(args.taxprof))
        # ensure columns sample, coverage, taxonomy
        for c in ['sample', 'coverage', 'taxonomy']:
            if c not in taxprof_df.columns:
                raise ValueError('Taxprof file must contain columns: sample, coverage, taxonomy')

    metadata_df = None
    if args.metadata is not None:
        metadata_df = safe_read_tsv(args.metadata)
        if 'sample' not in metadata_df.columns or 'group' not in metadata_df.columns:
            raise ValueError('metadata file must contain columns: sample and group')

    all_metrics = []
    for i, marker in enumerate(markers):
        sys.stdout.write(f"Computing metrics for marker {i+1}/{len(markers)}: {marker}\r")
        sys.stdout.flush()
        metrics = compute_metrics_for_marker(marker, samples, marker_sample_otu_counts, seq_taxonomy, taxprof_df=taxprof_df, metadata_df=metadata_df, bootstrap_n=args.bootstrap_n, rarefaction_reps=args.rarefaction_reps)
        all_metrics.append(metrics)
    print('\nDone computing marker metrics')

    metrics_df = pd.DataFrame(all_metrics).set_index('marker')

    # compute scores
    scored = compute_scores(metrics_df.reset_index(), mode=mode)
    # ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    base_prefix_path = os.path.join(args.output_folder, args.out_prefix)

    # if data-driven and fitted weights exist, save them
    if mode == 'data-driven' and hasattr(scored, 'attrs') and 'fitted_weights' in scored.attrs:
        fitted = scored.attrs['fitted_weights']
        weights_path = base_prefix_path + '_fitted_weights.txt'
        with open(weights_path, 'w') as fh:
            for k, v in fitted.items():
                fh.write(f"{k}\t{v}\n")
        print(f'Wrote fitted weights: {weights_path}')
    # sort by score descending and reorder columns to place rank first
    if 'score' in scored.columns:
        scored = scored.sort_values('score', ascending=False).reset_index(drop=True)
    if 'rank' in scored.columns:
        ordered_cols = ['rank'] + [c for c in scored.columns if c != 'rank']
        scored = scored[ordered_cols]

    # save outputs as TSV instead of CSV per user request
    out_tsv = base_prefix_path + '.tsv'
    scored.to_csv(out_tsv, sep='\t', index=False)
    print(f'Wrote {out_tsv}')

    # print top marker information
    if len(scored) > 0 and 'marker' in scored.columns and 'score' in scored.columns:
        top_marker = scored.iloc[0]
        try:
            top_score = float(top_marker['score'])
        except Exception:
            top_score = top_marker['score']
        print(f"Top marker: {top_marker['marker']} (score={top_score:.4f})")

    # make plots (heatmap + top bar) using base prefix path
    make_plots(scored, outprefix=base_prefix_path)
    print('Finished. Plots (if any) written to folder', args.output_folder)


if __name__ == '__main__':
    main()
