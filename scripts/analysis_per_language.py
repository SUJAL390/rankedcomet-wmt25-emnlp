#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import argparse
from tqdm import tqdm

rng = np.random.RandomState(12345)

def safe_metric(f, x, y):
    try:
        return float(f(x, y)[0])
    except Exception:
        return float('nan')

def fisher_z_mean(arr):
    arr = np.array(arr, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.nan
    r = np.clip(arr[mask], -1 + 1e-12, 1 - 1e-12)
    z = np.arctanh(r)
    return float(np.tanh(np.mean(z)))

def simple_mean(arr):
    arr = np.array(arr, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.nan
    return float(arr[mask].mean())

def weighted_mean(arr, weights):
    arr = np.array(arr, dtype=float)
    w = np.array(weights, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.nan
    return float((arr[mask] * w[mask]).sum() / w[mask].sum())

def compute_ranked(group, score_col='pred_raw'):
    r = group[score_col].rank(method='average', ascending=True)
    if (r.max() - r.min()) > 0:
        return (r - r.min()) / (r.max() - r.min())
    else:
        return pd.Series(0.5, index=group.index)

def pear(x, y):
    return safe_metric(pearsonr, x, y)

def spr(x, y):
    return safe_metric(spearmanr, x, y)

def kt(x, y):
    return safe_metric(kendalltau, x, y)

def read_segments(path):
    try:
        df = pd.read_csv(path, sep='\t', quoting=3, dtype=str)
    except Exception:
        df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip('\ufeff').strip() for c in df.columns]
    return df

def to_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute per-language Pearson/Spearman/Kendall and bootstrap CIs.")
    parser.add_argument('tsv', help='segments.tsv (tab-separated) with columns seg_id, langpair (or source_lang,target_lang), pred_raw, human (optionally pred_ranked)')
    parser.add_argument('--bootstrap', type=int, default=2000, help='number of bootstrap samples (default 2000)')
    parser.add_argument('--min_per_lang', type=int, default=5, help='minimum segments per langpair to compute bootstrap (default 5)')
    args = parser.parse_args()

    if not os.path.exists(args.tsv):
        print("ERROR: file not found:", args.tsv)
        sys.exit(1)

    print("Reading:", args.tsv)
    df = read_segments(args.tsv)

    colmap = {c.lower(): c for c in df.columns}
    cols_lower = set(colmap.keys())

    if 'langpair' in cols_lower:
        lang_col = colmap['langpair']
    elif 'source_lang' in cols_lower and 'target_lang' in cols_lower:
        src = colmap['source_lang']; tgt = colmap['target_lang']
        df['langpair'] = df[src].astype(str).str.strip() + '-' + df[tgt].astype(str).str.strip()
        lang_col = 'langpair'
    else:
        raise SystemExit("Need 'langpair' column or 'source_lang'+'target_lang' in the TSV.")

    if 'pred_raw' in cols_lower:
        pred_raw_col = colmap['pred_raw']
    elif 'raw_overall' in cols_lower:
        pred_raw_col = colmap['raw_overall']
        df.rename(columns={pred_raw_col: 'pred_raw'}, inplace=True)
        pred_raw_col = 'pred_raw'
    else:
        raise SystemExit("Need 'pred_raw' or 'raw_overall' column in the TSV.")

    human_candidates = ['human', 'mqm', 'score', 'label', 'human_score', 'human_label']
    human_col = None
    for c in human_candidates:
        if c in cols_lower:
            human_col = colmap[c]
            break
    if human_col is None:
        numeric_cols = []
        for c in df.columns:
            if c == pred_raw_col or c == lang_col: continue
            try:
                pd.to_numeric(df[c].dropna().iloc[:10])
                numeric_cols.append(c)
            except Exception:
                pass
        if len(numeric_cols) == 0:
            raise SystemExit("Could not find a human score column. Provide 'human' or 'mqm' or 'score' column.")
        human_col = numeric_cols[0]
        print("Auto-picked human column:", human_col)

    df = to_numeric_cols(df, [pred_raw_col, human_col])

    before = len(df)
    df = df.dropna(subset=[pred_raw_col, human_col, lang_col])
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows with missing pred_raw/human/langpair")

    if 'pred_ranked' not in df.columns:
        print("Computing per-language ranked predictions (pred_ranked)...")
        df['pred_ranked'] = df.groupby(lang_col, group_keys=False).apply(lambda g: compute_ranked(g, score_col=pred_raw_col))
        df['pred_ranked'] = pd.to_numeric(df['pred_ranked'], errors='coerce')
    else:
        df['pred_ranked'] = pd.to_numeric(df['pred_ranked'], errors='coerce')

    groups = sorted(df[lang_col].unique())
    rows = []
    print(f"Found {len(groups)} language pairs. Computing per-language metrics...")
    for g in tqdm(groups):
        sub = df[df[lang_col] == g]
        n = len(sub)
        raw = sub['pred_raw'].values
        ranked = sub['pred_ranked'].values
        human = sub[human_col].values
        p_raw = pear(raw, human)
        p_rank = pear(ranked, human)
        s_raw = spr(raw, human)
        s_rank = spr(ranked, human)
        k_raw = kt(raw, human)
        k_rank = kt(ranked, human)
        rows.append({
            'langpair': g, 'n': n,
            'pearson_raw': p_raw, 'pearson_rank': p_rank,
            'spearman_raw': s_raw, 'spearman_rank': s_rank,
            'kendall_raw': k_raw, 'kendall_rank': k_rank
        })

    per_lang_df = pd.DataFrame(rows).set_index('langpair')
    per_lang_df.to_csv('per_language_correlations.csv')
    print("Wrote per_language_correlations.csv")

    weights = per_lang_df['n'].astype(float).values
    agg = {}
    for metric in ['pearson_raw','pearson_rank','spearman_raw','spearman_rank','kendall_raw','kendall_rank']:
        vals = per_lang_df[metric].values
        agg[metric+'_simple_mean'] = simple_mean(vals)
        agg[metric+'_fisher_z'] = fisher_z_mean(vals)
        agg[metric+'_weighted_mean'] = weighted_mean(vals, weights)
    agg_df = pd.DataFrame([agg])
    agg_df.to_csv('aggregated_results.csv', index=False)
    print("Wrote aggregated_results.csv")

    B = args.bootstrap
    diffs = []
    pvals = []
    ci_low = []
    ci_high = []
    print(f"Bootstrapping Pearson differences per language (B={B})...")
    for g in tqdm(groups):
        sub = df[df[lang_col] == g]
        n = len(sub)
        if n < args.min_per_lang:
            diffs.append(np.nan); pvals.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
            continue
        raw = sub['pred_raw'].values
        ranked = sub['pred_ranked'].values
        human = sub[human_col].values
        boot_diffs = []
        for _ in range(B):
            idx = rng.choice(n, size=n, replace=True)
            try:
                pr = pear(raw[idx], human[idx])
                prk = pear(ranked[idx], human[idx])
            except Exception:
                pr = np.nan; prk = np.nan
            boot_diffs.append(prk - pr)
        boot_diffs = np.array(boot_diffs)
        ci = np.nanpercentile(boot_diffs, [2.5, 97.5])
        ci_low.append(ci[0]); ci_high.append(ci[1])
        pval = 2.0 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())
        pvals.append(pval)
        diffs.append(float(np.nanmean(boot_diffs)))
    bootdf = pd.DataFrame({
        'langpair': groups,
        'pearson_diff_mean': diffs,
        'pearson_diff_CI_low': ci_low,
        'pearson_diff_CI_high': ci_high,
        'pearson_diff_pval': pvals
    }).set_index('langpair')
    bootdf.to_csv('per_language_pearson_bootstrap.csv')
    print("Wrote per_language_pearson_bootstrap.csv")

    print("All done.")

if __name__ == '__main__':
    main()
