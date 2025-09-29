#!/usr/bin/env python3
import sys, pickle, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

def load_raw(pkl):
    with open(pkl,'rb') as f:
        return pickle.load(f)

def main(segments_tsv, raw_pkl):
    df = pd.read_csv(segments_tsv, sep='\t', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
    raw = load_raw(raw_pkl)
    if len(raw) != len(df):
        raise SystemExit("Length mismatch raw vs segments")
    df['raw_score'] = raw
    df['langpair'] = df['source_lang'].str.strip() + '-' + df['target_lang'].str.strip()

    rows = []
    for lp, g in df.groupby('langpair'):
        rawv = g['raw_score'].astype(float).values
        rankedv = g['overall'].astype(float).values
        try:
            p_r = pearsonr(rawv, rankedv)[0]
        except:
            p_r = float('nan')
        try:
            s_r = spearmanr(rawv, rankedv)[0]
        except:
            s_r = float('nan')
        try:
            k_r = kendalltau(rawv, rankedv)[0]
        except:
            k_r = float('nan')
        rows.append({'langpair': lp, 'pearson_raw_vs_ranked': p_r, 'spearman_raw_vs_ranked': s_r, 'kendall_raw_vs_ranked': k_r})
    out = pd.DataFrame(rows)
    out.to_csv('raw_vs_ranked_stats.csv', index=False)
    print("Wrote raw_vs_ranked_stats.csv")

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_raw_ranked.py segments_fifth.tsv raw_scores_backup.pkl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
