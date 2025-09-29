#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
def read_tsv(path):
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def read_raw_pickle(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return np.asarray(raw)

def numeric_overall_col(df, colname='overall'):
    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found in dataframe")
    df[colname] = pd.to_numeric(df[colname], errors='coerce')
    return df

def make_langpair(df, src='source_lang', tgt='target_lang'):
    return df[src].astype(str).str.strip() + '-' + df[tgt].astype(str).str.strip()

def safe_pearson(x, y):
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return np.nan

def safe_spearman(x, y):
    try:
        return float(spearmanr(x, y)[0])
    except Exception:
        return np.nan

def safe_kendall(x, y):
    try:
        return float(kendalltau(x, y)[0])
    except Exception:
        return np.nan
def variance_analysis(test_submission_tsv, raw_pkl, codabench_csv=None, outdir='var_analysis'):
    os.makedirs(outdir, exist_ok=True)
    df_ranked = read_tsv(test_submission_tsv)
    df_ranked = numeric_overall_col(df_ranked, 'overall')
    df_ranked['langpair'] = make_langpair(df_ranked)
    raw = read_raw_pickle(raw_pkl)
    if len(raw) != len(df_ranked):
        raise SystemExit(f"Length mismatch: raw len {len(raw)} vs ranked rows {len(df_ranked)}")
    df_ranked['raw_score'] = raw

    rows = []
    for lp, g in df_ranked.groupby('langpair', sort=True):
        rawv = g['raw_score'].astype(float).values
        rankedv = g['overall'].astype(float).values
        var_raw = float(np.nanvar(rawv, ddof=0))
        var_rank = float(np.nanvar(rankedv, ddof=0))
        delta_var = var_rank - var_raw
        pear_raw_rank = safe_pearson(rawv, rankedv)
        rows.append({'langpair': lp, 'n': len(g), 'var_raw': var_raw, 'var_rank': var_rank,
                     'delta_var': delta_var, 'pearson_raw_vs_ranked': pear_raw_rank})
    df_var = pd.DataFrame(rows).sort_values('n', ascending=False)
    df_var.to_csv(os.path.join(outdir, 'variance_raw_vs_ranked.csv'), index=False)

    if codabench_csv and os.path.exists(codabench_csv):
        cb = pd.read_csv(codabench_csv)
        cb.columns = [c.strip() for c in cb.columns]
        if 'pearson_raw' in cb.columns and 'pearson_rank' in cb.columns:
            cbm = cb[['langpair','pearson_raw','pearson_rank']].copy()
        else:
            pear_cols = [c for c in cb.columns if 'pearson' in c.lower()]
            if len(pear_cols) == 1:
                cbm = cb[['langpair', pear_cols[0]]].copy()
                cbm = cbm.rename(columns={pear_cols[0]:'pearson_snapshot'})
                merged = df_var.merge(cbm, on='langpair', how='left')
                merged['delta_pearson_proxy'] = merged['pearson_snapshot'] - merged['pearson_raw_vs_ranked']
                merged.to_csv(os.path.join(outdir, 'variance_with_codabench_delta.csv'), index=False)
                x = merged['delta_var'].values
                y = merged['delta_pearson_proxy'].values
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() > 2:
                    r = pearsonr(x[ok], y[ok])[0]
                else:
                    r = np.nan
                plt.figure(figsize=(6,4))
                plt.scatter(x[ok], y[ok], s=20)
                plt.axhline(0, color='gray', linewidth=0.7)
                plt.axvline(0, color='gray', linewidth=0.7)
                plt.xlabel('Delta variance (ranked - raw)')
                plt.ylabel('Delta Pearson (snapshot - raw_vs_ranked)')
                plt.title(f'Delta variance vs Delta Pearson (r={r:.3f})')
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, 'delta_var_vs_delta_pearson.png'), dpi=200)
                plt.close()
                return {'df_var': df_var, 'merged': merged}
        merged = df_var.merge(cb, on='langpair', how='left')
        merged.to_csv(os.path.join(outdir, 'variance_with_codabench_rawmerge.csv'), index=False)
        print("Wrote variance_with_codabench_rawmerge.csv")
    else:
        plt.figure(figsize=(6,4))
        plt.scatter(df_var['var_raw'], df_var['var_rank'], s=20)
        m, b = np.polyfit(df_var['var_raw'].fillna(0), df_var['var_rank'].fillna(0), 1)
        xs = np.linspace(df_var['var_raw'].min(), df_var['var_raw'].max(), 100)
        plt.plot(xs, m*xs + b, color='red', linewidth=1)
        plt.xlabel('var_raw')
        plt.ylabel('var_rank')
        plt.title('Variance: ranked vs raw (per language pair)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'var_rank_vs_var_raw.png'), dpi=200)
        plt.close()
    return {'df_var': df_var}

def dev_based_calibrate(dev_tsv, dev_raw_pkl, test_tsv, test_raw_pkl, out_prefix, outdir='dev_calibrate'):
    os.makedirs(outdir, exist_ok=True)
    dev_df = read_tsv(dev_tsv)
    dev_df = numeric_overall_col(dev_df, 'overall') if 'overall' in dev_df.columns else dev_df
    dev_df['raw'] = read_raw_pickle(dev_raw_pkl)
    dev_df['langpair'] = make_langpair(dev_df)

    test_df = read_tsv(test_tsv)
    test_df['raw'] = read_raw_pickle(test_raw_pkl)
    test_df['langpair'] = make_langpair(test_df)

    rows = []
    test_mapped = []
    for lp, gdev in dev_df.groupby('langpair', sort=True):
        gtest = test_df[test_df['langpair'] == lp]
        if len(gtest) == 0:
            continue
        dev_raw = np.asarray(gdev['raw'].astype(float))
        if np.all(np.isnan(dev_raw)) or dev_raw.size < 2:
            for idx in gtest.index:
                test_mapped.append({'index': idx, 'langpair': lp, 'mapped': 0.5})
            rows.append({'langpair': lp, 'dev_n': len(gdev), 'note': 'degenerate_dev'})
            continue
        sorted_vals = np.sort(dev_raw)
        uniq_vals, idx_first = np.unique(sorted_vals, return_index=True)
        ranks = np.arange(1, len(sorted_vals)+1)
        quantiles = (ranks - 1) / (len(sorted_vals) - 1) if len(sorted_vals) > 1 else np.array([0.5])
        q_at_uniq = []
        for v in uniq_vals:
            idxs = np.where(sorted_vals == v)[0]
            qvals = (idxs) / (len(sorted_vals)-1) if len(sorted_vals) > 1 else np.array([0.5])
            q_at_uniq.append(float(np.mean(qvals)))
        q_at_uniq = np.array(q_at_uniq)
        test_raw_vals = np.asarray(gtest['raw'].astype(float))
        minv, maxv = uniq_vals[0], uniq_vals[-1]
        test_raw_clamped = np.clip(test_raw_vals, minv, maxv)
        mapped = np.interp(test_raw_clamped, uniq_vals, q_at_uniq)
        for idx, val in zip(gtest.index, mapped):
            test_mapped.append({'index': idx, 'langpair': lp, 'mapped': float(val)})
        rows.append({'langpair': lp, 'dev_n': len(gdev), 'test_n': len(gtest), 'dev_min': float(minv), 'dev_max': float(maxv)})
    mapped_df = pd.DataFrame(test_mapped).set_index('index')
    test_df['pred_dev_mapped'] = mapped_df['mapped'].reindex(test_df.index).fillna(0.5).values
    out_test = os.path.join(outdir, f'{out_prefix}_test_dev_mapped.tsv')
    sub_cols = ['doc_id','segment_id','source_lang','target_lang','set_id','system_id','domain_name','method']
    for c in sub_cols:
        if c not in test_df.columns:
            test_df[c] = ''
    subdf = test_df[sub_cols].copy()
    subdf['overall'] = test_df['pred_dev_mapped'].astype(float)
    subdf.to_csv(out_test, sep='\t', index=False)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f'{out_prefix}_dev_mapping_stats.csv'), index=False)
    print("Wrote dev-mapped test submission to:", out_test)
    return {'test_mapped_df': test_df, 'mapping_stats': pd.DataFrame(rows)}

def ablation_experiments(test_tsv, raw_pkl=None, dev_tsv=None, dev_raw_pkl=None, outdir='ablation'):
    os.makedirs(outdir, exist_ok=True)
    test_df = read_tsv(test_tsv)
    test_df['raw'] = read_raw_pickle(raw_pkl) if raw_pkl else None
    test_df = numeric_overall_col(test_df, 'overall') if 'overall' in test_df.columns else test_df
    test_df['langpair'] = make_langpair(test_df)
    outputs = {}
    df1 = test_df.copy()
    df1['pred_ranked_local'] = 0.0
    for lp, g in df1.groupby('langpair', sort=True):
        idx = g.index
        ranks = g['raw'].astype(float).rank(method='average') if 'raw' in g.columns and g['raw'].notna().any() else pd.Series(np.nan, index=idx)
        if ranks.isna().all():
            ranks = g['overall'].astype(float).rank(method='average')
        min_r, max_r = ranks.min(), ranks.max()
        if np.isnan(min_r) or np.isnan(max_r) or (max_r - min_r) < 1e-12:
            df1.loc[idx, 'pred_ranked_local'] = 0.5
        else:
            df1.loc[idx, 'pred_ranked_local'] = (ranks - min_r) / (max_r - min_r)
    outputs['perlang_rankminmax'] = df1['pred_ranked_local'].values

    df2 = test_df.copy()
    df2['pred_perlang_z_minmax'] = 0.0
    for lp, g in df2.groupby('langpair', sort=True):
        idx = g.index
        vals = g['raw'].astype(float).values if ('raw' in g.columns and g['raw'].notna().any()) else g['overall'].astype(float).values
        if len(vals) == 0 or np.all(np.isnan(vals)):
            df2.loc[idx, 'pred_perlang_z_minmax'] = 0.5
            continue
        mean = np.nanmean(vals); std = np.nanstd(vals)
        if std < 1e-12:
            z = np.zeros_like(vals)
        else:
            z = (vals - mean) / std
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        if zmax - zmin < 1e-12:
            mapped = np.full_like(z, 0.5, dtype=float)
        else:
            mapped = (z - zmin) / (zmax - zmin)
        df2.loc[idx, 'pred_perlang_z_minmax'] = mapped
    outputs['perlang_z_minmax'] = df2['pred_perlang_z_minmax'].values

    df3 = test_df.copy()
    vals_global = df3['raw'].astype(float).values if ('raw' in df3.columns and df3['raw'].notna().any()) else df3['overall'].astype(float).values
    if len(vals_global) == 0 or np.all(np.isnan(vals_global)):
        df3['pred_global_minmax'] = 0.5
    else:
        gmin, gmax = np.nanmin(vals_global), np.nanmax(vals_global)
        if gmax - gmin < 1e-12:
            df3['pred_global_minmax'] = 0.5
        else:
            df3['pred_global_minmax'] = ((vals_global - gmin) / (gmax - gmin)).astype(float)
    outputs['global_minmax'] = df3['pred_global_minmax'].values

    df4 = test_df.copy()
    df4['pred_dev_isotonic'] = np.nan
    df4['pred_perlang_isotonic'] = np.nan
    if dev_tsv and dev_raw_pkl:
        dev_df = read_tsv(dev_tsv)
        dev_df['raw'] = read_raw_pickle(dev_raw_pkl)
        dev_df['langpair'] = make_langpair(dev_df)
        dev_vals = dev_df['raw'].astype(float).values
        test_vals = test_df['raw'].astype(float).values
        if not np.all(np.isnan(dev_vals)):
            dev_ranks = pd.Series(dev_vals).rank(method='average').values
            dev_quantiles = (dev_ranks - 1) / (len(dev_ranks)-1) if len(dev_ranks) > 1 else np.repeat(0.5, len(dev_ranks))
            ir = IsotonicRegression(out_of_bounds='clip')
            try:
                ir.fit(dev_vals, dev_quantiles)
                df4['pred_dev_isotonic'] = ir.predict(test_vals)
            except Exception as e:
                print("Global isotonic fit failed:", e)
        for lp, gdev in dev_df.groupby('langpair', sort=True):
            gtest = test_df[test_df['langpair'] == lp]
            if len(gtest) == 0:
                continue
            dv = gdev['raw'].astype(float).values
            tv = gtest['raw'].astype(float).values
            if dv.size < 2 or np.all(np.isnan(dv)):
                df4.loc[gtest.index, 'pred_perlang_isotonic'] = 0.5
                continue
            dev_ranks = pd.Series(dv).rank(method='average').values
            dev_q = (dev_ranks - 1) / (len(dev_ranks)-1) if len(dev_ranks) > 1 else np.repeat(0.5, len(dev_ranks))
            ir = IsotonicRegression(out_of_bounds='clip')
            try:
                ir.fit(dv, dev_q)
                mapped = ir.predict(tv)
                df4.loc[gtest.index, 'pred_perlang_isotonic'] = mapped
            except Exception as e:
                df4.loc[gtest.index, 'pred_perlang_isotonic'] = 0.5
    else:
        print("Dev not provided: skipping isotonic dev-based ablations")

    out_rows = []
    for lp, g in test_df.groupby('langpair', sort=True):
        idx = g.index
        rawv = g['raw'].astype(float).values if ('raw' in g.columns and g['raw'].notna().any()) else None
        row = {'langpair': lp, 'n': len(g)}
        for name, arr in outputs.items():
            arr_vals = np.asarray(arr)[idx]
            if rawv is not None:
                p = safe_pearson(rawv, arr_vals)
                s = safe_spearman(rawv, arr_vals)
                k = safe_kendall(rawv, arr_vals)
            else:
                p = s = k = np.nan
            row[f'{name}_pearson_vs_raw'] = p
            row[f'{name}_spearman_vs_raw'] = s
            row[f'{name}_kendall_vs_raw'] = k
        if 'pred_dev_isotonic' in df4.columns:
            pdv = df4.loc[idx, 'pred_dev_isotonic'].astype(float).values if not df4['pred_dev_isotonic'].isna().all() else None
            p2 = safe_pearson(rawv, pdv) if (pdv is not None and rawv is not None) else np.nan
            row['dev_isotonic_pearson_vs_raw'] = p2
            row['dev_isotonic_spearman_vs_raw'] = safe_spearman(rawv, pdv) if (pdv is not None and rawv is not None) else np.nan
        if 'pred_perlang_isotonic' in df4.columns:
            pdp = df4.loc[idx, 'pred_perlang_isotonic'].astype(float).values if not df4['pred_perlang_isotonic'].isna().all() else None
            row['perlang_isotonic_pearson_vs_raw'] = safe_pearson(rawv, pdp) if (pdp is not None and rawv is not None) else np.nan
            row['perlang_isotonic_spearman_vs_raw'] = safe_spearman(rawv, pdp) if (pdp is not None and rawv is not None) else np.nan
        out_rows.append(row)
    df_ablation = pd.DataFrame(out_rows).sort_values('n', ascending=False)
    df_ablation.to_csv(os.path.join(outdir, 'ablation_perlang_proxy_vs_raw.csv'), index=False)

    agg_rows = []
    for name, arr in outputs.items():
        arr_full = np.asarray(arr)
        if test_df['raw'].notna().any():
            p_all = safe_pearson(test_df['raw'].astype(float).values, arr_full)
            agg_rows.append({'method': name, 'pearson_vs_raw_all': p_all})
        else:
            agg_rows.append({'method': name, 'pearson_vs_raw_all': np.nan})
    if 'pred_dev_isotonic' in df4.columns and test_df['raw'].notna().any():
        agg_rows.append({'method': 'dev_isotonic', 'pearson_vs_raw_all': safe_pearson(test_df['raw'].astype(float).values, df4['pred_dev_isotonic'].astype(float).values)})
    if 'pred_perlang_isotonic' in df4.columns and test_df['raw'].notna().any():
        agg_rows.append({'method': 'perlang_isotonic', 'pearson_vs_raw_all': safe_pearson(test_df['raw'].astype(float).values, df4['pred_perlang_isotonic'].astype(float).values)})

    pd.DataFrame(agg_rows).to_csv(os.path.join(outdir, 'ablation_aggregate_proxy_vs_raw.csv'), index=False)
    print("Wrote ablation CSVs to:", outdir)
    return {'df_ablation': df_ablation, 'agg': pd.DataFrame(agg_rows)}

def main():
    parser = argparse.ArgumentParser(description="RankedCOMET full analysis scripts")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('variance-analysis', help='Analyze variance and explain Pearson changes')
    p1.add_argument('--test', required=True, help='Submitted calibrated TSV (segments_fifth.tsv)')
    p1.add_argument('--raw', required=True, help='raw_scores_backup.pkl (raw COMET floats for test)')
    p1.add_argument('--codabench', required=False, help='Optional Codabench per-lang CSV snapshot (with per-lang pearson)')
    p1.add_argument('--outdir', default='var_analysis', help='Output folder')

    p2 = sub.add_parser('dev-calibrate', help='Create dev->test mapping and produce dev-mapped test predictions')
    p2.add_argument('--dev_tsv', required=True, help='Dev segments TSV (same schema)')
    p2.add_argument('--dev_raw', required=True, help='Dev raw predictions pickle')
    p2.add_argument('--test_tsv', required=True, help='Test segments TSV (original test file)')
    p2.add_argument('--test_raw', required=True, help='Test raw predictions pickle')
    p2.add_argument('--out_prefix', default='devmap', help='Prefix for outputs')
    p2.add_argument('--outdir', default='dev_calibrate', help='Output folder')

    p3 = sub.add_parser('ablation', help='Run ablation suite')
    p3.add_argument('--test', required=True, help='Test submission TSV (segments_fifth or original)')
    p3.add_argument('--raw', required=False, help='Test raw pickle (optional but recommended)')
    p3.add_argument('--dev_tsv', required=False, help='Dev TSV (optional for isotonic)')
    p3.add_argument('--dev_raw', required=False, help='Dev raw pickle (optional)')
    p3.add_argument('--outdir', default='ablation', help='Output folder')

    p4 = sub.add_parser('run-all', help='Run variance-analysis, dev-calibrate (if provided), ablation')
    p4.add_argument('--test', required=True, help='Submitted calibrated TSV (segments_fifth.tsv)')
    p4.add_argument('--raw', required=True, help='raw_scores_backup.pkl for test')
    p4.add_argument('--dev_tsv', required=False, help='Dev TSV (optional)')
    p4.add_argument('--dev_raw', required=False, help='Dev raw pickle (optional)')
    p4.add_argument('--codabench', required=False, help='Optional Codabench snapshot CSV for additional comparisons')
    p4.add_argument('--outdir', default='results_all', help='Output folder')

    args = parser.parse_args()

    if args.cmd == 'variance-analysis':
        variance_analysis(args.test, args.raw, codabench_csv=args.codabench, outdir=args.outdir)
    elif args.cmd == 'dev-calibrate':
        dev_based_calibrate(args.dev_tsv, args.dev_raw, args.test_tsv, args.test_raw, args.out_prefix, outdir=args.outdir)
    elif args.cmd == 'ablation':
        ablation_experiments(args.test, raw_pkl=args.raw, dev_tsv=args.dev_tsv, dev_raw_pkl=args.dev_raw, outdir=args.outdir)
    elif args.cmd == 'run-all':
        os.makedirs(args.outdir, exist_ok=True)
        var_out = os.path.join(args.outdir, 'var_analysis')
        variance_analysis(args.test, args.raw, codabench_csv=args.codabench if hasattr(args,'codabench') else None, outdir=var_out)
        if args.dev_tsv and args.dev_raw:
            dev_out = os.path.join(args.outdir, 'dev_calibrate')
            dev_based_calibrate(args.dev_tsv, args.dev_raw, args.test, args.raw, out_prefix='devmap', outdir=dev_out)
        ablation_out = os.path.join(args.outdir, 'ablation')
        ablation_experiments(args.test, raw_pkl=args.raw, dev_tsv=args.dev_tsv, dev_raw_pkl=args.dev_raw, outdir=ablation_out)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
