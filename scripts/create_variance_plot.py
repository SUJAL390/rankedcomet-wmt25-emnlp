#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')
CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

def read_tsv(path):
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def read_raw_pickle(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return np.asarray(raw)

def numeric_col(df, colname='overall'):
    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found")
    df[colname] = pd.to_numeric(df[colname], errors='coerce')
    return df

def make_langpair(df, src='source_lang', tgt='target_lang'):
    return df[src].astype(str).str.strip() + '-' + df[tgt].astype(str).str.strip()

def generate_final_variance_plot(test_submission_tsv, raw_pkl, output_png):
    df_ranked = read_tsv(test_submission_tsv)
    df_ranked = numeric_col(df_ranked, 'overall')
    df_ranked['raw_score'] = read_raw_pickle(raw_pkl)
    df_ranked['langpair'] = make_langpair(df_ranked)

    rows = []
    for lp, g in df_ranked.groupby('langpair', sort=True):
        rows.append({
            'langpair': lp,
            'var_raw': float(np.nanvar(g['raw_score'].astype(float).values, ddof=0)),
            'var_rank': float(np.nanvar(g['overall'].astype(float).values, ddof=0))
        })
    df_var = pd.DataFrame(rows)
    if df_var.empty:
        print("No data available to plot."); return

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(df_var['var_raw'], df_var['var_rank'], s=35, alpha=0.8, color=CB_PALETTE[0], edgecolors='w', linewidth=0.5, zorder=10)

    ax.plot([0, 1], [0, 1], color=CB_PALETTE[1], linestyle='--', zorder=5, label='y = x (No Change in Variance)')

    x_min, x_max = df_var['var_raw'].min(), df_var['var_raw'].max()
    y_min, y_max = df_var['var_rank'].min(), df_var['var_rank'].max()
    
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_xlabel('Variance of Raw Scores', fontsize=12)
    ax.set_ylabel('Variance of Ranked Scores', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The definitive, camera-ready diagram has been saved to: {output_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the final camera-ready scatter plot of score variance.")
    parser.add_argument('--test-tsv', required=True, help='Path to the submission TSV file')
    parser.add_argument('--raw-pkl', required=True, help='Path to the pickle file with raw scores')
    parser.add_argument('--output', default='var_rank_vs_var_raw_final.png', help='Path to save the output PNG')
    args = parser.parse_args()
    
    generate_final_variance_plot(args.test_tsv, args.raw_pkl, args.output)