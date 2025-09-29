import pandas as pd
import os
import csv
import pickle
import numpy as np

INPUT_DECOMPRESSED_PATH = "mteval-task1-test25.tsv"
RAW_SCORES_BACKUP_PATH = "raw_scores_backup.pkl"
SUBMISSION_FILE_PATH = "segments_final_gambit.tsv"

def main():
    print("--- FINAL GAMBIT SCRIPT (VERY HIGH RISK) ---")

    print("Loading data and scores...")
    try:
        df_test = pd.read_csv(INPUT_DECOMPRESSED_PATH, sep='\t', header=0, encoding='utf-8', quoting=csv.QUOTE_NONE, low_memory=False)
        with open(RAW_SCORES_BACKUP_PATH, 'rb') as f:
            raw_scores = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Backup or data file not found!")
        return
    
    if len(raw_scores) != len(df_test):
        print("CRITICAL ERROR: Length mismatch!")
        return
    
    df_test['raw_overall'] = raw_scores
    
    print("Generating rank-normalized scores...")
    df_test['ranked_overall'] = 0.0
    language_pairs = df_test[['source_lang', 'target_lang']].drop_duplicates()
    
    for _, row in language_pairs.iterrows():
        sl, tl = row['source_lang'], row['target_lang']
        mask = (df_test['source_lang'] == sl) & (df_test['target_lang'] == tl)
        lp_raw_scores = df_test.loc[mask, 'raw_overall']
        
        if len(lp_raw_scores) <= 1:
            df_test.loc[mask, 'calibrated_overall'] = 0.5
            continue
        
        ranks = lp_raw_scores.rank(method='average')
        min_rank, max_rank = ranks.min(), ranks.max()
        
        if (max_rank - min_rank) > 1e-9:
            normalized_ranks = (ranks - min_rank) / (max_rank - min_rank)
        else:
            normalized_ranks = 0.5
            
        df_test.loc[mask, 'ranked_overall'] = normalized_ranks

    print("Creating the final ensemble...")
    
    min_raw, max_raw = df_test['raw_overall'].min(), df_test['raw_overall'].max()
    df_test['raw_scaled'] = (df_test['raw_overall'] - min_raw) / (max_raw - min_raw)
    
    ensemble_weight = 0.6 
    df_test['ensemble_score'] = (ensemble_weight * df_test['ranked_overall']) + \
                                ((1 - ensemble_weight) * df_test['raw_scaled'])
                                
    lower_quantile = df_test['ensemble_score'].quantile(0.01)
    upper_quantile = df_test['ensemble_score'].quantile(0.99)
    df_test['final_score'] = df_test['ensemble_score'].clip(lower=lower_quantile, upper=upper_quantile)

    min_final, max_final = df_test['final_score'].min(), df_test['final_score'].max()
    df_test['final_score'] = (df_test['final_score'] - min_final) / (max_final - min_final)

    print("Ensemble created.")
    
    print("\nCreating the final gambit submission DataFrame...")
    submission_df = df_test[['doc_id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id', 'domain_name', 'method']].copy()
    submission_df['overall'] = df_test['final_score']
    
    print(f"Saving final gambit submission file to {SUBMISSION_FILE_PATH}...")
    submission_df.to_csv(SUBMISSION_FILE_PATH, sep='\t', index=False, header=True, encoding='utf-8', quoting=csv.QUOTE_NONE)
    print(f"--- Final gambit submission file '{SUBMISSION_FILE_PATH}' created successfully! ---")

if __name__ == "__main__":
    main()