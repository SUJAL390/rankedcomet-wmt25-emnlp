import pandas as pd
import os
import csv
import pickle
import numpy as np

INPUT_DECOMPRESSED_PATH = "mteval-task1-test25.tsv"
RAW_SCORES_BACKUP_PATH = "raw_scores_backup.pkl"
SUBMISSION_FILE_PATH = "segments_ranked_final_v2.tsv"

def main():
    print("--- FINAL ATTEMPT SCRIPT: RANKING-BASED SCORE CALIBRATION ---")

    print(f"Loading metadata from: {INPUT_DECOMPRESSED_PATH}")
    try:
        df_test = pd.read_csv(
            INPUT_DECOMPRESSED_PATH, 
            sep='\t', 
            header=0, 
            encoding='utf-8', 
            quoting=csv.QUOTE_NONE,
            low_memory=False
        )
    except FileNotFoundError:
        print(f"ERROR: The decompressed test data file was not found at '{INPUT_DECOMPRESSED_PATH}'.")
        return
    
    print(f"Loading raw scores from backup: {RAW_SCORES_BACKUP_PATH}")
    try:
        with open(RAW_SCORES_BACKUP_PATH, 'rb') as f:
            raw_scores = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Backup file '{RAW_SCORES_BACKUP_PATH}' not found!")
        return
    
    if len(raw_scores) != len(df_test):
        print(f"CRITICAL ERROR: Length mismatch! Raw Scores: {len(raw_scores)}, Data rows: {len(df_test)}")
        return
    
    df_test['raw_overall'] = raw_scores
    
    print("\nApplying per-language-pair RANKING calibration...")
    
    df_test['calibrated_overall'] = 0.0
    
    language_pairs = df_test[['source_lang', 'target_lang']].drop_duplicates()
    
    for index, row in language_pairs.iterrows():
        sl, tl = row['source_lang'], row['target_lang']
        
        mask = (df_test['source_lang'] == sl) & (df_test['target_lang'] == tl)
        
        lp_raw_scores = df_test.loc[mask, 'raw_overall']
        
        if len(lp_raw_scores) <= 1:
            df_test.loc[mask, 'calibrated_overall'] = 0.5
            continue
        
        ranks = lp_raw_scores.rank(method='average')
        
        min_rank = ranks.min()
        max_rank = ranks.max()
        
        if (max_rank - min_rank) > 1e-9:
            normalized_ranks = (ranks - min_rank) / (max_rank - min_rank)
        else:
            normalized_ranks = 0.5
            
        df_test.loc[mask, 'calibrated_overall'] = normalized_ranks
        
    print("Ranking calibration complete.")
    
    print("\nCreating the final calibrated submission DataFrame...")
    submission_df = df_test[['doc_id', 'segment_id', 'source_lang', 'target_lang', 'set_id', 'system_id', 'domain_name', 'method']].copy()
    submission_df['overall'] = df_test['calibrated_overall']
    
    print(f"Saving final calibrated submission file to {SUBMISSION_FILE_PATH}...")
    submission_df.to_csv(
        SUBMISSION_FILE_PATH, 
        sep='\t', 
        index=False, 
        header=True, 
        encoding='utf-8', 
        quoting=csv.QUOTE_NONE
    )
    print(f"--- Calibrated submission file '{SUBMISSION_FILE_PATH}' created successfully! ---")

if __name__ == "__main__":
    main()