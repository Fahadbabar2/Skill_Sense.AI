"""
Merges the Kaggle resume dataset with synthetic behavioral candidate data
so the model can be trained on both "claimed" (resume) and "observed"
(behavioral) features.

Important honesty note (keep this in your writeup):
The resume rows and the behavioral rows come from two different, unrelated
sources -- there's no dataset anywhere that has both a real resume AND a
real behavioral assessment for the same person. This script creates that
link artificially by randomly pairing a resume row with a behavioral row.
That's a reasonable way to test whether your pipeline and model *can* use
resume features -- it does NOT mean the resulting model reflects a real
relationship between resumes and actual skill in the world. Real data
collection (same person, both sources) would be the genuine v2 fix.

Run (from project root):
    python ml/generate_synthetic_data.py     # if not already done
    python ml/merge_resume_data.py
Produces:
    ml/merged_training_data.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

RESUME_PATH = "ml/data/AI_Resume_Screening.csv"
BEHAVIORAL_PATH = "ml/synthetic_candidates.csv"
OUTPUT_PATH = "ml/merged_training_data.csv"


def load_resume_features():
    df = pd.read_csv(RESUME_PATH)
    df["claimed_experience_years"] = df["Experience (Years)"]
    df["claimed_skill_count"] = df["Skills"].apply(lambda s: len(str(s).split(",")))
    return df[["claimed_experience_years", "claimed_skill_count"]]


def load_behavioral_features():
    return pd.read_csv(BEHAVIORAL_PATH)


def merge():
    resume_feats = load_resume_features()
    behavioral = load_behavioral_features()

    n = len(behavioral)
    # Sample resume rows (with replacement if resume dataset is smaller)
    sampled_idx = RNG.choice(len(resume_feats), size=n, replace=len(resume_feats) < n)
    resume_sample = resume_feats.iloc[sampled_idx].reset_index(drop=True)

    merged = pd.concat([behavioral.reset_index(drop=True), resume_sample], axis=1)
    merged = merged.sample(frac=1, random_state=7).reset_index(drop=True)
    return merged


if __name__ == "__main__":
    merged = merge()
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(merged)} merged rows to {OUTPUT_PATH}")
    print(merged.head())
    print()
    print("Columns:", merged.columns.tolist())
