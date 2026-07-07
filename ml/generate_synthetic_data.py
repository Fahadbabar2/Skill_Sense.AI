"""
Generates a synthetic training dataset for skill-level prediction.

Why synthetic? The doc this project is based on is honest that there's no
public dataset mapping "answer behavior" -> "verified skill level." Real
data means running this tool on real people and getting their level
verified independently (instructor grade, certification, etc.) -- that's
the real v2 dataset. For now, we simulate plausible candidates at each
tier so the pipeline (feature vector -> XGBoost -> label) is real and
testable end to end. Swap this file's output for real collected sessions
later; nothing else in the project needs to change.

Run:
    python ml/generate_synthetic_data.py
Produces:
    ml/synthetic_candidates.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Rough behavioral profile per tier. Numbers are illustrative, not measured --
# document this assumption clearly in your README/report.
PROFILES = {
    "Beginner":     dict(overall=(0.35, 0.12), easy=(0.6, 0.15), medium=(0.25, 0.15), hard=(0.05, 0.08), time=(45, 15), quality=(0.3, 0.15), hints=(3, 1.5)),
    "Intermediate": dict(overall=(0.55, 0.12), easy=(0.85, 0.1),  medium=(0.55, 0.15), hard=(0.2, 0.15),  time=(35, 12), quality=(0.5, 0.15), hints=(1.5, 1.0)),
    "Advanced":     dict(overall=(0.75, 0.1),  easy=(0.95, 0.05), medium=(0.8, 0.12),  hard=(0.5, 0.18),  time=(25, 10), quality=(0.72, 0.12), hints=(0.5, 0.7)),
    "Expert":       dict(overall=(0.9, 0.07),  easy=(1.0, 0.03),  medium=(0.92, 0.08), hard=(0.78, 0.15), time=(18, 8),  quality=(0.88, 0.08), hints=(0.1, 0.3)),
}

N_PER_TIER = 300


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate():
    rows = []
    for label, p in PROFILES.items():
        for _ in range(N_PER_TIER):
            overall = clip01(RNG.normal(*p["overall"]))
            easy = clip01(RNG.normal(*p["easy"]))
            medium = clip01(RNG.normal(*p["medium"]))
            hard = clip01(RNG.normal(*p["hard"]))
            time_taken = max(3, RNG.normal(*p["time"]))
            quality = clip01(RNG.normal(*p["quality"]))
            hints = max(0, RNG.normal(*p["hints"]))
            hard_attempted = RNG.integers(1, 5)

            rows.append({
                "overall_accuracy": overall,
                "easy_accuracy": easy,
                "medium_accuracy": medium,
                "hard_accuracy": hard,
                "avg_time_taken": time_taken,
                "avg_quality_score": quality,
                "total_hints_used": hints,
                "hard_questions_attempted": hard_attempted,
                "label": label,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "ml/synthetic_candidates.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic candidates to {out_path}")
    print(df.groupby("label").mean(numeric_only=True).round(2))
