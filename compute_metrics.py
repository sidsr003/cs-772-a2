import json
import math
from tqdm import tqdm
from argparse import ArgumentParser

def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between strings a and b."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]

def lcs_length(a: str, b: str) -> int:
    """Compute length of the Longest Common Subsequence (LCS) using Eq. 2."""
    ed = edit_distance(a, b)
    return int((len(a) + len(b) - ed) / 2)

def compute_metrics(jsonl_path: str):
    """Compute Top-1 ACC, mean precision, recall, and F1 from a JSONL file."""
    total = 0
    correct = 0
    hindis, golds, predictions, precisions, recalls, f1s = [], [], [], [], [], []
    output_file = jsonl_path[:-5] + ".txt"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line.strip())
            hindi = item["hindi"]
            gold = item["gold"]
            pred = item["pred"]

            hindis.append(hindi)
            golds.append(gold)
            predictions.append(pred)

            total += 1
            if gold.lower() == pred.lower():
                correct += 1

            lcs = lcs_length(pred.lower(), gold.lower())
            if len(pred) == 0 or len(gold) == 0:
                precision = recall = f1 = 0.0
            else:
                precision = lcs / len(pred)
                recall = lcs / len(gold)
                f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    acc = correct / total if total > 0 else 0.0
    mean_p = sum(precisions) / total if total > 0 else 0.0
    mean_r = sum(recalls) / total if total > 0 else 0.0
    mean_f1 = sum(f1s) / total if total > 0 else 0.0

    with open(output_file, "w") as f:
        f.write(f"Top-1 Accuracy (ACC): {acc:.4f}\n")
        f.write(f"Mean Precision:       {mean_p:.4f}\n")
        f.write(f"Mean Recall:          {mean_r:.4f}\n")
        f.write(f"Mean F1 (Fuzziness):  {mean_f1:.4f}\n")

        f.write("\nWords for which F1-score < 0.5:\n")

        for i in range(len(predictions)):
            if f1s[i] < 0.5:
                f.write(f"Hindi: {hindis[i]:<20} Gold: {golds[i]:<20} Pred: {predictions[i]:<15}\n")

# Run it
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inputs", required=True, type=str, nargs="+")
    args = parser.parse_args()

    for input in args.inputs:
        compute_metrics(input)

