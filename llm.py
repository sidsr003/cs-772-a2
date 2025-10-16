import json
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import os

# -------------------------
# CONFIGURATION
# -------------------------
load_dotenv("./.env")
MODEL_NAME = "gpt-4.1"
TEST_FILE = "./data/hin_test.jsonl"
BATCH_SIZE = 50

client = OpenAI()

# -------------------------
# STEP 1 — Load dataset
# -------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


# -------------------------
# STEP 2 — Query the LLM (batched)
# -------------------------
def transliterate_batch_with_gpt(batch, temp=1.0, top_p=1.0):
    # batch is a list of (uid, hindi_word)
    joined = "\n".join([f"{i+1}. {h}" for i, (_, h) in enumerate(batch)])
    prompt = (
        "You are a transliteration assistant.\n"
        "Transliterate each of the following Hindi (Devanagari) words into English (Roman script), "
        "preserving pronunciation.\n"
        "Do not use accents on the."
        "Return only the English transliterations, one per line, numbered exactly the same.\n\n"
        f"{joined}\n\nEnglish transliterations:\n"
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
        temperature=temp,
        top_p = top_p
    )
    # Extract text output safely
    try:
        output_text = response.output_text
    except Exception as e:
        print("Error parsing response:", e)
        return [""] * len(batch)

    # Split output into lines and map
    lines = [line.strip() for line in output_text.split("\n") if line.strip()]
    preds = []
    for i, (_, hindi_word) in enumerate(batch):
        # Try to strip leading numbering like "1. maitrIlogist"
        if i < len(lines):
            pred = lines[i].split(".", 1)[-1].strip() if "." in lines[i] else lines[i]
        else:
            pred = ""
        preds.append(pred)
    print(preds)
    return preds


# -------------------------
# STEP 3 — Evaluation metrics
# -------------------------
def compute_metrics(preds, golds):
    correct = sum(p.lower() == g.lower() for p, g in zip(preds, golds))
    word_accuracy = correct / len(golds)

    all_true_chars, all_pred_chars = [], []
    for g, p in zip(golds, preds):
        for gt_char, pr_char in zip(g, p):
            all_true_chars.append(gt_char)
            all_pred_chars.append(pr_char)

    labels = sorted(list(set(all_true_chars + all_pred_chars)))
    cm = confusion_matrix(all_true_chars, all_pred_chars, labels=labels)

    precision = precision_score(all_true_chars, all_pred_chars, average="macro", zero_division=0)
    recall = recall_score(all_true_chars, all_pred_chars, average="macro", zero_division=0)
    f1 = f1_score(all_true_chars, all_pred_chars, average="macro", zero_division=0)

    return {
        "word_accuracy": word_accuracy,
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }


# -------------------------
# STEP 4 — Main
# -------------------------
def main():
    test_data = load_jsonl(TEST_FILE)

    temps = [0.1, 0.5, 1.0]
    top_p = [0.1, 0.5, 1.0]

    for t in temps:
        predictions, golds, ids = [], [], []

        # Break dataset into chunks
        for i in tqdm(range(0, len(test_data[:4000]), BATCH_SIZE)):
            batch = test_data[i : min(len(test_data), i + BATCH_SIZE)]
            batch_pairs = [(s["unique_identifier"], s["native word"]) for s in batch]

            try:
                batch_preds = transliterate_batch_with_gpt(batch_pairs, temp=t)
            except Exception as e:
                print(f"Error in batch {i//BATCH_SIZE}: {e}")
                batch_preds = [""] * len(batch_pairs)

            for sample, pred in zip(batch, batch_preds):
                uid = sample["unique_identifier"]
                hindi = sample["native word"]
                gold = sample["english word"]

                predictions.append(pred)
                golds.append(gold)
                ids.append(uid)

                with open(f"llm_predictions/temp={t}.jsonl", "a", encoding="utf-8") as out:
                    json.dump({"id": uid, "hindi": hindi, "pred": pred, "gold": gold}, out, ensure_ascii=False)
                    out.write("\n")
    for p in top_p:
        predictions, golds, ids = [], [], []

        # Break dataset into chunks
        for i in tqdm(range(0, len(test_data[:4000]), BATCH_SIZE)):
            batch = test_data[i : min(len(test_data), i + BATCH_SIZE)]
            batch_pairs = [(s["unique_identifier"], s["native word"]) for s in batch]

            try:
                batch_preds = transliterate_batch_with_gpt(batch_pairs, top_p=p)
            except Exception as e:
                print(f"Error in batch {i//BATCH_SIZE}: {e}")
                batch_preds = [""] * len(batch_pairs)

            for sample, pred in zip(batch, batch_preds):
                uid = sample["unique_identifier"]
                hindi = sample["native word"]
                gold = sample["english word"]

                predictions.append(pred)
                golds.append(gold)
                ids.append(uid)

                with open(f"llm_predictions/p={p}.jsonl", "a", encoding="utf-8") as out:
                    json.dump({"id": uid, "hindi": hindi, "pred": pred, "gold": gold}, out, ensure_ascii=False)
                    out.write("\n")

if __name__ == "__main__":
    main()
