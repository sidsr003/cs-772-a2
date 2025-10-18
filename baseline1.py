# ========================================
# CHARACTER MAPPING BASELINE EVALUATION
# ========================================

import json
import numpy as np
from tqdm import tqdm

# -------------------- Metric Functions --------------------

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def longest_common_subsequence_length(s1, s2):
    """Compute LCS length using formula from NEWS 2015"""
    return 0.5 * (len(s1) + len(s2) - edit_distance(s1, s2))

def compute_precision_recall_f1(pred, ref):
    """
    Compute precision, recall, and F1 for a single prediction
    Returns: (precision, recall, f1)
    """
    if len(pred) == 0 and len(ref) == 0:
        return 1.0, 1.0, 1.0
    elif len(pred) == 0 or len(ref) == 0:
        return 0.0, 0.0, 0.0
    
    # Compute LCS length
    lcs_len = longest_common_subsequence_length(pred, ref)
    
    # Precision and Recall
    precision = lcs_len / len(pred)
    recall = lcs_len / len(ref)
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1

def compute_word_accuracy(predictions, references):
    """Word Accuracy in Top-1 (ACC)"""
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
    return correct / len(predictions) if predictions else 0.0

def compute_metrics(predictions, references):
    """
    Compute all metrics: accuracy, mean precision, mean recall, mean F1
    Returns: (accuracy, mean_precision, mean_recall, mean_f1, per_example_metrics)
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        prec, rec, f1 = compute_precision_recall_f1(pred, ref)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    accuracy = compute_word_accuracy(predictions, references)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    per_example = list(zip(precisions, recalls, f1_scores))
    
    return accuracy, mean_precision, mean_recall, mean_f1, per_example

# -------------------- Character Mapping Dictionary --------------------

CHAR_MAP = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ॠ': 'rii', 'ऌ': 'lri', 'ए': 'e', 'ऐ': 'ai', 
    'ओ': 'o', 'औ': 'au',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'ळ': 'la', 'व': 'va',
    'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    
    # Matras (vowel signs)
    'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'ृ': 'ri', 'ॄ': 'rii', 'ॢ': 'lri', 'े': 'e', 'ै': 'ai',
    'ो': 'o', 'ौ': 'au',
    
    # Halant (virama)
    '्': '',
    
    # Special symbols
    'ं': 'n', 'ः': 'h', 'ँ': 'n', 'ऽ': '',
    
    # Additional consonants
    'क़': 'qa', 'ख़': 'kha', 'ग़': 'gha', 'ज़': 'za', 
    'ड़': 'da', 'ढ़': 'dha', 'फ़': 'fa',
    
    # Numbers
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

def baseline_char_map(hindi_word):
    """Simple character-by-character mapping"""
    result = []
    for char in hindi_word:
        if char in CHAR_MAP:
            result.append(CHAR_MAP[char])
        elif char == ' ':
            result.append(' ')
        elif char.isascii():
            result.append(char.lower())
        else:
            result.append('')  # Skip unknown chars
    return ''.join(result).lower()

# -------------------- Load Test Data --------------------

print("\n" + "="*70)
print("CHARACTER MAPPING BASELINE EVALUATION")
print("="*70)

print("\nLoading test data...")
test_data = []
with open('hin_test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line.strip())
            src = obj.get('native word', '')
            tgt = obj.get('english word', '')
            if src and tgt:
                test_data.append((src, tgt))
        except:
            continue

print(f"Loaded {len(test_data):,} test examples")

# -------------------- Generate Predictions --------------------

print("\nGenerating baseline predictions...")
baseline_predictions = []
baseline_references = []

for src_word, ref_word in tqdm(test_data, desc="Processing"):
    pred = baseline_char_map(src_word)
    baseline_predictions.append(pred)
    baseline_references.append(ref_word.lower())

# -------------------- Compute Metrics --------------------

print("\nComputing metrics...")
baseline_acc, baseline_prec, baseline_rec, baseline_f1, baseline_per_ex = compute_metrics(
    baseline_predictions, 
    baseline_references
)

# -------------------- Display Results --------------------

print("\n" + "="*70)
print("BASELINE RESULTS")
print("="*70)

print(f"\nTop-1 Accuracy (ACC): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Mean Precision:       {baseline_prec:.4f}")
print(f"Mean Recall:          {baseline_rec:.4f}")
print(f"Mean F1 (Fuzziness):  {baseline_f1:.4f}")

# -------------------- Low F1 Examples --------------------

print("\n" + "="*70)
print("Words for which F1-score < 0.5:")
print("="*70)

baseline_low_f1 = []
for i, (prec, rec, f1) in enumerate(baseline_per_ex):
    if f1 < 0.5:
        src_word = test_data[i][0]
        ref = baseline_references[i]
        pred = baseline_predictions[i]
        baseline_low_f1.append((src_word, ref, pred, f1, prec, rec))

print(f"\nTotal: {len(baseline_low_f1)} / {len(baseline_references)} ({len(baseline_low_f1)/len(baseline_references)*100:.2f}%)")

if baseline_low_f1:
    print(f"\nShowing first 20 examples:")
    print(f"\n{'Source':<20} {'Reference':<20} {'Prediction':<20} {'F1':<8} {'Prec':<8} {'Rec':<8}")
    print("-"*100)
    
    for src, ref, pred, f1, prec, rec in baseline_low_f1[:20]:
        print(f"{src:<20} {ref:<20} {pred:<20} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f}")

# -------------------- Sample Predictions --------------------

print("\n" + "="*70)
print("SAMPLE PREDICTIONS (First 15)")
print("="*70)
print(f"\n{'Source':<20} {'Reference':<20} {'Baseline Prediction':<20} {'Match':<8}")
print("-"*70)

for i in range(min(15, len(test_data))):
    src = test_data[i][0]
    ref = baseline_references[i]
    pred = baseline_predictions[i]
    match = "✓" if pred == ref else "✗"
    print(f"{src:<20} {ref:<20} {pred:<20} {match:<8}")

print("="*70)

print(f"\n✅ Baseline evaluation complete!")
