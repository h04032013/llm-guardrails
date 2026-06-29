import json
import random
from collections import defaultdict
from pathlib import Path

INPUT_FILE = "/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/JSONL_FORMAT/ShenLab_MentalChat16K/gpt-4.1-mini/16084samples.jsonl"      # change this
OUTPUT_FILE = "/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/JSONL_FORMAT/ShenLab_MentalChat16K/gpt-4.1-mini/500samples.jsonl"
TOTAL_TARGET = 500
LABEL_FIELD = "response"
SEED = 42

random.seed(SEED)

# Load JSONL
rows = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# Group by label
by_label = defaultdict(list)
for row in rows:
    by_label[row[LABEL_FIELD]].append(row)

labels = sorted(by_label.keys())
num_labels = len(labels)

# Shuffle each label group
for label in labels:
    random.shuffle(by_label[label])

# Balanced sampling with rare-class preservation
remaining_target = min(TOTAL_TARGET, len(rows))
selected = []

remaining_labels = set(labels)

while remaining_labels and remaining_target > 0:
    quota = max(1, remaining_target // len(remaining_labels))
    finished = []

    for label in list(remaining_labels):
        available = len(by_label[label])

        if available <= quota:
            take = available
            finished.append(label)
        else:
            take = quota

        selected.extend(by_label[label][:take])
        by_label[label] = by_label[label][take:]
        remaining_target -= take

        if remaining_target <= 0:
            break

    for label in finished:
        remaining_labels.remove(label)

# Final shuffle
random.shuffle(selected)

# Save JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(selected)} samples to {OUTPUT_FILE}")

# Print label counts
counts = defaultdict(int)
for row in selected:
    counts[row[LABEL_FIELD]] += 1

print("\nSample counts:")
for label in labels:
    print(f"{label}: {counts[label]}")