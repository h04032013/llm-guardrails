import argparse
import random
from collections import defaultdict
from datasets import load_from_disk, Dataset

def balanced_sample(dataset, n, category_key, seed):
    random.seed(seed)

    groups = defaultdict(list)
    for i, row in enumerate(dataset):
        cat = row.get(category_key)
        groups[cat].append(i)

    for cat in groups:
        random.shuffle(groups[cat])

    categories = sorted(groups.keys())
    base = n // len(categories)
    remainder = n % len(categories)

    selected = []
    for i, cat in enumerate(categories):
        k = base + (1 if i < remainder else 0)
        selected.extend(groups[cat][: min(k, len(groups[cat]))])

    if len(selected) < n:
        used = set(selected)
        leftovers = [i for i in range(len(dataset)) if i not in used]
        random.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    random.shuffle(selected)
    return dataset.select(selected[:n])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--category_key", type=str, default="input_category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_from_disk(args.input_dir)

    if hasattr(dataset, "keys"):
        dataset = dataset["train"]

    sampled = balanced_sample(
        dataset=dataset,
        n=args.sample_size,
        category_key=args.category_key,
        seed=args.seed,
    )

    sampled.save_to_disk(args.output_dir)

    print(f"Loaded: {len(dataset)}")
    print(f"Saved: {len(sampled)}")
    print(f"Output: {args.output_dir}")

    counts = defaultdict(int)
    for row in sampled:
        counts[row.get(args.category_key) or "UNKNOWN"] += 1

    print("Category counts:")
    for cat, count in sorted(counts.items()):
        print(f"{cat}: {count}")


if __name__ == "__main__":
    main()