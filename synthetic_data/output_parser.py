import argparse
import json

def extract_after_assistantfinal(text: str) -> str:
    marker = "assistantfinal"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker):].strip()

def main(in_path: str, out_path: str):
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"JSON parse error on line {line_num}: {e}"
                ) from e

            gens = obj.get("generations", [])
            new_gens = []

            for g in gens:
                raw = g.get("text", "")
                cleaned = extract_after_assistantfinal(raw)

                new_g = dict(g)  # preserve finish_reason etc.
                new_g["text"] = cleaned
                new_gens.append(new_g)

            obj["generations"] = new_gens
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="Input JSONL path (raw output)")
    parser.add_argument("--out_path", required=True, help="Output JSONL path (cleaned)")

    args = parser.parse_args()
    main(args.in_path, args.out_path)