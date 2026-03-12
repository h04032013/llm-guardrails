import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to sampled_data.parquet")
    p.add_argument("--n", type=int, default=5, help="How many rows to print")
    p.add_argument("--user_only", action="store_true", help="Only show rows where party == USER")
    args = p.parse_args()

    df = pd.read_parquet(args.data)

    if args.user_only and "party" in df.columns:
        df = df[df["party"] == "USER"]

    print("Total rows (after filters):", len(df))
    for i in range(min(args.n, len(df))):
        row = df.iloc[i]
        print("\n" + "=" * 100)
        if "id" in df.columns and "subreddit" in df.columns:
            print(f"Row {i} | id={row['id']} | subreddit={row['subreddit']} | party={row.get('party','NA')} | turn={row.get('turn','NA')}")
        print("-" * 100)
        print(row["extracted_dialogue"])

if __name__ == "__main__":
    main()