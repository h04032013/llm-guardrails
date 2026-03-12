import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Print full rows with one column per line.")
    p.add_argument("--data", required=True, help="Path to parquet file")
    p.add_argument("--n", type=int, default=3, help="Number of rows to print")
    p.add_argument("--start", type=int, default=0, help="Start row index (after optional filtering)")
    p.add_argument("--user_only", action="store_true", help="Only show rows where party == USER")
    args = p.parse_args()

    df = pd.read_parquet(args.data)

    if args.user_only and "party" in df.columns:
        df = df[df["party"] == "USER"]

    df = df.reset_index(drop=True)

    end = min(args.start + args.n, len(df))
    print(f"Total rows (after filters): {len(df)}")
    print(f"Showing rows [{args.start}:{end})")

    for i in range(args.start, end):
        row = df.iloc[i]
        print("\n" + "=" * 120)
        print(f"ROW {i}")
        print("-" * 120)
        for col in df.columns:
            val = row[col]
            print(f"{col}: {val}")

if __name__ == "__main__":
    main()