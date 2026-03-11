import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="View first few conversations from parquet dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to sampled_data.parquet")
    parser.add_argument("--n", type=int, default=3, help="Number of conversations to display")

    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))
    print("Unique conversations:", df["id"].nunique())

    for convo_id in df["id"].unique()[:args.n]:
        sub = df[df["id"] == convo_id].sort_values("turn")

        print("\n" + "="*80)
        print(f"CONVO ID: {convo_id} | Subreddit: {sub['subreddit'].iloc[0]}")
        print("-"*80)

        for _, row in sub.iterrows():
            print(f"{row['party']}: {row['text']}")

if __name__ == "__main__":
    main()