import json

INPUT_FILE = "articles.txt"
OUTPUT_FILE = "dataset_articles.jsonl"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)  # Load the list of dicts

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} prompt-response pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()