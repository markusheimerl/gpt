from datasets import load_dataset
from huggingface_hub import login
login()

TOTAL_SIZE_GB = 50
OUTPUT_FILE = "corpus.txt"
SOURCES = {
    "finePDFs":      ("HuggingFaceFW/finepdfs",          {},                     0.50, 2341667),
    "DCLM-baseline": ("mlfoundations/dclm-baseline-1.0", {},                     0.20, 4922678),
    "FineWeb-Edu":   ("HuggingFaceFW/fineweb-edu",       {"name": "sample-10BT"},0.20, 8865235),
    "LMSYS-Chat":    ("lmsys/lmsys-chat-1m",             {},                     0.10, 7746375),
}

def format_example(example):
    if "conversation" in example:
        body = "\n".join(f"<|{t['role']}|>\n{t['content']}" for t in example["conversation"])
    else:
        body = example["text"]
    return f"<|bos|>{body}\n"

def main():
    targets = {n: r * TOTAL_SIZE_GB * 1024**3 for n, (_, _, r, _) in SOURCES.items()}
    sizes = dict.fromkeys(SOURCES, 0)
    counts = dict.fromkeys(SOURCES, 0)

    print("Downloading and combining data...")
    for n in SOURCES:
        print(f"  {n}: {targets[n]/1024**3:.1f} GB ({SOURCES[n][2]:.0%})")

    iterators = {
        n: iter(load_dataset(path, split="train", streaming=True, **kw)
                .shuffle(seed=seed, buffer_size=10000))
        for n, (path, kw, _, seed) in SOURCES.items()
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while active := {n for n in SOURCES if sizes[n] < targets[n]}:
            source = min(active, key=lambda n: sizes[n] / targets[n])
            try:
                example = next(iterators[source])
            except StopIteration:
                print(f"Warning: {source} exhausted")
                targets[source] = 0
                continue

            if source == "LMSYS-Chat" and example.get("language") != "English":
                continue

            text = format_example(example)
            f.write(text)
            sizes[source] += len(text.encode("utf-8"))
            counts[source] += 1

            if sum(counts.values()) % 1000 == 0:
                total = sum(sizes.values())
                parts = " | ".join(f"{n}: {sizes[n]/1024**3:.2f}GB" for n in SOURCES)
                print(f"{sum(counts.values()):,} docs | {parts} | total: {total/1024**3:.2f}GB")

    print(f"\n✓ Done!")
    for n in SOURCES:
        print(f"  {n}: {counts[n]:,} docs ({sizes[n]/1024**3:.2f} GB)")
    print(f"  Total: {sum(sizes.values())/1024**3:.2f} GB")

if __name__ == "__main__":
    main()