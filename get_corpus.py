import requests
import pyarrow.parquet as pq
import os
import random
from huggingface_hub import login, list_repo_files, get_token

TOTAL_SIZE_GB = 50
TARGET_BYTES  = TOTAL_SIZE_GB * 1024**3
OUTPUT_FILE   = "corpus.txt"

SOURCES = {
    "finePhrases": ("HuggingFaceFW/finephrase", None, 0.70, 123456),
    "LMSYS-Chat":  ("lmsys/lmsys-chat-1m",      None, 0.30, 452387),
}

def get_parquet_files(repo_id, path_filter=None):
    files = list_repo_files(repo_id, repo_type="dataset")
    files = [f for f in files if f.endswith(".parquet")]
    if path_filter:
        files = [f for f in files if path_filter in f]
    return sorted(files)

def download_shard(repo_id, fname, tmp_path):
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{fname}"
    print(f"  Downloading [{repo_id}]: {fname}")
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

def load_shard(name, fname, seed):
    repo_id = SOURCES[name][0]
    tmp = f"tmp_{name}.parquet"
    download_shard(repo_id, fname, tmp)
    cols = ["conversation", "language"] if name == "LMSYS-Chat" else ["text"]
    table = pq.read_table(tmp, columns=cols)
    os.remove(tmp)
    rows = table.to_pylist()
    del table
    if name == "LMSYS-Chat":
        rows = [r for r in rows if r.get("language") == "English"]
    random.Random(seed).shuffle(rows)
    return rows

def refill_buffer(name, buffers, shard_lists, indices):
    idx = indices[name]
    if idx >= len(shard_lists[name]):
        return False
    seed = SOURCES[name][3]
    rows = load_shard(name, shard_lists[name][idx], seed + idx)
    buffers[name].extend(rows)
    indices[name] += 1
    return True

def format_row(name, row):
    if name == "LMSYS-Chat":
        body = "\n".join(f"<|{t['role']}|>\n{t['content']}" for t in row["conversation"])
    else:
        body = row["text"]
    return f"<|bos|>{body}\n"

def main():
    targets   = {n: r * TARGET_BYTES for n, (_, _, r, _) in SOURCES.items()}
    sizes     = dict.fromkeys(SOURCES, 0)
    indices   = dict.fromkeys(SOURCES, 0)
    doc_count = 0

    shard_lists = {}
    for name, (repo_id, path_filter, _, seed) in SOURCES.items():
        print(f"Listing shards for {name}...")
        shards = get_parquet_files(repo_id, path_filter)
        random.Random(seed).shuffle(shards)
        shard_lists[name] = shards
        print(f"  {len(shards)} shards found")

    buffers = {n: [] for n in SOURCES}
    for name in SOURCES:
        refill_buffer(name, buffers, shard_lists, indices)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        while sum(sizes.values()) < TARGET_BYTES:

            # sources that still have data
            active = {n for n in SOURCES
                      if buffers[n] or indices[n] < len(shard_lists[n])}
            if not active:
                print("All sources exhausted before reaching target!")
                break

            # pick source most behind its ratio, among those still available
            source = min(active, key=lambda n: sizes[n] / targets[n])

            # refill if running low
            if len(buffers[source]) < 100:
                refill_buffer(source, buffers, shard_lists, indices)

            if not buffers[source]:
                # this source truly ran out, adjust its target so others fill in
                print(f"  {source} exhausted at {sizes[source]/1024**3:.2f} GB")
                targets[source] = sizes[source]  # freeze its target
                # redistribute its remaining quota proportionally to others
                leftover = TARGET_BYTES - sum(targets.values())
                remaining_sources = [n for n in SOURCES if n != source]
                total_ratio = sum(SOURCES[n][2] for n in remaining_sources)
                for n in remaining_sources:
                    targets[n] += leftover * (SOURCES[n][2] / total_ratio)
                continue

            row  = buffers[source].pop()
            text = format_row(source, row)
            out.write(text)
            sizes[source] += len(text.encode("utf-8"))
            doc_count += 1

            if doc_count % 10000 == 0:
                total = sum(sizes.values())
                parts = " | ".join(f"{n}: {sizes[n]/1024**3:.2f}GB" for n in SOURCES)
                print(f"{doc_count:,} docs | {parts} | total: {total/1024**3:.2f}GB")

    print(f"\nDone!")
    for n in SOURCES:
        print(f"  {n}: {sizes[n]/1024**3:.2f} GB")
    print(f"  Total: {sum(sizes.values())/1024**3:.2f} GB")

if __name__ == "__main__":
    main()