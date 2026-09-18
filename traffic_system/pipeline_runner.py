import json
import sys
from pathlib import Path

# Ensure we can import from src/
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.nlp_pipeline import BatchNLPPipeline

RAW_DIR_TWITTER = current_dir / "data" / "raw" / "twitter"
RAW_DIR_NEWS = current_dir / "data" / "raw" / "news"
PROCESSED_DIR = current_dir / "data" / "processed"

BATCH_SIZE = 16  # Process 16 texts at a time to manage memory/CPU effectively

def process_directory(raw_dir: Path, source_type: str, cleaner: TextCleaner, nlp: BatchNLPPipeline):
    print(f"\nScanning for {source_type} data in {raw_dir}...")
    if not raw_dir.exists():
        print(f"Directory {raw_dir} does not exist. Skipping.")
        return

    json_files = list(raw_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {raw_dir}.")
        return

    print(f"Found {len(json_files)} {source_type} file(s).")

    for file_path in json_files:
        rel_path = file_path.relative_to(raw_dir)
        out_file = PROCESSED_DIR / source_type.lower() / rel_path
        print(f"\nProcessing file: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        items = payload.get("data", [])
        if not items:
            print("No data found in file.")
            continue

        if out_file.exists():
            try:
                with out_file.open('r', encoding='utf-8') as f:
                    existing_payload = json.load(f)
                existing_items = existing_payload.get("data", [])
            except Exception:
                existing_items = []

            if len(existing_items) == len(items):
                print(f"Skipping existing output: {out_file}")
                continue

            print(
                f"Existing output is incomplete ({len(existing_items)}/{len(items)} records); "
                "reprocessing."
            )

        processed_records = []

        # Process in batches
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i + BATCH_SIZE]
            batch_raw_texts = []
            batch_meta = []

            # Step 1: Text Pre-cleaning & Entity Extraction
            for t in batch:
                # Handle both Twitter and News JSON structures
                raw_text = t.get("full_text") or t.get("text") or ""
                if not raw_text.strip():
                    continue

                batch_raw_texts.append(raw_text)
                batch_meta.append({
                    "record_id": str(t.get("id") or t.get("id_str") or ""),
                    "timestamp": t.get("created_at") or t.get("createdAt") or ""
                })

            if not batch_raw_texts:
                continue

            cleaned_results = [cleaner.clean(text) for text in batch_raw_texts]
            cleaned_texts_only = [res["cleaned_text"] for res in cleaned_results]

            # Step 2: Translation & Embedding (MarianMT -> DistilBERT)
            nlp_results = nlp.process_batch(cleaned_texts_only)

            # Step 3: Assemble into requested schema
            for meta, raw, clean_info, nlp_res in zip(batch_meta, batch_raw_texts, cleaned_results, nlp_results):
                record = {
                    "record_id": meta["record_id"],
                    "timestamp": meta["timestamp"],
                    "source_type": source_type,
                    "raw_text": raw,
                    "cleaned_text": clean_info["cleaned_text"],
                    "english_translation": nlp_res["english_translation"],
                    "embedding_vector": nlp_res["embedding_vector"],
                    "entities": list(clean_info["entities"])
                }
                processed_records.append(record)

        # Save processed data preserving the folder structure
        if processed_records:
            out_file.parent.mkdir(parents=True, exist_ok=True)

            final_payload = {
                "metadata": payload.get("metadata", {}),
                "data": processed_records
            }

            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(final_payload, f, ensure_ascii=False, indent=2)

            print(f"Success: Processed {len(processed_records)} records -> {out_file}")

def run_pipeline():
    print("Initializing NLP models (Cleaner, MarianMT, DistilBERT)...")
    cleaner = TextCleaner()
    nlp = BatchNLPPipeline()

    # Process both unstructured data sources
    process_directory(RAW_DIR_TWITTER, "X", cleaner, nlp)
    process_directory(RAW_DIR_NEWS, "News", cleaner, nlp)

if __name__ == "__main__":
    run_pipeline()