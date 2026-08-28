import torch
import os
from torch.utils.data import DataLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.dataset import TwitterTrafficDataset
from src.preprocessing.nlp_pipeline import BatchNLPPipeline

def main():
    # File paths
    gazetteer_path = "configs/gazetteer.json"
    # Uses all date/time partitions created by the CCTV-aligned collector.
    raw_data_path = "data/raw/twitter"
    output_path = "data/processed/embeddings.pt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Initializing components and downloading models...")
    cleaner = TextCleaner(gazetteer_path)
    dataset = TwitterTrafficDataset(raw_data_path, cleaner)
    pipeline = BatchNLPPipeline()

    # DataLoader for batching. Reduce batch_size if running out of RAM/VRAM.
    batch_size = 16 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    print(f"Processing {len(dataset)} tweets in batches of {batch_size}...")
    for i, batch_texts in enumerate(dataloader):
        print(f"Processing batch {i + 1}/{len(dataloader)}")
        embeddings = pipeline.process_batch(batch_texts)
        all_embeddings.append(embeddings.cpu())

    # Concatenate and save the final tensor
    final_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    torch.save(final_embeddings_tensor, output_path)
    
    print(f"Processing complete! Embeddings saved to {output_path}")
    print(f"Final tensor shape: {final_embeddings_tensor.shape}")

if __name__ == "__main__":
    main()
