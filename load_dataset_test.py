from datasets import load_dataset

val_data = load_dataset("allenai/c4", "en", split="validation", download_mode="force_redownload")
print("OK!")