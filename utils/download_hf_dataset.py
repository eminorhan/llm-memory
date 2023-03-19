from datasets import load_dataset

# Load the wikitext-103-raw-v1 dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
print(dataset)