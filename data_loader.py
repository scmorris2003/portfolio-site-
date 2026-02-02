from datasets import load_dataset
import pandas as pd

dataset = load_dataset("SetFit/amazon_reviews_multi_en")
df = pd.DataFrame(dataset['train'])
print(df.head())

# Save a sample for local work
df.sample(10000).to_csv('sample_reviews.csv', index=False)