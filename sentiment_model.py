from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import accelerate


df = pd.read_csv('sample_reviews.csv')

#make label binaryy
df['label'] = (df['label'] >= 3).astype(int)   # 3-4 = positive (1)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# To Dataset (ensure columns are 'text' and 'label')
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)  # Add max_length to avoid OOM

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
    fp16=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


# After trainer.train() finishes
eval_results = trainer.evaluate()
print("Final evaluation results:", eval_results)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print("Confusion Matrix:\n", confusion_matrix(labels, preds))
print("\nClassification Report:\n", classification_report(labels, preds, target_names=["Negative", "Positive"]))


wrong_idx = np.where(preds != labels)[0][:5]  # First 5 errors
for idx in wrong_idx:
    print(f"Text: {test_df.iloc[idx]['text']}")
    print(f"True: {labels[idx]}, Predicted: {preds[idx]}\n")

#save the model
trainer.save_model("./fine_tuned_distilbert_sentiment")  #
tokenizer.save_pretrained("./fine_tuned_distilbert_sentiment")


#load the model and push to hugging face
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load the saved model and tokenizer from your local folder
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_distilbert_sentiment")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_distilbert_sentiment")

#Push to hugging face for live demo
from huggingface_hub import login
login()

# 3. Push model and tokenizer to the Hub
repo_id = "MathProblem/amazon-sentiment-distilbert"
model.push_to_hub(repo_id, private=False)
tokenizer.push_to_hub(repo_id, private=False)

#Test it is on hugging face
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="MathProblem/amazon-sentiment-distilbert")
print(classifier("This product is absolutely terrible and broke after one use."))
