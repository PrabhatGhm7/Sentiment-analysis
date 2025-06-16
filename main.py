import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import transformers
# Load dataset
df = pd.read_csv("sliced_data.csv", encoding='ISO-8859-1', header=None)
print('Dataset Loaded')

#Keeping only sentiment and text
df = df[[0, 5]]
df.columns = ['sentiment', 'text']

# keeping only 0 (negative) and 4 (positive)
df = df[df['sentiment'].isin([0, 4])]

df = df.sample(frac=0.01, random_state=42)  # Takes 10% of the data randomly

#converting negative to 0 and positive to 1
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

#Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42
)
print('Train-test split done')

#tokenizer loading
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print('Tokenizer loaded')
# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
print('Tokenization complete')
#Create PyTorch dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#Prepare datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

#Loading BERT
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print('Model loaded')
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="no",
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Trainer initialized.")

# Train
trainer.train()
print("Training complete.")

# Evaluate
trainer.evaluate()
print("Evaluation complete.")

# Save trained model and tokenizer locally
model.save_pretrained("exported_model")
tokenizer.save_pretrained("exported_model")
print("Model and tokenizer saved.")


#function to make predictions
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

# Test
sample_text = "I am very happy today!"
print(f"Input: {sample_text}")
print("Predicted Sentiment:", predict_sentiment(sample_text))