import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F  

model_path = "exported_model"  

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
labels = model.config.id2label  


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits  # raw output scores
        print(f"Logits: {logits}")  # Debugging line to check logits
        probs = F.softmax(logits, dim=1)  # convert logits to probabilities
        print(f"Probabilities: {probs}")
        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_id].item()  # confidence score (0 to 1)
        label = labels[predicted_class_id]
        if label.capitalize() == "Label_1":
            label = "Positive sentiment"
        elif label.capitalize() == "Label_0":
            label = "Negative sentiment"
        return f"{label} \n(Prediction confidence :{confidence*100:.2f}%)"


iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence here..."),
    outputs="text",
    title="Sentiment Analyzer",
    description="Predict whether the sentence expresses a Positive or Negative sentiment."
)

iface.launch()
