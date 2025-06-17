from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import gradio as gr

# Load the trained model and tokenizer from the saved directory
model_path = "./exported_model" # Or wherever your saved model is relative to app.py
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
print("Loading model and tokenizer...")


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode

# ... (your model loading code) ...

# Move model to GPU (or MPS on Apple Silicon) if available
if torch.cuda.is_available(): # For NVIDIA GPUs
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # For Apple Silicon (M-series)
    device = torch.device("mps")
else: # Fallback to CPU
    device = torch.device("cpu")

model.to(device)
model.eval() # Set model to evaluation mode

print(f"Model moved to {device} for inference.")

# Function to make predictions
def predict_sentiment(text):
    # Ensure input is string
    if not isinstance(text, str):
        return "Invalid input: Please provide a text string."

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Move inputs to the correct device (same as the model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the predicted class (0 for negative, 1 for positive)
    pred = torch.argmax(outputs.logits, dim=1).item()

    # Map the prediction to a human-readable sentiment
    return "Positive" if pred == 1 else "Negative"


print("Creating Gradio interface...")

# Simple examples for users to try
examples = [
    "I love this product! It's amazing and works perfectly.",
    "This movie was terrible and boring.",
    "The weather is nice today.",
    "Outstanding service! Highly recommended.",
    "I'm not sure how I feel about this."
]

# Create the interface with improvements
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=6, 
        placeholder="✨ Enter your text here...\n\nTry typing something like:\n• 'I love this!'\n• 'This is terrible'\n• 'It's okay, I guess'", 
        label="📝 Your Text"
    ),
    outputs=gr.Textbox(
        label="🎯 Sentiment Result",
        lines=2
    ),
    title="🤖 DistilBERT Sentiment Analyzer",
    description="🚀 **Analyze text sentiment instantly!** • Just type your text and see if it's positive or negative.",
    examples=examples,
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { 
            max-width: 700px !important; 
            margin: auto !important;
            font-family: 'Arial', sans-serif !important;
        }
        .gr-button-primary { 
            background: linear-gradient(45deg, #4facfe, #00f2fe) !important; 
            border: none !important;
            border-radius: 25px !important;
        }
    """
)

# Launch the Gradio app
if __name__ == "__main__":
    print("Launching Gradio app locally...")
    iface.launch()