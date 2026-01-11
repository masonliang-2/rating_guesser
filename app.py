import tkinter as tk

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained(f"{DRIVE_MOUNT_PATH}{MODEL_OUTPUT_PATH}model")
tokenizer = AutoTokenizer.from_pretrained(f"{DRIVE_MOUNT_PATH}{MODEL_OUTPUT_PATH}tokenizer")



def predict(model, tokenizer, text, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()

    return pred

def on_submit():
    user_input = text_box.get("1.0", tk.END)

    clf = pipeline(
        "text-classification",
        model="Gummiworm/rating_guesser",
    )
    output = clf(user_input)
    label = out[0]["label"]

    rating = int(label.split("_")[-1]) + 1
    
    result_label.config(text=rating)

# Create main window
root = tk.Tk()
root.title("Simple Tkinter GUI")
root.geometry("400x300")

# Input label
input_label = tk.Label(root, text="Enter text:")
input_label.pack(pady=(20, 5))

# Multi-line text box
text_box = tk.Text(root, width=40, height=8)
text_box.pack(pady=5)

# Submit button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

# Output label
result_label = tk.Label(root, text="", fg="blue")
result_label.pack(pady=10)

# Start event loop
root.mainloop()
