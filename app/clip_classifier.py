# app/clip_classifier.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# A focused list of common dish names CLIP will classify against
DISH_LABELS = [
    "dal makhani", "butter chicken", "biryani", "paneer tikka", "chole bhature",
    "aloo gobi", "rajma", "palak paneer", "samosa", "dosa",
    "idli", "upma", "poha", "paratha", "roti",
    "fried rice", "noodles", "pasta", "pizza", "burger",
    "soup", "salad", "sandwich", "omelette", "pancakes",
    "grilled chicken", "fish curry", "prawn masala", "mutton curry", "vegetable stew"
]


def load_clip_model():
    """
    Load CLIP model and processor onto GPU if available, else CPU.
    Call this once at app startup and reuse.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model ready.")
    return model, processor, device


def classify_dish(image: Image.Image, model, processor, device, top_k: int = 1) -> str:
    """
    Given a PIL image, return the most likely dish name from DISH_LABELS.

    Args:
        image: PIL Image uploaded by user
        model: loaded CLIPModel
        processor: loaded CLIPProcessor
        device: "cuda" or "cpu"
        top_k: how many top labels to return (default 1 — just the best match)

    Returns:
        Dish name string (e.g. "biryani") to use as the query
    """
    text_inputs = [f"a photo of {label}" for label in DISH_LABELS]

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # shape: (1, num_labels)
        probs = logits.softmax(dim=1)      # convert to probabilities

    top_idx = probs[0].argmax().item()
    top_label = DISH_LABELS[top_idx]
    top_prob = probs[0][top_idx].item()

    print(f"CLIP prediction: '{top_label}' with confidence {top_prob:.2%}")
    return top_label