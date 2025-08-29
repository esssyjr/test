import os
import io
import json
import base64
import logging
import numpy as np
from PIL import Image as PILImage
from google import generativeai

logger = logging.getLogger(__name__)

#  Gemini Configuration 
def configure_gemini():
    """Configure and return a Gemini model instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    generativeai.configure(api_key=api_key)
    return generativeai.GenerativeModel("gemini-1.5-flash")

#  Image Conversion 
def pil_to_bytes(image: PILImage.Image) -> bytes:
    """Convert PIL image to bytes (PNG format)."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def image_to_base64(image: PILImage.Image) -> str:
    """Convert a PIL image to Base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

#  Gemini Calls 
def run_gemini_diagnosis(image: PILImage.Image) -> str:
    """Ask Gemini if wound is infected or not."""
    try:
        model = configure_gemini()
        img_bytes = pil_to_bytes(image)

        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"mime_type": "image/png", "data": img_bytes},
                        "Based on this wound image, is it infected or not infected? Respond with only one word: 'infected' or 'not infected'."
                    ]
                }
            ]
        )

        text = response.text.strip().lower()
        if "infected" in text and "not" not in text:
            return "infected"
        elif "not" in text:
            return "not_infected"
        else:
            return "Unable to determine"
    except Exception as e:
        logger.error(f"Gemini diagnosis failed: {e}")
        return f"Error: {str(e)}"

def run_gemini_description(image: PILImage.Image) -> str:
    """Get a short wound description from Gemini."""
    try:
        model = configure_gemini()
        img_bytes = pil_to_bytes(image)

        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"mime_type": "image/png", "data": img_bytes},
                        "Describe this wound, including anything you can see, color, surrounding skin condition, and any visible signs of infection. Make it short and precise."
                    ]
                }
            ]
        )

        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini wound description failed: {e}")
        return f"Error: {str(e)}"

#  Area Calculation 
def calculate_area(mask_data, pixel_spacing_cm: float) -> float:
    """Calculate wound area in cm² from mask data."""
    total_pixels = sum(np.sum(mask) for mask in mask_data)
    return total_pixels * (pixel_spacing_cm ** 2)


#  JSON Handling 
def load_json(file_path: str) -> dict:
    """Load JSON from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)
