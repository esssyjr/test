import json
import os
from mcp.server.fastmcp import FastMCP
import os
import tempfile
import numpy as np
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from ultralytics import YOLO
from utils import (
    run_gemini_diagnosis,
    run_gemini_description,
    image_to_base64,
    calculate_area,
    load_json
)

# Constants 
PIXEL_SPACING_CM = 16.93 / 640
TREATMENT_FILE = "resources/treatment_plan.json"
DOCTOR_CONTACTS_FILE = "resources/doctor.json"

# Keep using the same MCP instance as your other tools/resources
mcp = FastMCP("test", host = '0.0.0.0', port = 8001)
yolo_model = YOLO("wound_models/wound_model_1.pt")

# ✅ Simple tool (for MCP test)
@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Tool 1
@mcp.tool()
def segment_wound(image_path: str, conf_thresh: float = 0.1) -> dict:
    """Segment wound, calculate area, return Base64 image."""
    image = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image)

    results = yolo_model.predict(image_np, conf=conf_thresh, save=False, imgsz=640)
    masks = results[0].masks

    if masks is None:
        diagnosis = run_gemini_diagnosis(image)
        return {
            "message": "No wound detected.",
            "area_cm2": 0,
            "diagnosis": diagnosis,
            "annotated_image": image_path,
            "annotated_image_base64": None
        }

    mask_data = masks.data.cpu().numpy()
    area_cm2 = calculate_area(mask_data, PIXEL_SPACING_CM)

    annotated_array = results[0].plot()
    annotated_img = PILImage.fromarray(annotated_array)

    temp_path = tempfile.mktemp(suffix=".png")
    annotated_img.save(temp_path)

    img_base64 = image_to_base64(annotated_img)
    wound_count = mask_data.shape[0]

    return {
        "area_cm2": round(area_cm2, 2),
        "annotated_image_base64": img_base64
    }


# Resource 1
@mcp.resource("treatment://{wound_type}")
def get_treatment_plan(wound_type: str) -> str:
    try:
        treatment_data = load_json(TREATMENT_FILE)
        if wound_type not in treatment_data:
            return f"# No treatment plan found for: {wound_type}"
        plan = treatment_data[wound_type]

        content = f"# {plan['name']}\n\n**Description**: {plan['description']}\n\n"
        if "materials" in plan:
            content += "## Materials Needed\n" + "\n".join(f"- {m}" for m in plan["materials"]) + "\n\n"
        if "steps" in plan:
            for idx, step in sorted(plan["steps"].items(), key=lambda x: int(x[0])):
                content += f"{idx}. {step}\n"
        if "precautions" in plan:
            content += f"\n**Precautions**: {plan['precautions']}"
        if "when_to_seek_help" in plan:
            content += f"\n**When to Seek Help**: {plan['when_to_seek_help']}"
        if "duration" in plan:
            content += f"\n**Estimated Duration**: {plan['duration']}"
        if "is_self_treatable" in plan:
            content += f"\n**Self-Treatable**: {'Yes' if plan['is_self_treatable'] else 'No'}"
        return content.strip()
    except Exception as e:
        return f"# Error reading treatment plans: {e}"


# ✅ Run the server
if __name__ == "__main__":
    mcp.run(transport='streamable-http')
