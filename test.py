import torch
import json
from PIL import Image, ImageDraw
import os
from dotenv import load_dotenv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load the model
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    # Determine the image path
    image_dir = "test"
    image_path = os.path.join(image_dir, "basketball-frame.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, "basket3.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError("Could not find basketball image in the test directory.")
        
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    
    # Prompt the model with text
    prompt = "players of both team, referees, court landmark - for a later homography, including the backet"
    print(f"Sending prompt: '{prompt}'")
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # Get the masks, bounding boxes, and scores
    masks = output.get("masks", [])
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    
    boxes_list = boxes.tolist() if hasattr(boxes, 'tolist') else boxes
    scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores

    # Prepare JSON data
    results = {
        "prompt": prompt,
        "image": image_path,
        "detections": []
    }
    
    draw = ImageDraw.Draw(image)
    
    for i, box in enumerate(boxes_list):
        score = float(scores_list[i]) if i < len(scores_list) else None
        results["detections"].append({
            "id": i,
            "box": box,
            "score": score
        })
        # Draw the bounding box for visual check
        if len(box) == 4:
            draw.rectangle(box, outline="red", width=3)
            # Optional: Add score text
            if score is not None:
                draw.text((box[0], max(0, box[1]-10)), f"{score:.2f}", fill="red")

    # Output JSON
    json_path = "output.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {json_path}")
    
    # Output visual check image
    output_image_path = "output_visual.jpg"
    image.save(output_image_path)
    print(f"Visual check image saved to {output_image_path}")

if __name__ == "__main__":
    main()