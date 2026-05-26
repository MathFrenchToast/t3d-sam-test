import torch
import json
import numpy as np
from PIL import Image, ImageDraw
import os
from dotenv import load_dotenv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_cxcywh_to_xyxy

def main():
    load_dotenv()

    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    image_path = os.path.join("test", "basketball-frame.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join("test", "basket3.png")
        
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # 1. TARGETED PROMPTS
    prompts = [
        "basketball player wearing a white jersey uniform", 
        "basketball player wearing a blue jersey uniform", 
        "referee in black and white vertical striped shirt", 
        "white lines and landmarks on the basketball court", 
        "basketball hoop and orange rim",
        "wooden basketball court floor",       # Core ROI (5)
        "spectators in the background",        # Noise (6)
        "people sitting on the bench"          # Noise (7)
    ]
    
    colors = [
        (255, 255, 255), # White team
        (0, 100, 255),   # Blue team
        (255, 255, 0),   # Referees
        (0, 255, 0),     # Landmarks
        (255, 100, 0),   # Basket
    ]
    
    print(f"Running multi-class competition with weighted spatial grounding...")
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(image)
        num_prompts = len(prompts)
        processor.find_stage.text_ids = torch.arange(num_prompts, device="cuda")
        processor.find_stage.img_ids = torch.zeros(num_prompts, dtype=torch.long, device="cuda")
        
        text_outputs = model.backbone.forward_text(prompts, device="cuda")
        inference_state["backbone_out"].update(text_outputs)
        inference_state["geometric_prompt"] = model._get_dummy_prompt(num_prompts=num_prompts)
            
        outputs = model.forward_grounding(
            backbone_out=inference_state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=inference_state["geometric_prompt"],
            find_target=None,
        )

        out_logits = outputs["pred_logits"].sigmoid().squeeze(-1)
        presence_score = outputs["presence_logit_dec"].sigmoid()
        final_scores = out_logits * presence_score
        
        scores_per_query = final_scores.t()
        best_scores, best_prompt_indices = scores_per_query.max(dim=-1)
        
        # BALANCE: threshold set to 0.20 to filter weak detections
        keep = best_scores > 0.20 
        keep_indices = torch.where(keep)[0]

    # --- STRICT COURT ROI (Wooden Floor Only) ---
    # We use only the wooden floor to keep the boundary tight
    court_mask = np.zeros((height, width), dtype=bool)
    floor_prompt_idx = 5
    for idx in keep_indices:
        if best_prompt_indices[idx].item() == floor_prompt_idx:
            m_logits = outputs["pred_masks"][floor_prompt_idx, idx].unsqueeze(0).unsqueeze(0)
            m = torch.nn.functional.interpolate(m_logits, (height, width), mode="bilinear").sigmoid().squeeze() > 0.5
            court_mask = np.logical_or(court_mask, m.cpu().numpy())
    
    # NO DILATION: Keep the floor boundary strict
    # ---------------------------------------------

    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    results = {"detections": []}

    for idx in keep_indices:
        p_idx = best_prompt_indices[idx].item()
        score = best_scores[idx].item()
        
        if p_idx >= 5: continue 
            
        # Get object mask
        mask_logits = outputs["pred_masks"][p_idx, idx].unsqueeze(0).unsqueeze(0)
        mask_upscaled = torch.nn.functional.interpolate(mask_logits, (height, width), mode="bilinear").sigmoid().squeeze() > 0.5
        mask_np = mask_upscaled.cpu().numpy()
        
        # WEIGHTED SPATIAL GROUNDING
        if p_idx in [0, 1, 2]: # Players/Referees
            # 1. Get the bounding box to find the bottom region
            box_cxcywh = outputs["pred_boxes"][p_idx, idx]
            box_xyxy = box_cxcywh_to_xyxy(box_cxcywh.unsqueeze(0)).squeeze(0)
            b = (box_xyxy * torch.tensor([width, height, width, height], device="cuda")).tolist()
            
            # 2. Define the "Foot Region" (Bottom 25% of the detection)
            y_foot_start = int(b[1] + (b[3] - b[1]) * 0.75)
            y_foot_end = int(b[3])
            x_start = int(b[0])
            x_end = int(b[2])
            
            # Ensure indices are within bounds
            y_foot_start = max(0, min(height-1, y_foot_start))
            y_foot_end = max(0, min(height, y_foot_end))
            x_start = max(0, min(width-1, x_start))
            x_end = max(0, min(width, x_end))
            
            # 3. Check for overlap in the FOOT REGION only
            # Bench players will have their feet on the sideline/bench area, not the wooden floor ROI
            foot_mask_slice = mask_np[y_foot_start:y_foot_end, x_start:x_end]
            court_mask_slice = court_mask[y_foot_start:y_foot_end, x_start:x_end]
            
            overlap = np.logical_and(foot_mask_slice, court_mask_slice)
            if not np.any(overlap):
                continue
        
        # If we reached here, the detection is grounded on the court
        box_cxcywh = outputs["pred_boxes"][p_idx, idx]
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh.unsqueeze(0)).squeeze(0)
        box = (box_xyxy * torch.tensor([width, height, width, height], device="cuda")).tolist()
        
        color = colors[p_idx]
        results["detections"].append({"class": prompts[p_idx], "box": box, "score": score})
        
        mask_layer = np.zeros((height, width, 4), dtype=np.uint8)
        mask_layer[mask_np] = list(color) + [130]
        vis_image.paste(Image.fromarray(mask_layer, 'RGBA'), (0, 0), Image.fromarray(mask_layer, 'RGBA'))
        
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1]-10)), f"{score:.2f}", fill=color)

    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)
    vis_image.save("output_visual.jpg")
    print(f"\nFinalized {len(results['detections'])} grounded on-court detections.")

if __name__ == "__main__":
    main()
