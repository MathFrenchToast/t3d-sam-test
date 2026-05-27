
import json
import os
from PIL import Image, ImageDraw

# Provided JSON data (fixed missing closing brace)
data_json = """
{
"landmarks": [
{
"name": "right_basket",
"position": "Right side of the court",
"bbox": [1600, 350, 1850, 650]
},
{
"name": "three_point_intersection_top",
"position": "Intersection of the three-point line and the right baseline (top side of image)",
"bbox": [1560, 730, 1590, 770]
}
],
"field_corners": {
"top_right": [1580, 720]
}}
"""

def main():
    data = json.loads(data_json)
    
    image_path = os.path.join("test", "basket3.png")
    output_path = os.path.join("test", "basket3_overlay.png")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Opening {image_path}...")
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Draw landmarks
    print("Drawing landmarks...")
    for landmark in data["landmarks"]:
        bbox = landmark["bbox"] # [xmin, ymin, xmax, ymax]
        name = landmark["name"]
        
        # Draw box
        draw.rectangle(bbox, outline="red", width=5)
        # Draw label
        draw.text((bbox[0], max(0, bbox[1] - 20)), name, fill="red")
        print(f" - {name} at {bbox}")
        
    # Draw field corners
    print("Drawing field corners...")
    corners = data.get("field_corners", {})
    points = []
    
    # Define expected order for polygon if present
    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    for name in order:
        if name in corners:
            points.append(tuple(corners[name]))

    # Draw points and labels for all available corners
    for name, pt in corners.items():
        r = 10
        draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r), fill="cyan", outline="white")
        draw.text((pt[0] + 15, pt[1]), name, fill="cyan")
        print(f" - Corner {name} at {pt}")

    # Draw polygon for court if we have at least 3 points
    if len(points) >= 3:
        draw.polygon(points, outline="cyan", width=5)
    elif len(points) == 2:
        draw.line(points, fill="cyan", width=5)
    
    image.save(output_path)
    print(f"Success: Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
