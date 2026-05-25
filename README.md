# T3D SAM Test

A test project for Meta's Segment Anything Model (SAM) to extract bounding boxes from images using text prompts.

## Prerequisites

- Python 3.8+
- A valid [Hugging Face Access Token](https://huggingface.co/settings/tokens) with access to the `facebook/sam3` repository.

## Setup Instructions

We recommend using a standard Python virtual environment (`venv`) to keep your dependencies isolated.

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

Choose one of the following methods depending on whether you want CPU-only support or GPU (CUDA) acceleration.

**Option A: CPU-Only (Recommended for machines without a dedicated Nvidia GPU)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Option B: With CUDA Support (For Nvidia GPU acceleration)**
```bash
pip install torch torchvision
```

### 3. Install Requirements
Install the remaining packages from the requirements file:
```bash
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face
Because SAM is a gated model, you must provide your Hugging Face token. You can either export it in your terminal or create a `.env` file in the project root containing:
```env
HF_TOKEN="your_hf_token_here"
```

## Running the Script

1. Ensure your test images (`basketball-frame.jpg` or `basket3.png`) are located inside the `test/` directory.
2. Execute the python script:

```bash
python test.py
```

### Expected Output
- **`output.json`**: Contains the input prompt, file path, and a list of detections including bounding box coordinates (in pixel format) and confidence scores.
- **`output_visual.jpg`**: A visual representation of the original image with bounding boxes drawn over detected objects in red.