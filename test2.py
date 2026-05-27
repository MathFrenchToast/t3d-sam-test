import os
import argparse
import torch
from functools import partial
from dotenv import load_dotenv

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SAM3 Agent Inference")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--prompt", default="the leftmost child wearing blue vest", help="Natural language prompt")
    parser.add_argument("--model", default="qwen3_vl_8b_thinking", help="LLM model name")
    parser.add_argument("--url", default=None, help="LLM server URL (default: http://0.0.0.0:8001/v1 or from .env)")
    parser.add_argument("--api_key", default=None, help="LLM API Key (default: DUMMY_API_KEY or from .env)")
    parser.add_argument("--output_dir", default="agent_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve URL and API Key
    llm_url = args.url or os.getenv("LLM_SERVER_URL", "http://0.0.0.0:8001/v1")
    llm_api_key = args.api_key or os.getenv("LLM_API_KEY", "DUMMY_API_KEY")
    
    # 1. Setup GPU and Autocast
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Inference might be very slow.")
    
    # 2. Build SAM3 Model
    print("Loading SAM3 model...")
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    if not os.path.exists(bpe_path):
        print(f"BPE path not found at {bpe_path}, trying without it...")
        model = build_sam3_image_model()
    else:
        model = build_sam3_image_model(bpe_path=bpe_path)
        
    processor = Sam3Processor(model, confidence_threshold=0.5)
    
    # 3. LLM Configuration
    llm_config = {
        "name": args.model,
        "model": args.model,
        "api_key": llm_api_key,
        "provider": "vllm" if "0.0.0.0" in llm_url or "localhost" in llm_url else "openai"
    }
    
    # 4. Prepare Partial Functions
    send_generate_request = partial(
        send_generate_request_orig, 
        server_url=llm_url, 
        model=llm_config["model"], 
        api_key=llm_config["api_key"]
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    
    # 5. Run Inference
    print(f"Running inference on {args.image} with prompt: '{args.prompt}'")
    print(f"Using LLM at {llm_url} (Model: {args.model})")
    
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Use autocast and inference mode
    autocast_ctx = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()
    
    with autocast_ctx, torch.inference_mode():
        output_image_path = run_single_image_inference(
            image_path, 
            args.prompt, 
            llm_config, 
            send_generate_request, 
            call_sam_service,
            debug=True, 
            output_dir=args.output_dir
        )
        
        if output_image_path:
            print(f"Success! Output saved to: {output_image_path}")
        else:
            print("Inference failed to produce an output image.")

if __name__ == "__main__":
    main()
