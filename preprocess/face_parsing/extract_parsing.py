# preprocess/extract_parsing.py
import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model definition from the local networks folder
from networks import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Face Parsing Maps (M_ref)")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="../../Dataset/AF_dataset/images", 
        help="Path to input target-domain images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../../Dataset/AF_dataset/parsing1", 
        help="Path to save the generated parsing maps"
    )
    parser.add_argument(
        "--arch", 
        type=str, 
        default="FaceParseNet50", 
        help="Model architecture name"
    )
    parser.add_argument(
        "--weight_path", 
        type=str, 
        default="../../pretrained_models/38_G.pth", 
        help="Path to pre-trained parser generator weight"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID to use"
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize and load the generator model
    logger_msg = f"Initializing {args.arch} on device {device}..."
    print(logger_msg)
    
    # get_model comes from preprocess/networks/
    G = get_model(args.arch, pretrained=False).to(device)
    G.load_state_dict(torch.load(args.weight_path, map_location=device))
    G.eval()

    # Define label filtering and mapping rules from your thesis
    keep_values = {1, 2, 4, 5, 6, 7, 10, 11, 12}
    mapping = {2: 6, 6: 3, 7: 2, 4: 5, 5: 4, 10: 8, 11: 7, 12: 9}

    img_names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(img_names)} images in {args.input_dir}. Starting inference...")

    start_time = time.time()
    for i, filename in enumerate(img_names):
        input_path = os.path.join(args.input_dir, filename)
        
        # Load and preprocess image
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            continue
            
        if len(image_bgr.shape) < 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[0], image_rgb.shape[1]
        
        # Convert to tensor: CxHxW normalized to [0, 1]
        image_tensor = image_rgb.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_tensor).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Model Inference
        outputs = G(image_tensor)
        if args.arch == 'CE2P' or 'FaceParseNet' in args.arch:
            outputs = outputs[0][-1]

        # Upsample outputs back to original image size
        outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
        pred_map = torch.argmax(outputs, dim=1).cpu().numpy()[0].astype(np.uint8)

        # Apply label filtering (keep only the 10 core classes)
        filtered_map = np.where(np.isin(pred_map, list(keep_values)), pred_map, 0)

        # Apply label mapping
        remapped_map = filtered_map.copy()
        for old_val, new_val in mapping.items():
            remapped_map[filtered_map == old_val] = new_val

        # Save result as index-grayscale PNG
        output_file = f"{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(args.output_dir, output_file)
        cv2.imwrite(output_path, remapped_map)

        if (i + 1) % 50 == 0 or (i + 1) == len(img_names):
            print(f"Processed [{i + 1}/{len(img_names)}] images.")

    total_time = time.time() - start_time
    print(f"Finished! Average Inference Time: {total_time / len(img_names):.4f}s per image.")


if __name__ == "__main__":
    main()