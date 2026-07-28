# test.py
import argparse
import os
import pprint
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.integrate import simps

# Configuration and Model Imports
from Landmark2.Config import cfg, update_config
from Landmark2.model import Sparse_alignment_network

# Dataloaders (Renamed to prevent name collisions)
from data.W300 import get_W300_dataloader
from data.Cartoon import get_cartoon_dataloader as get_cariface_dataloader
from data.Artistic import get_cartoon_dataloader as get_artiface_dataloader


def tensor2numpy(x):
    """Convert PyTorch tensor [C, H, W] to NumPy array [H, W, C]."""
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    """Convert RGB image to BGR for OpenCV."""
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def create_logger(cfg):
    """Initialize a simple logger for evaluation output."""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console = logging.StreamHandler()
        logger.addHandler(console)
    return logger


# =====================================================================
# Arguments Parser (Defaulting to evaluate all and save visuals)
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PriSM Landmark Detector')

    # Model & Checkpoint Settings
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation (must be 1)')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default="pretrained_models/UDA_Artiface.pt", 
        help='path to the trained alignment network checkpoint'
    )

    # Evaluation Dataset Toggles (Default: True. Use these flags to DISABLE evaluation)
    parser.add_argument('--no_300w', action='store_false', dest='eval_300w', help='disable 300W evaluation')
    parser.add_argument('--no_cariface', action='store_false', dest='eval_cariface', help='disable CariFace evaluation')
    parser.add_argument('--no_artiface', action='store_false', dest='eval_artiface', help='disable ArtiFace evaluation')

    # Visualization Toggle (Default: True. Use this flag to DISABLE visualization saving)
    parser.add_argument('--no_visuals', action='store_false', dest='save_visuals', help='disable visualization saving')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='directory to save visual results')

    # Dataset Directories (Relative paths for GitHub release)
    parser.add_argument("--src_data", type=str, default='Dataset/300W', help="path to the 300W dataset")
    parser.add_argument("--tgt_data_cari", type=str, default='Dataset/CariFace_dataset', help="path to CariFace")
    parser.add_argument("--tgt_data_arti", type=str, default='Dataset/AF_dataset', help="path to ArtiFace")

    # Unused arguments from baseline (kept for compatibility with baseline Config.py)
    parser.add_argument('--modelDir', type=str, default='./Weight')
    parser.add_argument('--logDir', type=str, default='./log')
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--prevModelDir', type=str, default=None)

    args = parser.parse_args()
    return args


# =====================================================================
# Metrics & Evaluation Loop
# =====================================================================

def calculate_loss(name, pred, gt, trans):
    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name in ['300W', 'Cartoon', 'Artistic']:
        norm = np.linalg.norm(gt[36, :] - gt[45, :])  # Inter-ocular distance normalization
    elif name == 'WFLW':
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name == 'COFW':
        norm = np.linalg.norm(gt[17, :] - gt[16, :])
    else:
        raise ValueError(f'Unsupported dataset normalization: {name}')

    error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)
    return error_real


class FR_AUC:
    """Failure Rate and AUC evaluation metric class."""
    def __init__(self, thresh=0.08):
        self.thresh = thresh

    def test(self, nmes, thres=None, step=0.0001):
        if thres is None:
            thres = self.thresh

        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
        fr = 1.0 - ys[-1]
        auc = simps(ys, x=xs) / thres
        return [round(fr, 4), round(auc, 6)]


def build_model_and_load(device, ckpt_path):
    """Safely build and load the alignment network state dict."""
    model = Sparse_alignment_network(
        cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM, cfg.MODEL.TRAINABLE,
        cfg.MODEL.INTER_LAYER, cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
        cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg
    ).to(device)
    
    print(f" [*] Loading model checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
    if 'alignment_net' in state:
        model.load_state_dict(state['alignment_net'])
    elif 'warpA2B' in state:
        model.load_state_dict(state['warpA2B'])
    else:
        model.load_state_dict(state)
        
    model.eval()
    return model


def evaluate_dataset(model, dataloader, dataset_name, save_visuals=False, vis_dir=None, device="cuda"):
    """Generic evaluation function for high code reuse and modularity."""
    errors = []
    
    save_path = os.path.join(vis_dir, dataset_name) if save_visuals else None
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    print(f"\n=================[ Testing on {dataset_name} ]=================")
    with torch.no_grad():
        for i, meta in enumerate(dataloader):
            input_tensor = meta['image']
            annotated_points = meta['Annotated_Points'].numpy()[0]
            trans = meta['trans'].numpy()[0]

            # Forward pass
            outputs = model(input_tensor.to(device))
            pred_pts = outputs[2][0, -1, :, :].cpu().numpy()

            # Calculate NME
            error = calculate_loss('300W', pred_pts * cfg.MODEL.IMG_SIZE, annotated_points, trans)
            errors.append(error)

            if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
                print(f"Processed [{i + 1}/{len(dataloader)}] images. Current NME: {np.mean(errors)*100.0:.3f}%")

            # Save visualization if enabled
            if save_visuals and save_path:
                img_name = os.path.basename(meta['Img_path'][0]) if 'Img_path' in meta else f"{i:06d}.png"
                
                # Convert PyTorch Tensor to contiguous OpenCV BGR image using embedded helpers
                img_bgr = RGB2BGR(tensor2numpy(input_tensor[0] * 0.5 + 0.5)) * 255.0
                img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
                
                # Draw predicted landmarks (Red circles)
                for pt in pred_pts * 255.0:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(img_bgr, (x, y), 1, (0, 0, 255), 2)
                
                cv2.imwrite(os.path.join(save_path, img_name), img_bgr)

    mean_nme = np.mean(errors) * 100.0
    print(f" [*] Finished {dataset_name}. Mean NME: {mean_nme:.3f}%")
    return errors, mean_nme


# =====================================================================
# Main Execution
# =====================================================================

def main():
    args = parse_args()
    update_config(cfg, args)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Initialize Embedded Logger
    logger = create_logger(cfg)
    logger.info(pprint.pformat(args))

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Load alignment model
    alignment_net = build_model_and_load(device, args.checkpoint)

    # Read evaluation options
    run_300w = args.eval_300w
    run_cariface = args.eval_cariface
    run_artiface = args.eval_artiface

    metric_evaluator = FR_AUC(thresh=0.08)

    # 1. Evaluate on 300W
    if run_300w:
        _, valid_loader_300w = get_W300_dataloader(args.batch_size, args)
        errors, mean_nme = evaluate_dataset(
            alignment_net, valid_loader_300w, "300W", 
            save_visuals=args.save_visuals, vis_dir=args.vis_dir, device=device
        )
        fr, auc = metric_evaluator.test(errors)
        print(f"\n>> [300W Results] Mean NME: {mean_nme:.4f}% | Failure Rate @0.08: {fr*100.0:.2f}% | AUC @0.08: {auc:.4f}")

    # 2. Evaluate on CariFace
    if run_cariface:
        args.tgt_data = args.tgt_data_cari
        _, valid_loader_cari = get_cariface_dataloader(args.batch_size, args)
        errors, mean_nme = evaluate_dataset(
            alignment_net, valid_loader_cari, "CariFace", 
            save_visuals=args.save_visuals, vis_dir=args.vis_dir, device=device
        )
        fr, auc = metric_evaluator.test(errors)
        print(f"\n>> [CariFace Results] Mean NME: {mean_nme:.4f}% | Failure Rate @0.08: {fr*100.0:.2f}% | AUC @0.08: {auc:.4f}")

    # 3. Evaluate on ArtiFace
    if run_artiface:
        args.tgt_data = args.tgt_data_arti
        _, valid_loader_arti = get_artiface_dataloader(args.batch_size, args)
        errors, mean_nme = evaluate_dataset(
            alignment_net, valid_loader_arti, "ArtiFace", 
            save_visuals=args.save_visuals, vis_dir=args.vis_dir, device=device
        )
        fr, auc = metric_evaluator.test(errors)
        print(f"\n>> [ArtiFace Results] Mean NME: {mean_nme:.4f}% | Failure Rate @0.08: {fr*100.0:.2f}% | AUC @0.08: {auc:.4f}")

    print("\n [*] All evaluations completed!")


if __name__ == '__main__':
    main()