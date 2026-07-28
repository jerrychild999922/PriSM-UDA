import os
import glob
import argparse
from argparse import Namespace

import cv2
import torch
import dlib
from torchvision import transforms

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.psp import pSp
from latent_optimization import latent_optimization
from utils.inference_utils import save_image


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="StyleGANEX Batch Inversion")
        self.parser.add_argument(
            "--input_dir", type=str, default ="../../Dataset/AF_dataset/images",
            help="path to the directory containing target images"
        )
        self.parser.add_argument(
            "--ckpt", type=str, default="../../pretrained_models/styleganex_inversion.pt",
            help="path of the saved model checkpoint"
        )
        self.parser.add_argument(
            "--output_dir", type=str, default="../../Dataset/AF_dataset/output_batch",
            help="path to save inverted latents and images"
        )
        self.parser.add_argument(
            "--cpu", action="store_true",
            help="if true, run on CPU"
        )

    def parse(self):
        opt = self.parser.parse_args()
        print("Load options:")
        for k, v in sorted(vars(opt).items()):
            print(f"{k}: {v}")
        return opt


def main():
    args = TestOptions().parse()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cpu" if args.cpu else "cuda"

    # Prepare transform (if needed later)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Load StyleGANEX inversion model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.ckpt
    opts['device'] = device
    opts = Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)

    # Load dlib face landmark predictor
    predictor_path = '../../pretrained_models/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        import wget, bz2
        bz2_path = predictor_path + '.bz2'
        wget.download(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', bz2_path
        )
        with bz2.BZ2File(bz2_path) as f_in, open(predictor_path, 'wb') as f_out:
            f_out.write(f_in.read())
    landmark_predictor = dlib.shape_predictor(predictor_path)

    # Gather all image files in input_dir
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(img_paths)} images. Starting batch inversion...")

    # Optionally load pre-warped latents if needed
    # e4e_latent = torch.load('from_warping.pt', map_location='cpu')['latent']
    # e4e_latent = e4e_latent.unsqueeze(0).to(device)

    for img_path in img_paths:
        print(f"Processing {img_path}...")
        # Read and preprocess image
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H = W = opts.input_size if hasattr(opts, 'input_size') else 256
        frame = cv2.resize(frame, (W, H))

        # Inversion optimization
        wplus_hat, f_hat, noises_hat, _, _ = latent_optimization(
            frame, pspex, landmark_predictor,
            step=500, device=device
        )

        # Decode and save image
        with torch.no_grad():
            y_hat, _ = pspex.decoder(
                [wplus_hat], input_is_latent=True,
                randomize_noise=False,
                first_layer_feature=f_hat,
                noise=noises_hat
            )
            y_hat = torch.clamp(y_hat, -1, 1)

        base = os.path.splitext(os.path.basename(img_path))[0]
        # Save latent pt
        save_dict = {
            'wplus': wplus_hat.detach().cpu(),
            'f': [f.detach().cpu() for f in f_hat],
        }
        pt_path = os.path.join(args.output_dir, f"{base}.pt")
        torch.save(save_dict, pt_path)

        # Save inverted image
        img_out_path = os.path.join(args.output_dir, f"{base}_inversion.jpg")
        save_image(y_hat[0].cpu(), img_out_path)

        print(f"Saved PT to {pt_path} and image to {img_out_path}")

    print("Batch inversion complete!")


if __name__ == '__main__':
    main()