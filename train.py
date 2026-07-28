import argparse
import os
import logging
from datetime import datetime
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import models
import numpy as np

# Custom project imports
from Landmark2.model import Sparse_alignment_network
from Landmark2.Config import cfg
from Landmark2.test import main_function_test, save_point
from Landmark2.backbone import Alignment_Loss
import models.stylegan2.lpips as lpips

# Dataloaders (300W)
from data.W300 import get_W300_dataloader
# Dataloaders (Cariface)
#from data.Cartoon import get_cartoon_dataloader
#from data.Combine_new import get_combine_dataloader
# Dataloaders (Artiface)
from data.Artistic import get_cartoon_dataloader
from data.Combine_new_artistic import get_combine_dataloader 

# Generative & Segmentation Models
from models.psp import pSp
from models.segmentation import LandmarkSegmentationNet

# Initialize logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_file = datetime.now().strftime("log_%Y%m%d_%H%M%S.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully!")


def build_model(opt):
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Initialize our main alignment network
    alignment_net = Sparse_alignment_network(
        cfg.W300.NUM_POINT, cfg.MODEL.OUT_DIM,
        cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
        cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
        cfg.TRANSFORMER.FEED_DIM, cfg.W300.INITIAL_PATH, cfg
    ).to(device)

    # Load pre-trained models
    logger.info(f"Loading checkpoint from: {opt.checkpoint}")
    if opt.checkpoint is None:
        checkpoint_file = opt.pretrain_path
        checkpoint = torch.load(checkpoint_file, map_location=device)
        alignment_net.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(opt.checkpoint, map_location=device)
        alignment_net.load_state_dict(checkpoint)

    for params in alignment_net.parameters():
        params.requires_grad = True
        
    align_optim = torch.optim.Adam(alignment_net.parameters(), lr=opt.lr_w)
    landmark_optim = torch.optim.Adam(alignment_net.parameters(), lr=opt.lr_l)
    return alignment_net, align_optim, landmark_optim


def train(alignment_net, align_optim, landmark_optim, opt):
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    best_score = float('inf')
    best_model_path = os.path.join(opt.snapshot_dir, "best_alignment_net.pth")
    save_path = os.path.join(opt.save_data_dir, 'pseudo_labels_init/')
    
    trainA_loader, testA_loader = get_W300_dataloader(opt.batch_size, opt)
    trainB_loader, testB_loader = get_cartoon_dataloader(opt.batch_size, opt)
    trainB_generate_loader, testB_generate_loader = get_cartoon_dataloader(1, opt)
    
    # Initialize segmentation model
    segmentation_model = LandmarkSegmentationNet(latent_dim=512, output_size=256).to(device)
    checkpoint_seg = torch.load(opt.seg_net_ckpt, map_location=device) 
    segmentation_model.load_state_dict(checkpoint_seg)
    segmentation_model.eval()
    for param in segmentation_model.parameters():
        param.requires_grad = False
        
    # Initialize StyleGANEX (pSp)
    ckpt = torch.load(opt.stylegan_ckpt, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = opt.stylegan_ckpt
    opts['device'] = device
    opts = Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)
    for name, param in pspex.decoder.named_parameters():
        param.requires_grad = False
        
    landmark_loss = Alignment_Loss(cfg)
    criterion = nn.CrossEntropyLoss().to(device)
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg").to(device)
    
    # Build Label Mapping Matrix (10 classes -> 19 classes)
    mapping_matrix = torch.zeros(10, 19).to(device)
    for i in range(10):
        mapping_matrix[i, i] = 1.0

    special_map = {6: 10, 3: 2, 2: 3, 5: 4, 4: 5, 8: 11, 7: 12, 9: 13}
    for old_idx, new_idx in special_map.items():
        if old_idx < 10:
            mapping_matrix[old_idx, old_idx] = 0.0
            mapping_matrix[old_idx, new_idx] = 1.0
            
    for step in range(opt.epoch):
        alignment_net.train()
        if (step // opt.update_frequency) % 2 == 0:
            logger.info('--------- Start UDA Phase ----------')
            for batch, meta_B in enumerate(trainB_loader):
                target_img = meta_B['image'].to(device)
                target_parsing = meta_B['parsing'].to(device)
                latent_cpu = meta_B['latent']
                
                if latent_cpu.dim() == 4 and latent_cpu.size(1) == 1:
                    latent = latent_cpu.squeeze(1).to(device)
                elif latent_cpu.dim() == 3:
                    latent = latent_cpu.to(device)
                else:
                    latent = latent_cpu.to(device)
                    if latent.dim() == 1:
                        latent = latent.unsqueeze(0)
                    if latent.dim() == 2:
                        n_ws = pspex.opts.n_styles if hasattr(pspex.opts, "n_styles") else 18
                        latent = latent.unsqueeze(1).repeat(1, n_ws, 1)
                        
                align_optim.zero_grad()
                pred_landmarks = alignment_net(target_img)[2][:, -1, :, :]
                
                seg_logits = segmentation_model(pred_landmarks)  # [B, 10, 256, 256]
                seg_probs = F.softmax(seg_logits, dim=1)         # [B, 10, 256, 256]

                # Perform matrix multiplication to convert channels (10 -> 19)
                seg_probs_permuted = seg_probs.permute(0, 2, 3, 1)  # [B, H, W, 10]
                one_hot_soft = torch.matmul(seg_probs_permuted, mapping_matrix)  # [B, H, W, 19]
                one_hot_soft = one_hot_soft.permute(0, 3, 1, 2)  # [B, 19, H, W]
                
                # StyleGAN Reconstruction
                y_hat = pspex(
                    x1=one_hot_soft,
                    x2=None,
                    resize=True,
                    latent_mask=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                    use_skip=pspex.opts.use_skip,
                    inject_latent=latent
                )
                generated = torch.clamp(y_hat, -1, 1)
                
                W_loss = 0.25*criterion(seg_logits, target_parsing) + percept(generated, target_img).mean()
                logger.info("[%5d/%5d/%5d] W_loss: %.8f" % (batch, step, opt.epoch, W_loss))
                W_loss.backward()
                align_optim.step()

            if (step + 1) % opt.update_frequency == 0:
                save_path = os.path.join(opt.save_data_dir, 'pseudo_labels_epoch_' + str(step) + '/')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_point(alignment_net, trainB_generate_loader, save_path, logger)
                current_test_score, _ = save_point(alignment_net, testB_generate_loader, save_path, logger)
                logger.info(f"[Step {step}] test score = {current_test_score:.6f}, best_score = {best_score:.6f}")
                if current_test_score < best_score:
                    best_score = current_test_score
                    torch.save(alignment_net.state_dict(), best_model_path)
                    logger.info(f"  >>> New best model at step {step}! best_score = {best_score:.6f}")

        else:
            logger.info('--------- Start Fine-Tuning Phase ----------')
            # Use mixed dataloader (source with GT & target with pseudo labels) to update landmarker
            train_combine_loader, test_combine_loader = get_combine_dataloader(opt, save_path)
            for batch, meta in enumerate(train_combine_loader):
                combined_img = meta['image'].to(device)
                combined_landmarks = meta['points'].to(device)
                landmark_optim.zero_grad()
                pred = alignment_net(combined_img)
                loss = landmark_loss(pred[0], combined_landmarks) * 0.2 + landmark_loss(pred[1], combined_landmarks) * 0.3 + landmark_loss(pred[2], combined_landmarks) * 0.5
                loss.backward() 
                landmark_optim.step()
                logger.info("[%5d/%5d/%5d] L_loss: %.8f" % (batch, step, opt.epoch, loss))

        if (step + 1) % (opt.update_frequency * 2) == 0:
            logger.info('--------- Saving periodic checkpoint ----------')
            checkpoint_path = os.path.join(opt.snapshot_dir, 'params_%07d.pt' % step)
            params = {'alignment_net': alignment_net.state_dict()}
            torch.save(params, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        params = {'alignment_net': alignment_net.state_dict()}
        torch.save(params, os.path.join(opt.snapshot_dir, 'final_state.pt'))


def check_args(args):
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    os.makedirs(args.save_data_dir, exist_ok=True)

    if args.epoch < 1:
        raise ValueError('Number of epochs must be larger than or equal to one')
    if args.batch_size < 1:
        raise ValueError('Batch size must be larger than or equal to one')
    return args





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="how many samples for one batch",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="total number of epoch",
    )
    parser.add_argument(
        "--update_frequency",
        type=int,
        default=2,
        help="update warpA2B as warper and landmarker in turn for every n epochs",
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default='pretrained_models/model_best.pth',
        help="the path to the source pretrained model weights",
    )
    parser.add_argument(
        "--src_data",
        type=str,
        default='Dataset/300W',
        help="the path to the source dataset",
    )
    parser.add_argument(
        "--tgt_data",
        type=str,
        default='Dataset/AF_dataset',
        help="the path to the target dataset",
    )
    parser.add_argument(
        "--save_data_dir",
        type=str,
        default='pseudo_data',
        help="the path to save the pseudo labels",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='results',
        help="the path to save the validation results",
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default='snapshots',
        help="the path to save the checkpoint file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help='checkpoint file',
    )
    parser.add_argument(
        "--stylegan_ckpt",
        type=str,
        default="pretrained_models/styleganex_mask2face.pt",
        help="path to the StyleGANEX (pSp) checkpoint",
    )
    parser.add_argument(
        "--seg_net_ckpt",
        type=str,
        default="pretrained_models/landmark_segmentation_model2_50.pth",
        help="path to the LandmarkSegmentationNet checkpoint",
    )
    parser.add_argument(
        "--lr_w",
        type=float,
        default=1e-10,
        help="learning rate for warper (W_optim)"
    )
    parser.add_argument(
        "--lr_l",
        type=float,
        default=1e-5,
        help="learning rate for landmarker (L_optim)"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="lpips",
    )


    opt = parser.parse_args()
    check_args(opt)
    logger.info(f"Running with arguments: {vars(opt)}")
    
    warpA2B, W_optim, L_optim = build_model(opt)
    train(warpA2B, W_optim, L_optim, opt)
    print(" [*] Training finished!")