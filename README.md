# PriSM: Parsing and Style-Mixed Consistency for Unsupervised Domain Adaptation in Facial Landmark Detection (ECCV 2026)

### [Paper] 

**Chieh-Yu Yang¹**, **Hou-Ning Hu²**, **Sykai Chen²**, **Yu-Lun Liu¹**, and **Yen-Yu Lin¹**  
¹ *National Yang Ming Chiao Tung University (NYCU), Hsinchu, Taiwan*  
² *MediaTek, Hsinchu, Taiwan*  

---

## Teaser
![teaser](assets/teaser.png) 


---

## Abstract
Facial landmark detectors trained on real human faces often fail to generalize effectively to stylized domains such as caricatures and artistic portraits, necessitating unsupervised domain adaptation (UDA). Although self-training is a widely used UDA strategy to bridge domain gaps, it frequently breaks down under large domain shifts as it is prone to amplifying confident yet erroneous pseudo-label predictions. To this end, we propose **PriSM (Parsing and Style-Mixed Consistency)**, a novel method for robust landmark pseudo-label validation. PriSM leverages two complementary signals: a **Parsing Network** enforces high-level structural alignment through face parsing consistency, while a **StyleMix Network** enforces fine-grained geometric constraints by reducing the reconstruction error between the input face and another face synthesized from the pseudo-label’s structure and the input face’s appearance. Extensive experiments on the challenging CariFace and ArtiFace benchmarks under the UDA setting demonstrate that PriSM significantly outperforms existing state-of-the-art methods and exhibits strong generalizability to unseen domains.

---

## Overview
![framework](assets/framework.png)


---

## Directory Structure
To run our code, please organize your workspace as follows:

```text
PriSM-UDA/
├── configs/                      # Configuration files for the baseline landmark detector
│
├── data/                         # Dataloader scripts
│
├── Landmark2/                    # Source code for the baseline SLPT landmark alignment network
│
├── models/                       # PriSM model modules (PSP, Segmentation)
│
├── preprocess/                   # Isolated preprocessing scripts
│   ├── face_parsing/             (Face parsing extraction tool)
│   └── GAN_inversion/            (StyleGAN latent inversion tool)
│
├── Dataset/                      # Place your datasets here
│   ├── 300W/
│   ├── CariFace_dataset/
│   └── AF_dataset/
│
├── pretrained_models/            # Directory containing all pretrained model weights
│
└── train.py                      # Main training script
│
└── test.py                       # Evaluation and visualization script

```

## Pretrained Models
Please download all the necessary pretrained weights from our [Google Drive Link] and place them inside the `pretrained_models/` directory:

| Filename | Description | Original Source / Credit |
| :--- | :--- | :--- |
| `model_best.pth` | Baseline source landmark detector (SLPT) pretrained on 300W. Used as initial weights for our landmark alignment network. | [SLPT (Xia et al., CVPR 2022)](https://github.com/Jiahao-UTS/SLPT-master) |
| `landmark_segmentation_model2_50.pth` | Landmark segmentation network. Serves as our Point-to-Parse (P2P) module. | Ours (PriSM, This work) |
| `styleganex_mask2face.pt` | StyleGANEX (pSp) facial style-mixing model. Used as the StyleMix Network for target image reconstruction. | [StyleGANEX (Yang et al., ICCV 2023)](https://github.com/williamyang1991/StyleGANEX) |
| `styleganex_inversion.pt` | StyleGANEX inversion optimization model. Used in `preprocess/GAN_inversion` for image latent optimization. | [StyleGANEX (Yang et al., ICCV 2023)](https://github.com/williamyang1991/StyleGANEX) |
| `38_G.pth` | Face parsing generator (EHANet). Used in `preprocess/face_parsing` to extract reference parsing maps ($M_{\text{ref}}$). | Architecture: EHANet (Luo et al., 2020) / Weights: [TracelessLe](https://github.com/TracelessLe/FaceParsing.PyTorch) |
| `shape_predictor_68_face_landmarks.dat` | Dlib 68-point shape predictor. Used for face alignment during GAN inversion preprocessing. | [Dlib Official](http://dlib.net/) |
| `UDA_Cariface.pt` | Our final adapted landmark alignment network checkpoint on the CariFace benchmark. | Ours (PriSM, This work) |
| `UDA_Artiface.pt` | Our final adapted landmark alignment network checkpoint on the ArtiFace benchmark. | Ours (PriSM, This work) |





























