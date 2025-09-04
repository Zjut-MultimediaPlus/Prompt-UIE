# Prompt-UIE: A Unified Prompt-Driven Framework for Underwater Image Enhancement

This repository contains the official PyTorch implementation of our ICASSP 2025 paper:

---

## 📂 Project Structure

```
├── train.py                # Training script
├── test.py                 # Testing script
├── models/                 # Network architectures
│   └── ...                 # (e.g., myModel, Prompt modules, etc.)
├── utility/                # Utilities
│   ├── train_dataloader.py # Training dataset loader
│   ├── val_dataloader.py   # Validation dataset loader
│   └── metrics_calculation.py
├── data/                   # Dataset directory
│   ├── train/              # Training set
│   │   ├── input/          # Input underwater images
│   │   ├── gt/             # Ground-truth images
│   │   └── input.txt       # Training file list
│   └── test/               # Testing set
│       ├── input/          
│       ├── gt/             
│       └── input.txt
└── results/                # Results will be saved here
```

---

## 📦 Dataset Preparation

We use the **LSUI dataset**, proposed in the paper  
> *U-shape Transformer for Underwater Image Enhancement*  
Available at [GitHub Repository](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement?tab=readme-ov-file).

### Folder Structure

Download LSUI dataset and organize it as follows:

```
data/
├── train/
│   ├── input/    # Training input images
│   ├── gt/       # Ground truth images
│   └── input.txt # File list (filename)
└── test/
    ├── input/    # Testing input images
    ├── gt/       # Ground truth images
    └── input.txt
```

- **`input.txt` format**:  
  Each line contains the image filename, e.g.:
  ```
  0001.jpg 
  0002.jpg
  ...
  ```

---

## 🔄 Data Preprocessing

During training, images are randomly cropped to the desired size.  
From [`train_dataloader.py`](utility/train_dataloader.py):

- Images are resized if smaller than crop size.
- A random crop of size `(crop_width, crop_height)` is taken.
- Both input and ground truth are normalized to `[-1, 1]`.

---

## 🚀 Training

Run the following command to train:

```bash
python train.py   -train_data_dir ./data/train/   -val_data_dir ./data/test/   -labeled_name input.txt   -val_filename1 input.txt   -exp_name weight   -weight_out uie_icassp2025
```

---

## 🧪 Testing

Run the following command to test:

```bash
python test.py   -val_data_dir ./data/test/   -val_filename1 input.txt   -weight_path ./weight/uie_icassp2025/best_model.pth
```

The enhanced results will be saved in the `./results/` directory.

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2025prompt,
  title={Prompt-UIE: A Unified Prompt-Driven Framework for Underwater Image Enhancement},
  author={Zhang, Yanling and Luo, Linxuan and Mu, Pan and Bai, Cong},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

---

## 🙏 Acknowledgements

- LSUI dataset from [U-shape Transformer for Underwater Image Enhancement].
- This project is built with PyTorch.
