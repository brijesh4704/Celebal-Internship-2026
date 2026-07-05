# MNIST Image Denoising using Autoencoders

## Project Overview

This project implements a **Deep Learning-based Image Denoising Autoencoder** using the **MNIST handwritten digit dataset**. The model is trained to remove **Gaussian noise** from images and reconstruct the original clean images.

Three different Autoencoder architectures are implemented and compared:

- Fully Connected Autoencoder (FFNN)
- Transposed Convolutional Autoencoder
- Upsampling Convolutional Autoencoder

---

## Objective

The objective of this project is to build an Autoencoder that can learn the mapping from **noisy handwritten digit images** to their **clean versions**.

```
Original Image
       │
Add Gaussian Noise
       │
       ▼
Noisy Image
       │
       ▼
Autoencoder
       │
       ▼
Reconstructed Clean Image
```

---

## Dataset

The project uses the **MNIST** handwritten digit dataset.

- Training Images: **60,000**
- Testing Images: **10,000**
- Image Size: **28 × 28**
- Grayscale Images

The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## Models Implemented

### 1. Fully Connected Autoencoder (FFNN)

- Linear Encoder
- Linear Decoder
- Learns a compressed representation of the input image.

---

### 2. Transposed Convolutional Autoencoder

- Convolutional Encoder
- Transposed Convolution Decoder
- Preserves spatial information for improved reconstruction.

---

### 3. Upsampling Convolutional Autoencoder

- Convolutional Encoder
- Upsampling + Convolution Decoder
- Reduces checkerboard artifacts commonly produced by transpose convolutions.

---

## Adding Noise

Gaussian noise is added to the input images during training.

```python
def add_noise(images, noise_factor=0.6):
    noisy = images + noise_factor * torch.randn_like(images)
    noisy = torch.clamp(noisy, 0., 1.)
    return noisy
```

Training Input:

```
Noisy Image
```

Training Target:

```
Clean Image
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 20 |
| Dataset | MNIST |

---

## Loss Function

Mean Squared Error (MSE) is used because image denoising is a reconstruction problem where the predicted pixel values should closely match the original image.

```python
loss = criterion(output, clean_images)
```

---

## Results

The trained Autoencoder successfully removes Gaussian noise from MNIST images.

The notebook displays:

- Original Image
- Noisy Image
- Denoised Image

Example workflow:

```
Original Image
      ↓
Noisy Image
      ↓
Denoised Image
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Jupyter Notebook

---

## Project Structure

```
MNIST-Denoising-Autoencoder/
│
├── autoencoder_mnist.ipynb
├── README.md
├── requirements.txt
├── denoising_ffnn.pth
├── denoising_transposecnn.pth
├── denoising_upsamplecnn.pth
├── data/
└── assets/
```

---

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/your-username/MNIST-Denoising-Autoencoder.git
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Open the notebook.

```bash
jupyter notebook autoencoder_mnist.ipynb
```

4. Run all cells to train the models and visualize the denoising results.

---

## Future Improvements

- Train on CIFAR-10 and Fashion-MNIST datasets.
- Improve denoising quality using U-Net or Variational Autoencoders (VAEs).
- Evaluate performance using PSNR and SSIM metrics.
- Deploy the model as a web application using Streamlit.

---

## Author

**Brijesh Singh**

B.Tech Computer Science & Data Science

Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur

---

## License

This project is developed for educational and learning purposes.
