
# 🧠 Deep Dive into Image Compression  
### Convolutional Autoencoders on MNIST & CelebA  
*Turning pixels into purpose*

---

## 🔍 Overview

This project explores the magic of **convolutional autoencoders** (CAEs) to compress and reconstruct image data. By training on both the classic **MNIST** dataset and the more complex **CelebA** dataset, we learn how neural networks can effectively capture essential features of images — all while reducing dimensionality.

We built, trained, and tested autoencoders using **PyTorch**, with GPU acceleration to dramatically reduce training time and boost performance.

---

## 📊 Datasets

| Dataset | Description | Size | Type |
|--------|-------------|------|------|
| **MNIST** | Handwritten digits (0–9) | 28×28 | Grayscale |
| **CelebA** | Celebrity faces | 64×64 | RGB |

---

## 🧱 Model Architecture

A symmetric convolutional autoencoder designed to encode input images into a lower-dimensional **latent space**, and then decode them back to their original form.

```
Input Image
   ↓
[Conv2D → ReLU → MaxPool] × N
   ↓
Latent Space
   ↓
[ConvTranspose2D → ReLU] × N
   ↓
Reconstructed Image
```

🧠 The bottleneck (latent representation) helps the model learn meaningful compressed features from each image.

---

## ⚙️ Training Configuration

- **Framework**: PyTorch  
- **Hardware**: GPU-enabled (NVIDIA Tesla T4 / CUDA)  
- **Optimizer**: Adam  
- **Loss Function**: MSELoss  
- **Batch Size**: 128  
- **Epochs**: 20  

### 🚀 GPU Acceleration

Using GPU reduced training time drastically:

| Dataset | Training Time | Final Loss |
|--------|---------------|------------|
| MNIST  | ~4 minutes    | ~0.030     |
| CelebA | ~4 minutes    | ~0.050     |

> _Total Time: ~8 minutes on GPU_

---

## 📈 Results

### MNIST Reconstruction

- Crisp reconstruction of digits after compression.
- Latent space learned to generalize well across digit types.

| Original | Reconstructed |
|----------|----------------|
| ![MNIST Original](assets/mnist_orig.png) | ![MNIST Recon](assets/mnist_recon.png) |

### CelebA Reconstruction

- Retained facial structure and features well.
- Captured details like face shape, hairline, and tone.

| Original | Reconstructed |
|----------|----------------|
| ![CelebA Original](assets/celeb_orig.png) | ![CelebA Recon](assets/celeb_recon.png) |

> “Compression is not loss — it’s learning what matters.” 💡

---

## 📦 Installation & Usage

Clone the repository and run the notebook:

```bash
git clone https://github.com/your-username/autoencoder-mnist-celeba.git
cd autoencoder-mnist-celeba
pip install -r requirements.txt
jupyter notebook yousif.ipynb
```

Make sure you're using a GPU-enabled environment (e.g., Google Colab or CUDA-supported local setup).

---

## 🔮 Future Ideas

- Add **Variational Autoencoders (VAEs)** for probabilistic generation  
- Apply **t-SNE** on latent space for visualization  
- Explore **denoising autoencoders** for image cleaning tasks  

---

## 🧑‍💻 Author

**Yousif**  
_Student of pixels and neural dreams._

📬 Reach out: [Your Email or GitHub](https://github.com/your-username)  
🌟 Star the repo if this project resonates with your interests!

---

## 🖼️ Assets & Notes

Store visual outputs in a folder named `assets/`:

- `mnist_orig.png` – Sample original MNIST image
- `mnist_recon.png` – Reconstructed MNIST image
- `celeb_orig.png` – Sample original CelebA face
- `celeb_recon.png` – Reconstructed CelebA face

You can generate and save them using:

```python
plt.imsave("assets/mnist_orig.png", original_image, cmap="gray")
plt.imsave("assets/mnist_recon.png", reconstructed_image, cmap="gray")
plt.imsave("assets/celeb_orig.png", celeb_original)
plt.imsave("assets/celeb_recon.png", celeb_reconstructed)
```

---

📘 *Learning to see — one pixel at a time.*
