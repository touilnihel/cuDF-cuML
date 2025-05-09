# MNIST Classification: CPU vs GPU (cuDF & cuML)

This project demonstrates how to accelerate a machine learning pipeline using RAPIDS libraries—**cuDF** and **cuML**—on NVIDIA GPUs. The MNIST dataset is used to compare performance and accuracy between traditional CPU-based methods (Scikit-learn) and GPU-accelerated methods (cuML).

---

## 🚀 Technologies Used

- Python
- cuDF (GPU DataFrame API similar to Pandas)
- cuML (GPU ML API similar to Scikit-learn)
- Scikit-learn
- NumPy, pandas, Matplotlib
- RAPIDS.ai
- CUDA-enabled GPU (for GPU acceleration)

---

## 📦 Installation

Run the following in a Google Colab environment:

```bash
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/install_rapids.sh stable


Dataset

MNIST (Handwritten Digits)
Fetched using sklearn.datasets.fetch_openml
70,000 grayscale images (28x28), flattened to 784 features


