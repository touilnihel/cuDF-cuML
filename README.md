# MNIST Classification with RAPIDS (cuDF & cuML)

This project demonstrates how to leverage NVIDIA RAPIDS libraries — **cuDF** and **cuML** — to accelerate machine learning workflows using GPUs. We compare the performance of Random Forest classification on the MNIST dataset using traditional **CPU-based Scikit-learn** and **GPU-accelerated cuML**.

---

## 🔧 RAPIDS Overview

**RAPIDS** is an open-source suite of GPU-accelerated libraries designed to speed up data science workflows.

### 🧮 cuDF (GPU DataFrames)

- cuDF is the GPU-accelerated equivalent of `pandas`.
- It provides fast data manipulation using CUDA and NVIDIA GPUs.
- Syntax is nearly identical to `pandas`.

### 🧠 cuML (GPU Machine Learning)

- cuML is a library of GPU-accelerated machine learning algorithms.
- Implements common models like Logistic Regression, Random Forests, KMeans, PCA, etc.
- Scikit-learn compatible API for easy migration.

---

## 📦 Installation

To use RAPIDS in a Colab environment:

```bash
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/install_rapids.sh stable```


## 📊 Results Comparison

| Method              | Accuracy | Training Time (sec) |
|---------------------|----------|----------------------|
| Scikit-learn (CPU)  | 0.9459   | 20.23                |
| cuML (GPU)          | 0.9447   | 4.58                 |

---

## 📌 Conclusion

- **cuML on GPU** is significantly faster while maintaining nearly identical accuracy.
- For larger datasets or more computationally intensive tasks, **GPU acceleration offers substantial time savings**.
- If **speed** is a priority, **cuML** is the preferred choice. For small datasets, **CPU-based solutions may suffice**.
