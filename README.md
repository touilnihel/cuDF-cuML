MNIST Classification: CPU vs GPU (cuDF & cuML)

This project demonstrates how to accelerate a machine learning pipeline using RAPIDS libraries—cuDF and cuML—on NVIDIA GPUs. The MNIST dataset is used to compare performance and accuracy between traditional CPU-based methods (Scikit-learn) and GPU-accelerated methods (cuML).

🚀 Technologies Used
- Python
- cuDF (GPU DataFrame API similar to Pandas)
- cuML (GPU ML API similar to Scikit-learn)
- Scikit-learn
- NumPy, pandas, Matplotlib
- RAPIDS.ai
- CUDA-enabled GPU (for GPU acceleration)

📦 Installation
Run the following in a Google Colab environment:
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/install_rapids.sh stable

Set CUDA path:
import os
os.environ["PATH"] += ":/usr/local/cuda/bin"

Then install required Python libraries:
import cuml
import cudf

📚 Dataset
- MNIST (Handwritten Digits)
- Fetched using sklearn.datasets.fetch_openml
- 70,000 grayscale images (28x28), flattened to 784 features

⚙️ Preprocessing
- Standardization using StandardScaler
- Train/test split (80/20)
- Conversion of NumPy arrays to cudf.DataFrame and cudf.Series for GPU training

🧠 Models
CPU (Scikit-learn)
- RandomForestClassifier from Scikit-learn
- Parameters: n_estimators=100, max_depth=10

GPU (cuML)
- RandomForestClassifier from cuML
- Parameters: n_estimators=100, max_depth=10, n_bins=8

📈 Results
| Method              | Accuracy | Training Time (sec) |
|---------------------|----------|----------------------|
| Scikit-learn (CPU)  | 0.9459   | 20.23                |
| cuML (GPU)          | 0.9447   | 4.58                 |

cuML achieves comparable accuracy with a 4x speedup in training time.

📌 Conclusion
- For small datasets like MNIST, both CPU and GPU methods perform well.
- cuML is significantly faster and ideal for scaling to large datasets or real-time systems.
- RAPIDS offers a seamless transition from Scikit-learn for GPU acceleration with minimal code changes.

📝 License
This project is released under the MIT License.

🙌 Acknowledgments
- RAPIDS.ai
- Scikit-learn
- OpenML MNIST
"""
