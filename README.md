# PP5/6 Deep Learning on a Non-Linearly Separable Moons Dataset

## Project Overview

This project explores deep learning for binary classification using a **non-linearly separable dataset**. The dataset was designed to simulate a challenging decision boundary and was modeled using **two deep learning frameworks**: **PyTorch** and **fastai**.

The main objective was to train a neural network to distinguish between two classes using three non-linearly correlated features. The task includes full data preprocessing, visualization, neural network implementation and training, evaluation, and visual confirmation of decision boundaries.

---

## Dataset Description

The dataset used is based on the classic `make_moons` structure, extended with a third feature to introduce further vertical displacement. It consists of the following columns:

- **X1**: First moons feature  
- **X2**: Second moons feature  
- **X3**: Vertical displacement (added noise/offset to increase complexity)  
- **label**: Target class (0 or 1), where 1 indicates "Yes" and 0 indicates "No"

### Key Characteristics:

- 3 numerical input features  
- Binary classification target  
- Non-linear decision boundary  
- 1,000+ samples with no duplicates  
- Well-balanced class distribution

The dataset was thoroughly analyzed using boxplots, histograms, and a correlation matrix to understand the relationships between features and to confirm class separability is non-trivial.

---

## Methodology

Two separate deep learning pipelines were built:

### 1. PyTorch
- Manual dataset and dataloader creation
- Custom `nn.Module` with 3 hidden layers (ReLU activations)
- Binary cross-entropy loss (`BCELoss`)
- Trained using Adam optimizer
- Model performance monitored via accuracy and confusion matrix
- 3D decision surface plotted using `matplotlib`

### 2. fastai
- Used `TabularPandas` to preprocess and encode data
- Created a learner with `CrossEntropyLossFlat`
- Trained with built-in training loop (`fit_one_cycle`)
- Visualized classification performance and generated predictions

Both frameworks followed the same data split:  
- 60% training  
- 20% validation  
- 20% testing

---

## Key Findings

- The dataset poses a non-trivial classification challenge due to the curved and overlapping feature space.
- Both models achieved high accuracy (>95%) on the test set.
- PyTorch provided more control and transparency during training, especially for manual inspection of weights and predictions.
- fastai enabled rapid prototyping with minimal code and auto-handled much of the boilerplate preprocessing and training logic.
- Visual decision boundaries showed clear learned separation even in the presence of noise.

---

## Framework Comparison

Between the two frameworks, **fastai** was the most efficient for quick experimentation and high-level abstraction. Its API allowed for compact, readable code and fast training. I especially appreciated how fastai handled data preprocessing, splitting, and training loops with just a few lines of code, making it ideal for rapid iteration and model comparison.

On the other hand, **PyTorch** gave full control over model architecture, weight updates, and custom metrics. It was an excellent framework for learning and debugging deep learning logic from the ground up. It required more code, but it made every training step explicit and transparent.

That said, for real-world tasks where productivity and speed matter, I found **fastai** to be the better choice overall—especially for tabular or structured data classification tasks like this one. However, when deep customization or flexibility is required, **PyTorch** remains unmatched. Each framework has its place, and using both has helped me appreciate their strengths in different scenarios.

---

### Requirements

**requirements.txt** :
```
python == 3.10.18
pandas == 2.3.1
matplotlib == 3.10.0
seaborn == 0.13.2
numpy == 1.26.4
scikit-learn == 1.7.1
pytorch == 2.2.2
fastai == 2.7.17
```