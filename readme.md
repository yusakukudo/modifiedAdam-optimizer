# ModifiedAdam Optimizer Project 🚀 ![Python](https://img.shields.io/badge/python-3.12-blue)

This repository implements and evaluates a novel optimizer, **ModifiedAdam**, which enhances the standard Adam optimizer by introducing a "lookahead" gradient adjustment. 

---

## Motivation
The standard Adam optimizer is widely used in deep learning due to its adaptive learning rates and momentum. However, it struggles with:
- Slow adaptation to rapid gradient changes.
- Suboptimal updates in non-stationary landscapes.

ModifiedAdam introduces a "lookahead" adjustment to improve responsiveness and convergence, making it suitable for complex optimization tasks.

---
## Features
- Incorporates a "lookahead" gradient adjustment for better adaptability.
- Tested on classification and regression tasks.
- Outperforms Adam and RMSprop on multiple benchmarks.
---
## Datasets
- **[MNIST Dataset](http://yann.lecun.com/exdb/mnist/):** Handwritten digit recognition dataset with 60,000 training and 10,000 testing images.
- **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist):** Dataset of 10 classes of clothing items.
- **[Diamond Dataset](https://www.kaggle.com/shivam2503/diamonds):** Regression dataset with features like carat, cut, and price.

---

## 📁 Project Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for preprocessing, training, and visualization.
- `results/`: Contains generated charts, figures, and evaluation metrics.
- `src/`: Source code for training models and utility functions.

---

## 📊 Results
### MNIST Dataset
| Optimizer     | Loss        | Accuracy (%) | F1 Score (%) |
|---------------|-------------|--------------|--------------|
| ModifiedAdam  | **0.0793**  | **98.02**    | **98.29**    |
| Adam          | 0.0845      | 97.96        | 97.93        |
| RMSprop       | 0.1576      | 96.64        | 96.63        |

### Fashion-MNIST Dataset
| Optimizer     | Accuracy (%) | Precision (%) | Recall (%) |
|---------------|--------------|---------------|------------|
| ModifiedAdam  | **89.13**    | **89.08**     | **89.13**  |
| Adam          | 89.03        | 89.00         | 89.03      |
| RMSprop       | 85.27        | 85.67         | 85.27      |

### Diamond Dataset
| Optimizer     | Loss        | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) |
|---------------|-------------|---------------------------|---------------------------------|
| ModifiedAdam  | 901245.14   | **513.87**                | 949.44                         |
| Adam          | 852978.38   | 535.81                   | 923.27                         |
| RMSprop       | 2091954.57  | 756.42                   | 1443.20                        |

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/ModifiedAdam-Project.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks:
   ```bash
   jupyter notebook
   ```

## Acknowledgments
- MNIST and Fashion-MNIST datasets by [Yann LeCun et al.](http://yann.lecun.com/exdb/mnist/)
- Diamond dataset from [Kaggle](https://www.kaggle.com/shivam2503/diamonds)
- Original Adam optimizer by [Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)

---
## Future Work
- Extend benchmarking to larger datasets like CIFAR-10 or ImageNet.
- Incorporate additional optimizers for comparison, such as Nadam or AdaMax.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.