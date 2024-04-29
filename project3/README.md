# Recommender System Project

This project explores different recommender system algorithms using the MovieLens 100k dataset. The primary focus is on the Singular Value Decomposition (SVD) method, but it also includes comparisons with K-Nearest Neighbors (KNN) and Non-negative Matrix Factorization (NMF) algorithms.

## Repository Structure

The repository contains the following Python scripts:

- `SVD.py`: Implements the SVD algorithm and outputs its performance metrics.
- `Comparison.py`: Compares the performance of SVD, KNNBasics, and NMF on the same dataset.

## Getting Started

To run these scripts, you will need Python installed on your machine along with the necessary libraries such as `numpy`, `pandas`, `scikit-surprise`, and others typically used for data processing and machine learning tasks.

### Prerequisites

Install all required dependencies using pip:

```bash
pip install numpy pandas scikit-surprise


Install all required dependencies using pip:
```


### Running the Scripts

1. **SVD Algorithm Output**
   - To see the output of the SVD algorithm, run the following command in the terminal:
   ```bash
   python SVD.py
   ```

2. **Performance Comparison**
   - To compare the performance of SVD with KNNBasics and NMF, execute:
   ```bash
   python Comparison.py
   ```

## Additional Information

- Ensure that the dataset is correctly placed in the expected directory if the scripts require loading external data.
- Adjust the script parameters or dataset path as needed to fit your specific setup.

Enjoy exploring the various recommender systems and their efficiencies!
```
