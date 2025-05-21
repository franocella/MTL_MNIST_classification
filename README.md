# Multi-Task Learning (MTL) vs. Single-Task Learning (STL) for MNIST-derived Binary Classification

This notebook explores and compares **Multi-Task Learning (MTL)** and **Single-Task Learning (STL)** approaches for binary image classification using a derivative of the MNIST dataset. The project investigates different parameter sharing mechanisms within simplified Multi-Layer Perceptron (MLP) architectures.

## Table of Contents

* [Tasks](#tasks)
* [Learning Approaches Explored](#learning-approaches-explored)
* [Objectives](#objectives)
* [Libraries Used](#libraries-used)
* [Setup and Configuration](#setup-and-configuration)
    * [Device Selection](#device-selection)
* [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    * [MNIST Dataset and Task Definitions](#mnist-dataset-and-task-definitions)
    * [Data Normalization](#data-normalization)
    * [Custom Dataset Class](#custom-dataset-class)
    * [DataLoaders](#dataloaders)
* [Model Architectures](#model-architectures)
    * [MLP Base Architecture](#mlp-base-architecture)
    * [Hard Parameter Sharing MTL](#hard-parameter-sharing-mtl)
    * [Soft Parameter Sharing MTL (Cross-Stitch Networks)](#soft-parameter-sharing-mtl-cross-stitch-networks)
    * [Single-Task Learning](#single-task-learning)
* [Utility Functions](#utility-functions)
* [Training and Evaluation](#training-and-evaluation)
* [Results and Discussion](#results-and-discussion)
    * [Training Progress Visualization](#training-progress-visualization)
    * [Final Test Set Performance Summary](#final-test-set-performance-summary)
    * [Interpretation of Results](#interpretation-of-results)
* [Conclusion](#conclusion)
* [License](#license)

-----

## Tasks

Two distinct binary classification tasks are defined based on the MNIST dataset:

* **Task 1: Is Even?** Classify digits as "even" (1) or "odd" (0).
* **Task 2: Is Multiple of 3?** Classify digits as "multiple of 3 and not 0".

-----

## Learning Approaches Explored

The notebook implements and compares three main learning strategies:

* **Hard Parameter Sharing MTL:** A single MLP network with **shared hidden layers** (acting as a feature extractor) and **task-specific output heads** (classifiers) for each task.

* **Soft Parameter Sharing MTL (Cross-Stitch Networks):** Two separate MLP networks (one for each task) are used, but they incorporate `CrossStitchUnit`s. These units are placed at intermediate layers and learn to combine feature maps between the two tasks. Essentially, the activation of a layer for one task becomes a **learned linear combination** of its own activation and the activation of the corresponding layer from the other task. This allows for a flexible way of sharing information, where the model learns *how much* information to share. The scalar coefficients for these combinations are initialized to behave like an identity matrix and are adjusted during training.

* **Single-Task Learning (STL):** Two separate MLP networks, one for each task, are trained **independently** with no parameter sharing. This serves as a baseline for comparison.

-----

## Objectives

* Implement 2 Multi-Task learning techniques:
    * Hard parameter sharing
    * Soft parameter sharing (Cross-Stitch Networks)
* Compare the performance of MTL approaches against independent single-task models.
* Analyze the trade-offs in terms of **model complexity** (parameter count), **computational cost** (training and inference time), and **task performance**.

-----

## Libraries Used

* **PyTorch & Torchvision:** For building and training neural networks, loading the MNIST dataset, and applying image transformations.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For plotting training results and confusion matrices.
* **Pandas:** For creating and managing DataFrames to summarize results.
* **Scikit-learn:** For calculating evaluation metrics like confusion matrices and accuracy score.
* **Time:** For measuring training and inference durations.

-----

## Setup and Configuration

### Device Selection

For compatibility and consistent execution within various environments, the notebook is configured to explicitly use:

* **CPU** (`torch.device('cpu')`).

-----

## Data Loading and Preprocessing

### MNIST Dataset and Task Definitions

The standard MNIST dataset, consisting of grayscale 28x28 pixel images of handwritten digits (0-9), is automatically downloaded. A custom `MNISTCustomTasks` dataset class is used to generate labels for the two binary tasks clarified before.

### Data Normalization

Images are converted to PyTorch tensors and then normalized using pre-calculated MNIST mean and standard deviation:

* **Mean:** `0.1307`
* **Standard Deviation:** `0.3081`

### Custom Dataset Class

A custom PyTorch `Dataset` class, `MNISTCustomTasks`, wraps the original MNIST dataset. Its `__getitem__` method is overridden to return:

* The transformed image.
* The binary label for "Is Even?".
* The binary label for "Is Multiple of 3?".

This setup allows a single dataset instance to provide data for both tasks simultaneously, which is essential for MTL.

### DataLoaders

PyTorch `DataLoader` instances are created for both training and testing datasets using a `BATCH_SIZE = 128`.

* `train_loader` shuffles the data at each epoch.
* `test_loader` does not shuffle the data for consistent evaluation.

-----

## Model Architectures

All MLP backbones (shared or task-specific) consist of **two hidden layers**.

### MLP Base Architecture

The foundational MLP structure, used in both STL and as the building block for MTL models, includes:

* An input layer to flatten the 28x28 image (784 features).
* Two hidden layers with ReLU activation (sizes `[256, 128]`).
* An output layer for binary classification.

### Hard Parameter Sharing MTL

**Architecture:**

* A shared backbone comprising the flattened input layer and the two hidden layers with ReLU activations.
* Two task-specific output heads, one for each binary task, connected to the shared backbone's output.

**Total Parameters:** 52,386. This is significantly less than the sum of STL models, highlighting **parameter efficiency**.

### Soft Parameter Sharing MTL (Cross-Stitch Networks)

**Architecture:**

* Two separate MLP pathways, one for each task, each containing the input layer and two hidden layers with ReLU activations.
* `CrossStitchUnit` modules are placed after the ReLU activation of each hidden layer in both pathways. These units learn to combine feature maps from corresponding layers across the two task pathways.

**Total Parameters:** 104,714. This is slightly more than the sum of STL models due to the added `CrossStitchUnit` parameters.

### Single-Task Learning

**Architecture:**

* A completely separate MLP model is trained for each task. Each model has its own input layer, two hidden layers with ReLU activations, and a task-specific output layer.

**Total Parameters:**

* Even-only Single-Task Model: 52,353
* MultipleOf3-only Single-Task Model: 52,353
* Sum of Single-Task Model Parameters: 104,706

-----

## Utility Functions

* `train_model(model, loader, optimizer, criterion, model_type, num_epochs)`: Manages the training loop, including forward/backward passes, optimizer steps, loss calculation, and saving the best model weights based on validation accuracy per epoch.
* `evaluate_model(model, loader, criterion, model_type)`: Performs evaluation on a dataset, calculating loss and accuracy metrics without gradient computation.
* `get_predictions(model, loader, device, model_type)`: Gathers true labels and predicted labels for confusion matrix generation.
* `plot_training_history(history, model_name, tasks_involved)`: Visualizes training and validation loss/accuracy curves.
* `plot_confusion_matrix_custom(y_true, y_pred, classes, title_suffix)`: Generates and displays a confusion matrix.

-----

## Training and Evaluation

For each model type (STL Even, STL MultipleOf3, HardShareMTL, CrossStitchMTL):

* **Initialization:** Model, `nn.CrossEntropyLoss` (for binary classification), and Adam optimizer (with `lr=0.001`) are initialized.
* **Training Loop:** Models are trained for `NUM_EPOCHS = 20`. In each epoch, the model is trained on the training set and periodically evaluated on the test set.
* **Best Model Saving:** The model weights that achieve the best test accuracy (or average test accuracy for MTL models) are saved to disk. The final evaluation uses these saved "best" models.
* **Metrics Recording:** Training losses and test accuracies for each task are recorded per epoch for later visualization.
* **Final Testing:** The best saved model is loaded and evaluated on the test set to report final metrics.

-----

## Results and Discussion

### Training Progress Visualization

Plots are generated for each model, showing:

* Training Loss vs. Epoch.
* Validation Accuracy vs. Epoch for Each Task.

These plots help in analyzing training dynamics, convergence, and potential overfitting. Confusion matrices for final test predictions are also generated for each task.

### Final Test Set Performance Summary

| Metric | Single-Task | Hard Sharing MTL | Cross-Stitch MTL |
| :-------------------------- | :---------- | :--------------- | :--------------- |
| Total Parameters | 104706.0 | 52386.0 | 104714.0 |
| Training Time (s) | 771.1 | 393.25 | 425.95 |
| Best Average Accuracy (%) | 98.51 | 98.44 | 98.64 |

### Interpretation of Results

Based on the provided data for this introductory example:

* **Parameter Efficiency:** The **Hard Sharing MTL** model (52,386 parameters) demonstrates significant parameter efficiency, requiring roughly half the parameters of the **Single-Task** models (104,706 total parameters) and the **Cross-Stitch MTL** model (104,714 parameters). This is a major advantage for deployment on resource-constrained devices.
* **Training Time:** Both MTL approaches show considerable training time reduction. **Hard Sharing MTL** completed training in 393.25 seconds, and **Cross-Stitch MTL** in 425.95 seconds, which is roughly half the time compared to training two independent **Single-Task** models (771.1 seconds combined). This highlights the computational benefit of joint training.
* **Performance:**
    * The **Cross-Stitch MTL** model achieved the highest average accuracy (98.64%), slightly outperforming **Single-Task** (98.51%) and **Hard Sharing MTL** (98.44%). This suggests that for these tasks, the learned flexible sharing of Cross-Stitch might offer a minor edge in combined performance.
    * The performance difference across all three approaches in terms of average accuracy is very small (less than 0.2%). This indicates that for these specific, relatively simple, and likely highly correlated binary classification tasks on MNIST, all methods are quite effective.
    * This example does not show significant "negative transfer" (where MTL negatively impacts one task's performance), which is positive.

-----

## Conclusion

This notebook provides an introductory implementation and comparative analysis of Single-Task Learning, Hard Parameter Sharing MTL, and Soft Parameter Sharing (Cross-Stitch) MTL for binary classification tasks derived from MNIST.

The key takeaways are:

* **Efficiency:** MTL, especially with Hard Parameter Sharing, offers substantial advantages in terms of parameter efficiency and reduced training time.
* **Performance:** While all methods achieved high accuracy for these tasks, Soft Parameter Sharing (Cross-Stitch) showed a slight advantage in average accuracy, indicating its potential for more nuanced information sharing.
* **Margin for Improvement:** This is an introductory example. Multi-Task Learning is a promising field, and there is significant room for improvement and further exploration. This includes experimenting with more complex architectures, diverse datasets, different task relationships, advanced loss weighting schemes, and exploring the learned alpha values in Cross-Stitch units to understand information flow.

-----

## License

MIT License

Copyright (c) 2025 [Francesco Nocella]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
