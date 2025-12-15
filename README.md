## Deep Learning for California Housing Pricing Regression

This project compares the performance of Linear/Polynomial Regression and Neural Networks (MLP) on a regression task using various hyperparameters and training strategies.

**Dataset:** California Housing Dataset (20,640 samples, 8 features).

### 1. Exploratory Data Analysis (EDA)

* Data inspection using `head()` and `describe()`.
* Missing value analysis (confirmed 0 missing values).

#### Random linear model
Visualization of decision boundaries using dummy data.
* ![Contour plot of dummy model decision boundary](figures/Linear%20Models.png)

### 2. Data Preparation

* **Feature Engineering:** Generated Polynomial Features (Degrees: 1, 2, 3, 4).
* **Scaling:** Applied Standard Scaling to features.
* **Splitting:** Split data into Train, Validation, and Test sets.
* **Conversion:** Converted data to PyTorch Tensors.
* **Data Loading:** Created DataLoaders for different strategies (Stochastic, Mini-Batch, Full Batch).

### 3. Models Evaluated

* **Linear Regression** (Closed Form & Gradient Descent).
* **Polynomial Regression** (Degrees 1-4).
* **Neural Networks (MLP)** with varying hidden layers (e.g., 2 layers) and activation functions (e.g., Sigmoid).
* **Optimization Strategies:** Normal Equation, SGD (Normal, Momentum), RMSProp, Adam.
* **Regularization:** None, L1 (Lasso), L2 (Ridge).

### 4. Model Evaluation

* **Metrics:** MSE Loss, R2 Score (labeled as Accuracy in logs).
* **Performance Plots:**
    * ![Training Report](figures/Report%201-mb16-norm-none.png)
    * ![Training Report](figures/Report%201-mb512-adam-2.png)
    * ![Training Report](figures/Report%201-mb512-closed-1.png)
    * ![Hyperparameter variation](figures/Variation%20against%201-batch-closed-none.png)
    * ![Hyperparameter variation](figures/Variation%20against%203-batch-adam-none.png)

![Corr](figures/corr%20hyps%20vs%20metrs.png)

### 5. Final Results

* **Best Model Configuration:** Neural Network (2 Hidden Layers, Sigmoid Activation), Adam Optimizer (LR=0.0033), L2 Regularization, Mini-Batch size 2048.
    * ![Hyperparameter variation](figures/Variation%20against%202-mb2048-adam-none.png)
* **Final Metrics:**
    * **Train Loss:** ~0.22
    * **Train R2 Score:** ~0.90
* **Key Findings:**
    * Data Loader strategy was the most dominant factor affecting performance.
    * L1 Regularization proved too aggressive for this dataset.
    * Adam optimizer provided faster convergence than standard SGD.
* **Final Visualizations:**
    * ![Final Model Training Report on Test Set](figures/Report%202-mb2048-adam-0.png)
    * ![Final Model Training Report on Test Set](figures/variations%20finer.png)