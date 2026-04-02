# Deep Learning — Neural Network Training: Beginner's Guide

> A structured walkthrough of how neural networks learn — covering forward propagation, backpropagation, activation functions, and loss functions.

---

## Table of Contents

- [Neural Network Architecture](#neural-network-architecture)
- [Forward Propagation](#forward-propagation)
- [Backpropagation and Weight Updates](#backpropagation-and-weight-updates)
- [Gradient Descent and Vanishing Gradient Problem](#gradient-descent-and-vanishing-gradient-problem)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Quick Reference — What to Use When](#quick-reference--what-to-use-when)
- [Key Vocabulary Cheatsheet](#key-vocabulary-cheatsheet)

---

## Neural Network Architecture

### Overall Structure

    Deep Learning Model
    │
    ├── Input Layer
    │     └── Features: x1, x2, x3  (raw data fed into the network)
    │
    ├── Hidden Layer(s)
    │     ├── Hidden Layer 1
    │     │     └── Neurons: weighted sum + bias + activation
    │     ├── Hidden Layer 2
    │     │     └── Neurons: weighted sum + bias + activation
    │     └── ... (more layers = deeper network)
    │
    └── Output Layer
          └── Final prediction: y_hat

### What Happens Inside Each Neuron

    Step 1 — Weighted Sum:
          y = (w1 * x1) + (w2 * x2) + (w3 * x3) + b

    Step 2 — Vectorized Form:
          y = w^T * x + b

    Step 3 — Apply Activation Function:
          z = f(y)

| Component           | Symbol   | Role                                                       |
|---------------------|----------|------------------------------------------------------------|
| Inputs              | x1,x2,x3 | Raw data features                                          |
| Weights             | w        | Control how much influence each input has                  |
| Bias                | b        | Lets the neuron activate even when all inputs are zero     |
| Activation Function | f( )     | Introduces non-linearity so the network learns patterns    |
| Prediction          | y_hat    | The final output of the network                            |

> Note: Without activation functions, stacking multiple layers is mathematically equivalent
> to a single linear transformation. The network would fail to learn complex patterns.

---

## Forward Propagation

Forward propagation is the process of passing input data through the network,
layer by layer, to produce a prediction.

### Flow Diagram

    Input Data (x1, x2, x3)
          |
          v
    [ Hidden Layer 1 ]
      y = w^T * x + b
      z = f(y)
          |
          v
    [ Hidden Layer 2 ]
      y = w^T * x + b
      z = f(y)
          |
          v
    [ Output Layer ]
      Prediction: y_hat
          |
          v
    [ Loss Function ]
      L(y, y_hat)  <-- measures how wrong the prediction is

### Key Points

- Weights are randomly initialized before training begins
- Data flows in one direction only: input → hidden layers → output
- The loss function compares the prediction (y_hat) to the true label (y)
- Training goal: minimize the loss by adjusting weights and biases

---

## Backpropagation and Weight Updates

After forward propagation computes the loss, backpropagation sends
the error backward through the network to update weights and reduce the loss.

### Weight Update Formula

    w_new = w_old  -  learning_rate * (dL / dw_old)

    Where:
      learning_rate  = how large each update step is (eta)
      dL / dw        = gradient (slope of the loss w.r.t. the weight)

### The Chain Rule — How Gradients Flow Backward

For deep networks, the gradient of the loss w.r.t. an early weight
is computed by multiplying derivatives layer by layer (chain rule):

    dL/dw4 = (dL/do2) * (do2/dw4)

    Full chain across layers:

    Output Layer --> Hidden Layer 2 --> Hidden Layer 1 --> Input Layer
       dL/dw          * do/dw              * do/dw          = final gradient

    In deeper networks, multiple paths contribute to the gradient,
    requiring summation across all routes.

### Backpropagation Step-by-Step

    1. Run forward propagation  -->  get prediction y_hat
    2. Compute loss             -->  L(y, y_hat)
    3. Compute gradient         -->  dL/dw for each weight
    4. Update each weight       -->  w_new = w_old - eta * (dL/dw)
    5. Repeat for many epochs   -->  until loss converges

### Learning Rate Guide

| Learning Rate     | Effect                                          |
|-------------------|-------------------------------------------------|
| Too large         | Overshoots minimum, oscillates or diverges      |
| Too small         | Extremely slow convergence                      |
| Ideal (0.001–0.01)| Stable and efficient learning                   |

---

## Gradient Descent and Vanishing Gradient Problem

### Gradient Descent

Gradient descent is the optimization algorithm that iteratively updates
weights to reach the global minimum of the loss function.

    Loss
      |
      |   *  <-- Start (random weights, high loss)
      |    \
      |     *  <-- After update 1
      |      \
      |       *  <-- After update 2
      |        \_____ Global Minimum  <-- Goal
      |
      +--------------------------------- Weights

    Rule:
      Negative slope  -->  increase the weight
      Positive slope  -->  decrease the weight
      Slope near 0    -->  at or near the minimum (converged)

---

### Vanishing Gradient Problem

One of the most critical challenges when training deep neural networks.

#### Why It Happens

    Sigmoid derivative range:  0 to 0.25

    Backpropagation through 4 layers multiplies small derivatives:
      gradient * 0.2 * 0.2 * 0.2 * 0.2  =  0.0016   <-- nearly zero

    The further back a layer is, the smaller its gradient becomes.
    Early layers stop learning because their weight updates are near zero.

#### Vanishing Gradient — Layer by Layer

    Output Layer      --> gradient = 0.5       (healthy)
         |
    Hidden Layer 3    --> gradient = 0.10      (reduced)
         |
    Hidden Layer 2    --> gradient = 0.02      (very small)
         |
    Hidden Layer 1    --> gradient = 0.0016    (almost zero, barely learns)

| Symptom                              | Cause                                         |
|--------------------------------------|-----------------------------------------------|
| Weights barely change in early layers| Gradients shrink exponentially going backward |
| Training slows down or stops         | Near-zero gradients = near-zero updates       |
| Deep networks underperform           | Early layers fail to capture useful features  |

> Solution: Use ReLU or its variants in hidden layers instead of Sigmoid or Tanh.

---

## Activation Functions

Activation functions introduce non-linearity. Their choice directly impacts
gradient flow, training speed, and overall network performance.

### Activation Functions — Decision Tree

    Which activation function should I use?
    │
    ├── Output Layer?
    │     ├── Binary Classification   --> Sigmoid
    │     ├── Multi-Class             --> Softmax
    │     └── Regression              --> Linear (no activation)
    │
    └── Hidden Layer?
          ├── Default choice          --> ReLU
          ├── Dead neuron problem?    --> Leaky ReLU
          ├── Need zero-centered?     --> Tanh
          └── Advanced / research     --> ELU or PReLU

---

### Sigmoid

    Formula  :  sigma(x) = 1 / (1 + e^(-x))
    Output   :  0 to 1
    Derivative: 0 to 0.25

    Graph shape:
      1 |          .........
        |       ...
        |     ..
      0 |.....
        +-----------------> x

| Pros                          | Cons                                      |
|-------------------------------|-------------------------------------------|
| Output normalized between 0–1 | Not zero-centered, inefficient updates    |
| Smooth gradient               | Prone to vanishing gradient               |
| Interpretable as probability  | Computationally expensive (exponentials)  |

---

### Tanh (Hyperbolic Tangent)

    Formula  :  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Output   :  -1 to +1   (zero-centered)
    Derivative: 0 to 1

    Graph shape:
     +1 |         ..........
        |      ...
        |    ..
        |  ..
     -1 |..
        +-----------------> x

| Pros                             | Cons                                        |
|----------------------------------|---------------------------------------------|
| Zero-centered output             | Still suffers from vanishing gradient       |
| Better gradient flow than Sigmoid| Computationally expensive (exponentials)    |

---

### ReLU (Rectified Linear Unit)

    Formula  :  f(x) = max(0, x)
    Output   :  0 to infinity

    Derivative:
      f'(x) = 0   if x <= 0
      f'(x) = 1   if x >  0

    Graph shape:
        |         /
        |        /
        |       /
      0 |______/
        +-----------------> x

| Pros                                    | Cons                                        |
|-----------------------------------------|---------------------------------------------|
| Computationally very efficient          | Dead neuron problem (zero gradient for x<=0)|
| No vanishing gradient for positive x   | Neurons can permanently stop learning       |
| Enables fast convergence               |                                             |

> Default activation for hidden layers in most modern networks.

---

### Leaky ReLU

    Formula  :  f(x) = 0.01 * x    if x <= 0
                f(x) = x            if x >  0

    Graph shape:
        |          /
        |         /
      0 |--------/
        |   /  (small negative slope = 0.01)
        +-----------------> x

| Pros                        | Cons                                   |
|-----------------------------|----------------------------------------|
| Fixes dead neuron problem   | Slope value (0.01) is a manual choice  |
| Gradient never zero         | Still experimental in some settings    |

---

### ELU and PReLU

    ELU (Exponential Linear Unit):
      f(x) = x               if x > 0
      f(x) = alpha*(e^x - 1) if x <= 0

    PReLU (Parametric ReLU):
      f(x) = x               if x > 0
      f(x) = a * x           if x <= 0
      where 'a' is a learnable parameter

| Function | Key Improvement               | Trade-off                         |
|----------|-------------------------------|-----------------------------------|
| ELU      | Smooth negative values        | Computationally expensive         |
| PReLU    | Learnable negative slope      | Adds parameters to train          |

---

### Softmax

Used exclusively in the output layer for multi-class classification.

    Formula  :  softmax(zi) = e^zi / sum(e^zj)   for all classes j

    Example (3 classes):
      Raw scores:        [2.0,  1.0,  0.5]
      After softmax:     [0.63, 0.23, 0.14]
                          ^--- all values sum to 1.0

- Converts raw output scores (logits) into a probability distribution
- Each output value represents the probability of belonging to that class
- All outputs always sum to 1

---

### Activation Functions — Full Comparison Table

| Function    | Output Range  | Zero-Centered | Vanishing Gradient | Dead Neurons | Best Used In              |
|-------------|---------------|---------------|--------------------|--------------|---------------------------|
| Sigmoid     | 0 to 1        | No            | Yes (severe)       | No           | Binary output layer       |
| Tanh        | -1 to +1      | Yes           | Yes (moderate)     | No           | Hidden layers (shallow)   |
| ReLU        | 0 to infinity | No            | No                 | Yes          | Hidden layers (default)   |
| Leaky ReLU  | -inf to +inf  | No            | No                 | No           | Hidden layers              |
| ELU         | ~-1 to +inf   | Approx        | No                 | No           | Hidden layers (advanced)  |
| PReLU       | -inf to +inf  | No            | No                 | No           | Hidden layers (advanced)  |
| Softmax     | 0 to 1        | No            | --                 | No           | Multi-class output layer  |

---

## Loss Functions

Loss functions measure the gap between the predicted output and the true label.
They guide the weight updates during training.

### Loss Functions — Tree Structure

    Loss Functions
    │
    ├── Regression (predicting a continuous value)
    │     ├── MSE   — Mean Squared Error
    │     │           Sensitive to outliers, fast convergence
    │     ├── MAE   — Mean Absolute Error
    │     │           Robust to outliers, slower convergence
    │     └── Huber — Hybrid of MSE + MAE
    │                 Best of both worlds
    │
    └── Classification (predicting a category)
          ├── Binary Cross-Entropy
          │     For 2-class problems (yes/no, pass/fail)
          │     Paired with Sigmoid output
          └── Categorical Cross-Entropy
                For 3+ class problems
                Paired with Softmax output

---

### MSE — Mean Squared Error

    Loss per sample  :  L = (1/2) * (y - y_hat)^2
    Cost over dataset:  J = (1/2) * sum( (yi - y_hat_i)^2 )

| Pros                        | Cons                                         |
|-----------------------------|----------------------------------------------|
| Differentiable everywhere   | Sensitive to outliers (squares large errors) |
| Single global minimum       | Penalizes large errors very heavily          |
| Converges quickly           |                                              |

---

### MAE — Mean Absolute Error

    Loss per sample  :  L = (1/2) * |y - y_hat|
    Cost over dataset:  J = (1/2) * sum( |yi - y_hat_i| )

| Pros                        | Cons                                             |
|-----------------------------|--------------------------------------------------|
| Robust to outliers          | Not differentiable at zero                       |
| Penalizes errors linearly   | Requires subgradient methods                     |
|                             | Slower convergence compared to MSE               |

---

### Huber Loss

Combines MSE and MAE using a threshold hyperparameter delta.

    If |y - y_hat| <= delta :  L = (1/2) * (y - y_hat)^2    <-- behaves like MSE
    If |y - y_hat| >  delta :  L = delta * (|y - y_hat| - (1/2)*delta)  <-- behaves like MAE

    Transition point:
      Small error  -->  MSE behaviour  (smooth, fast gradient)
      Large error  -->  MAE behaviour  (robust to outliers)

| Pros                                   | Cons                             |
|----------------------------------------|----------------------------------|
| Robust to outliers like MAE            | Requires tuning delta            |
| Differentiable everywhere like MSE     |                                  |

---

### Binary Cross-Entropy (Log Loss)

Used for binary classification problems where output is 0 or 1.
Always paired with Sigmoid activation in the output layer.

    L = -[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

    Example:
      True label y = 1,  prediction y_hat = 0.9  -->  Low loss (correct)
      True label y = 1,  prediction y_hat = 0.1  -->  High loss (wrong)

---

### Categorical Cross-Entropy

Used for multi-class classification with 3 or more classes.
Always paired with Softmax activation in the output layer.

    L = - sum( y_ij * log(y_hat_ij) )   for all classes j

    True label (one-hot): [0, 1, 0]       (class 2 is correct)
    Prediction (softmax): [0.1, 0.8, 0.1] (high probability on class 2)
    Loss                : low             (prediction matches truth)

---

### Loss Functions — Full Comparison Table

| Loss Function           | Task           | Outlier Robust | Paired With      |
|-------------------------|----------------|----------------|------------------|
| MSE                     | Regression     | No             | Linear output    |
| MAE                     | Regression     | Yes            | Linear output    |
| Huber                   | Regression     | Yes            | Linear output    |
| Binary Cross-Entropy    | Classification | --             | Sigmoid output   |
| Categorical Cross-Entropy| Classification| --             | Softmax output   |

---

## Quick Reference — What to Use When

    Problem Type          Hidde