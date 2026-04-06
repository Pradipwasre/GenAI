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

    Problem Type          Hidden Layers       Output Layer     Loss Function
    ─────────────────────────────────────────────────────────────────────────
    Binary Classification  ReLU / Leaky ReLU  Sigmoid          Binary Cross-Entropy
    Multi-Class            ReLU / Leaky ReLU  Softmax          Categorical Cross-Entropy
    Regression             ReLU / Leaky ReLU  Linear (none)    MSE / MAE / Huber
    ─────────────────────────────────────────────────────────────────────────

    Learning Rate Recommendations:
      Start with  : 0.001
      If too slow : try 0.01
      If unstable : reduce back to 0.0001

---

## Key Vocabulary Cheatsheet

| Term                      | Simple Definition                                                    |
|---------------------------|----------------------------------------------------------------------|
| Forward Propagation       | Passing input data through the network to get a prediction           |
| Backward Propagation      | Sending error backward through the network to update weights         |
| Weight (w)                | Controls how much influence an input has on a neuron                 |
| Bias (b)                  | Allows a neuron to activate even when all inputs are zero            |
| Activation Function       | Introduces non-linearity into the network                            |
| Loss Function             | Measures how wrong the prediction is                                 |
| Gradient                  | The slope of the loss curve — shows direction to update weights      |
| Gradient Descent          | Optimization algorithm that minimizes loss by updating weights       |
| Learning Rate (eta)       | Controls the size of each weight update step                         |
| Chain Rule                | Math rule that allows gradients to flow backward through layers      |
| Vanishing Gradient        | When gradients shrink to near zero in early layers, stalling learning|
| Epoch                     | One complete pass through the entire training dataset                |
| Sigmoid                   | Activation outputting 0–1, used for binary output layers             |
| Tanh                      | Activation outputting -1 to +1, zero-centered                       |
| ReLU                      | Activation outputting 0 to infinity, default for hidden layers       |
| Leaky ReLU                | ReLU variant that fixes the dead neuron problem                      |
| Softmax                   | Converts scores to probabilities for multi-class output layers       |
| MSE                       | Mean Squared Error — regression loss sensitive to outliers           |
| MAE                       | Mean Absolute Error — regression loss robust to outliers             |
| Huber Loss                | Hybrid of MSE and MAE — robust yet differentiable                    |
| Binary Cross-Entropy      | Classification loss for 2-class problems                             |
| Categorical Cross-Entropy | Classification loss for 3+ class problems                            |


## Interview Cheatsheet : All Key Terms

| Term | Interview-Style Answer |
|---|---|
| Forward Propagation | It is the process where input data passes through the network layer by layer — applying weights, bias, and activation functions — to produce a final prediction. We always run this first before computing the loss. |
| Backward Propagation | After computing the loss, we send the error signal backward through the network using the chain rule to calculate gradients for each weight, so we know how much to adjust them to reduce the loss. |
| Weight (w) | Weights are learnable parameters that control how much influence each input has on a neuron's output. During training, backpropagation updates these weights to minimize the loss. |
| Bias (b) | Bias is an extra learnable parameter added to the weighted sum. It allows a neuron to activate even when all input values are zero, giving the model more flexibility to fit the data. |
| Activation Function | A mathematical function applied after the weighted sum to introduce non-linearity. Without it, no matter how many layers we stack, the network behaves like a single linear model and cannot learn complex patterns. |
| Loss Function | A function that measures how far the model's prediction is from the true label. It gives us a single number that we try to minimize during training. The choice of loss function depends on the task — regression or classification. |
| Gradient | The gradient is the partial derivative of the loss with respect to a weight. It tells us the slope of the loss curve at that point — specifically, in which direction and by how much we should update the weight to reduce the loss. |
| Gradient Descent | An iterative optimization algorithm that updates model weights in the direction opposite to the gradient, moving toward the global minimum of the loss function. The size of each step is controlled by the learning rate. |
| Learning Rate (eta) | A hyperparameter that controls how large each weight update step is during gradient descent. Too large causes the model to overshoot and diverge. Too small causes extremely slow convergence. Typical values are 0.001 to 0.01. |
| Chain Rule | A calculus rule used in backpropagation to compute gradients through multiple layers. It breaks down the derivative of a composite function into a product of simpler derivatives, layer by layer, from output back to input. |
| Vanishing Gradient | A problem that occurs in deep networks when gradients shrink exponentially as they propagate backward through layers. Activation functions like Sigmoid have derivatives between 0 and 0.25, so multiplying many of these together causes early layers to receive near-zero gradients and stop learning. |
| Epoch | One complete pass through the entire training dataset — both forward and backward propagation across all samples. Training typically requires many epochs before the model converges to a good solution. |
| Sigmoid | An activation function that outputs values between 0 and 1. It is used in binary classification output layers because its output can be interpreted as a probability. However, it suffers from the vanishing gradient problem and should not be used in hidden layers of deep networks. |
| Tanh | An activation function that outputs values between -1 and +1. It is zero-centered, which makes gradient updates more efficient than Sigmoid. However, it still suffers from vanishing gradients in very deep networks. |
| ReLU | Rectified Linear Unit. It outputs 0 for negative inputs and the input value itself for positive inputs. It is the default choice for hidden layers because it is computationally efficient, avoids vanishing gradients for positive values, and enables faster convergence. Its main weakness is the dead neuron problem. |
| Leaky ReLU | A variant of ReLU that fixes the dead neuron problem by allowing a small negative slope (0.01 * x) for negative inputs instead of outputting zero. This ensures gradients never become exactly zero, so all neurons continue to learn. |
| Softmax | An activation function used in the output layer for multi-class classification. It converts raw output scores (logits) into a probability distribution where all values sum to 1, allowing us to interpret each value as the probability of belonging to that class. |
| MSE | Mean Squared Error. A regression loss function that squares the difference between prediction and true value. It converges quickly and has a single global minimum, but it is sensitive to outliers because large errors get penalized exponentially. |
| MAE | Mean Absolute Error. A regression loss function that takes the absolute difference between prediction and true value. It is more robust to outliers than MSE because it penalizes errors linearly, but it is not differentiable at zero which makes optimization slightly harder. |
| Huber Loss | A hybrid regression loss that behaves like MSE for small errors and like MAE for large errors, controlled by a threshold hyperparameter delta. It gives the best of both worlds — fast convergence on small errors and robustness to outliers on large errors. |
| Binary Cross-Entropy | The standard loss function for binary classification tasks where the output is either 0 or 1. It heavily penalizes confident wrong predictions using a logarithmic scale. It is always paired with a Sigmoid activation in the output layer. |
| Categorical Cross-Entropy | The standard loss function for multi-class classification tasks with 3 or more classes. It measures the difference between the predicted probability distribution and the true one-hot encoded label. It is always paired with a Softmax activation in the output layer. |





# Activation Functions — Forward Pass to Final Prediction

> Same inputs. Different activation functions. See exactly what changes at each step.

---

## Table of Contents

- [Given Values](#given-values)
- [Step 1 — Weighted Sum](#step-1--weighted-sum-same-for-all-activations)
- [Step 2 — Apply Each Activation](#step-2--apply-each-activation-at-hidden-neuron)
- [Step 3 — Output Layer](#step-3--output-layer-weighted-sum)
- [Step 4 — Sigmoid at Output](#step-4--sigmoid-activation-at-output-layer)
- [Step 5 — Loss](#step-5--loss-calculation)
- [Full Summary Table](#full-end-to-end-summary-table)
- [Key Takeaways](#key-takeaways)
- [When to Use Which](#when-to-use-which-activation)
- [Softmax — Multi-Class Output](#softmax--output-layer-for-multi-class)

---

## Given Values

    Inputs  :  x1 = 1.0,   x2 = 0.5,   x3 = -0.8
    Weights :  w1 = 0.4,   w2 = 0.6,   w3 = 0.5,   b1 = 0.1
    Output  :  w4 = 0.8,   b2 = 0.0
    True label y = 1  (binary classification)

---

## Step 1 — Weighted Sum (same for all activations)

    y  =  (w1 * x1)  +  (w2 * x2)  +  (w3 * x3)  +  b1
    y  =  (0.4 * 1.0)  +  (0.6 * 0.5)  +  (0.5 * -0.8)  +  0.1
    y  =   0.40  +  0.30  -  0.40  +  0.10
    y  =   0.40

> The pre-activation value y = 0.40 is identical for all activations.
> What changes is how we transform y into z in the next step.

---

## Step 2 — Apply Each Activation at Hidden Neuron

### 1. ReLU — default choice

    Formula    :  z = max(0, y)
    z          =  max(0, 0.40)  =  0.40
    Derivative :  dz/dy = 1   (because y = 0.40 > 0)

    Output range : 0 to infinity
    y > 0        : value passes through unchanged
    y <= 0       : output = 0  and  derivative = 0  (dead neuron)

---

### 2. Leaky ReLU — fixes dead neuron problem

    Formula    :  z = y          if y > 0
                  z = 0.01 * y   if y <= 0

    Since y = 0.40 > 0:
    z          =  0.40
    Derivative :  dz/dy = 1

    Output range : -infinity to +infinity
    y > 0        : same as ReLU
    y <= 0       : small negative signal kept alive

    Compare when y = -0.55:
      ReLU        →  z = 0           gradient = 0      (neuron is dead)
      Leaky ReLU  →  z = -0.0055     gradient = 0.01   (neuron stays alive)

---

### 3. Tanh — zero-centered output

    Formula    :  z = (e^y - e^(-y)) / (e^y + e^(-y))

    e^0.40  = 1.4918
    e^-0.40 = 0.6703

    z  =  (1.4918 - 0.6703) / (1.4918 + 0.6703)
    z  =  0.8215 / 2.1621
    z  =  0.3799

    Derivative :  dz/dy = 1 - z^2 = 1 - (0.3799)^2 = 1 - 0.1443 = 0.8557

    Output range   : -1 to +1
    Zero-centered  : yes  (better gradient flow than Sigmoid)
    Vanishing grad : still possible in very deep networks

---

### 4. ELU — Exponential Linear Unit  (alpha = 1.0)

    Formula    :  z = y                  if y > 0
                  z = alpha * (e^y - 1)  if y <= 0

    Since y = 0.40 > 0:
    z          =  0.40
    Derivative :  dz/dy = 1

    Output range : -alpha to +infinity
    y > 0        : same as ReLU
    y <= 0       : smooth exponential curve instead of flat zero

    Compare when y = -0.55:
      Leaky ReLU  →  z = -0.0055     derivative = 0.01
      ELU         →  z = 1.0 * (e^-0.55 - 1)
                   →  z = 1.0 * (0.5769 - 1)
                   →  z = -0.4231    derivative = 0.5769
      ELU learns much faster on negative inputs because its derivative is larger.

---

### Activation Comparison Table  (y = 0.40)

| Activation  | Formula                          | z output | dz/dy | Output range      |
|-------------|----------------------------------|----------|-------|-------------------|
| ReLU        | max(0, y)                        | 0.40     | 1     | 0 to infinity     |
| Leaky ReLU  | y if y>0,  else 0.01*y           | 0.40     | 1     | -inf to +inf      |
| Tanh        | (e^y - e^-y) / (e^y + e^-y)     | 0.3799   | 0.8557| -1 to +1          |
| ELU (a=1)   | y if y>0,  else a*(e^y - 1)      | 0.40     | 1     | -1 to +inf        |

---

## Step 3 — Output Layer Weighted Sum

    Formula :  y_out  =  w4 * z  +  b2
               w4 = 0.8,   b2 = 0.0

| Activation | z      | y_out = 0.8 * z + 0.0 |
|------------|--------|------------------------|
| ReLU       | 0.4000 | 0.8 * 0.4000 = 0.3200  |
| Leaky ReLU | 0.4000 | 0.8 * 0.4000 = 0.3200  |
| Tanh       | 0.3799 | 0.8 * 0.3799 = 0.3039  |
| ELU        | 0.4000 | 0.8 * 0.4000 = 0.3200  |

---

## Step 4 — Sigmoid Activation at Output Layer

    Formula :  y_hat  =  Sigmoid(y_out)  =  1 / (1 + e^(-y_out))

    ReLU path:
      y_hat  =  1 / (1 + e^(-0.3200))  =  1 / (1 + 0.7261)  =  1 / 1.7261  =  0.5793

    Leaky ReLU path:
      y_hat  =  1 / (1 + e^(-0.3200))  =  0.5793  (same as ReLU, z was identical)

    Tanh path:
      y_hat  =  1 / (1 + e^(-0.3039))  =  1 / (1 + 0.7380)  =  1 / 1.7380  =  0.5754

    ELU path:
      y_hat  =  1 / (1 + e^(-0.3200))  =  0.5793  (same as ReLU, z was identical)

---

## Step 5 — Loss Calculation

    Formula :  L  =  -[ y * log(y_hat)  +  (1-y) * log(1 - y_hat) ]

    Since true label y = 1:
    L  =  -[ 1 * log(y_hat) ]  =  -log(y_hat)

| Activation | y_hat  | L = -log(y_hat) |
|------------|--------|-----------------|
| ReLU       | 0.5793 | 0.5456          |
| Leaky ReLU | 0.5793 | 0.5456          |
| Tanh       | 0.5754 | 0.5524          |
| ELU        | 0.5793 | 0.5456          |

---

## Full End-to-End Summary Table

| Step                    | ReLU   | Leaky ReLU | Tanh   | ELU    |
|-------------------------|--------|------------|--------|--------|
| y  (weighted sum)       | 0.4000 | 0.4000     | 0.4000 | 0.4000 |
| z  (after activation)   | 0.4000 | 0.4000     | 0.3799 | 0.4000 |
| y_out = w4*z + b2       | 0.3200 | 0.3200     | 0.3039 | 0.3200 |
| y_hat = Sigmoid(y_out)  | 0.5793 | 0.5793     | 0.5754 | 0.5793 |
| Loss L = -log(y_hat)    | 0.5456 | 0.5456     | 0.5524 | 0.5456 |

---

## Key Takeaways

    1. The weighted sum y = 0.40 is identical for all activations.
       The activation function only changes what happens AFTER that.

    2. ReLU, Leaky ReLU, and ELU all output z = 0.40 here because y is positive.
       Their real difference only shows when y is NEGATIVE.

    3. Tanh compresses z to 0.3799 instead of 0.40.
       Slightly lower final prediction (0.5754 vs 0.5793).
       But Tanh is zero-centered which makes weight updates more stable
       in shallow networks.

    4. The real difference between activations appears on NEGATIVE inputs.
       Run the same example with y = -0.55 to see:

         ReLU        →  z = 0          gradient = 0      (learns nothing)
         Leaky ReLU  →  z = -0.0055    gradient = 0.01   (barely learns)
         ELU         →  z = -0.4231    gradient = 0.5769 (learns well)
         Tanh        →  z = -0.5028    gradient = 0.7172 (learns well)

---

## When to Use Which Activation

    Layer type              Activation      Reason
    ─────────────────────────────────────────────────────────────────────────
    Hidden layer (default)  ReLU            Fast, simple, works in most cases
    Hidden layer (negative) Leaky ReLU      No dead neurons, gradient stays alive
    Hidden layer (research) ELU or PReLU    Better negative gradients than Leaky
    Hidden layer (shallow)  Tanh            Zero-centered, stable updates
    Output (binary class)   Sigmoid         Outputs probability between 0 and 1
    Output (multi-class)    Softmax         All class probabilities sum to 1
    Output (regression)     None / Linear   Predicts a raw continuous number

---

## Softmax — Output Layer for Multi-Class

> Softmax is NOT for hidden layers.
> Use it ONLY at the output layer when predicting 3 or more classes.

    Formula :  softmax(zi)  =  e^zi / (e^z1 + e^z2 + ... + e^zK)

    Example — 3 class output scores:
      z1 = 0.40,   z2 = 1.20,   z3 = -0.30

      e^0.40  = 1.4918
      e^1.20  = 3.3201
      e^-0.30 = 0.7408
      Sum     = 5.5527

      P(class 1)  =  1.4918 / 5.5527  =  0.2687
      P(class 2)  =  3.3201 / 5.5527  =  0.5979   <- highest = predicted class
      P(class 3)  =  0.7408 / 5.5527  =  0.1334
      Total       =  0.2687 + 0.5979 + 0.1334  =  1.0000  <- always sums to 1

    Loss — Categorical Cross-Entropy  (true label = class 2):
      L  =  -sum( yj * log(y_hat_j) )
      L  =  -[ 0*log(0.2687)  +  1*log(0.5979)  +  0*log(0.1334) ]
      L  =  -log(0.5979)
      L  =  0.5148