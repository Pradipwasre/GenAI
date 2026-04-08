# Recurrent Neural Networks (RNN) — Beginner Notes

> **Before reading this:** You already know what an ANN (Artificial Neural Network) is.  
> This guide builds on that knowledge and explains **why RNN was created** and **how it works**.

---

## 📌 Quick Recap: What was ANN?

In ANN:
- You give an **input** → it goes through hidden layers → you get an **output**
- Every input is treated as **independent** — it has NO memory of past inputs
- Works great for images, tabular data, etc.

**But ANN has a problem...**

---

## ❓ The Problem ANN Cannot Solve

Imagine you are reading this sentence:

> **"The food was not good."**

If you process each word **independently** (like ANN does), you might think "good" = positive sentiment.  
But the word **"not"** before it completely changes the meaning!

**ANN forgets the past. It cannot understand ORDER and CONTEXT.**

---

## Enter: Recurrent Neural Network (RNN)

RNN is designed to **remember previous inputs** while processing the current one.

It has a **memory loop**, the output from the previous step is fed back as input to the next step.

---

## 🔁 ANN vs RNN — Side by Side

```
ANN:
  Input → [Hidden Layer] → Output
  (No memory, each input independent)

RNN:
  Input(t=1) → [Hidden Layer] → Output(t=1)
                    ↓ (memory passed)
  Input(t=2) → [Hidden Layer] → Output(t=2)
                    ↓ (memory passed)
  Input(t=3) → [Hidden Layer] → Output(t=3)
                    ↓
               Final Output (e.g., Positive / Negative)
```

> 💡 The hidden layer in RNN **talks to itself** across time steps. That's the key difference!

---

## 🏗️ RNN Architecture : Simple Diagram

```
                  ┌─────────────────────────────────────┐
                  │           RNN Unrolled Over Time     │
                  └─────────────────────────────────────┘

Time Step:        t=1           t=2           t=3

Input:           "food"         "is"          "bad"
                   │             │              │
                   ▼             ▼              ▼
              ┌────────┐    ┌────────┐    ┌────────┐
              │ Hidden │───▶│ Hidden │───▶│ Hidden │
              │ Layer  │    │ Layer  │    │ Layer  │
              └────────┘    └────────┘    └────────┘
                                                │
                                                ▼
                                          Final Output
                                       (Negative Sentiment)
```

> The arrow between hidden layers = **memory being passed forward** (hidden state)

---

## 📐 The Math (Keep it Simple!)

At each time step `t`, the RNN computes:

```
o(t) = f( x(t) × W  +  o(t-1) × W' )
```

Where:
| Symbol    | Meaning                                         |
|-----------|-------------------------------------------------|
| `x(t)`    | Current word input (as a number vector)         |
| `o(t-1)`  | Output from the **previous** time step (memory) |
| `W`       | Weight matrix for input                         |
| `W'`      | Weight matrix for recurrent (memory) connection |
| `f()`     | Activation function (tanh, ReLU, sigmoid)       |
| `o(t)`    | Output at current time step                     |

**At the very end:**
```
Final Prediction = softmax( o(T) × W'' )
```
The `softmax` converts the last hidden output into **probabilities** (e.g., 80% Negative, 20% Positive).

---

## Simple Worked Example: Sentiment Analysis

**Sentence:** `"food is bad"`

| Time Step | Word Input | Memory from Before | Output (Hidden State) |
|-----------|------------|--------------------|-----------------------|
| t = 1     | "food"     | None (zero)        | o1 (food context)     |
| t = 2     | "is"       | o1                 | o2 (food + is)        |
| t = 3     | "bad"      | o2                 | o3 (food + is + bad)  |

 **Final:** `softmax(o3)` → **Negative Sentiment** 

> Each step **builds on the previous one**, just like how you read and understand a sentence!

---

##  Why Not Just Use ANN Here?

| Feature                        | ANN        | RNN                   |
|-------------------------------|-------------|------------------------|
| Handles sequences?             | No          | Yes                    |
| Remembers past inputs?         | No          | Yes                    |
| Word order matters?            | Ignored     | Preserved              |
| Good for text/speech/time data?| No          | Yes                    |
| Good for images/tabular data?  | Yes         | Not ideal              |

---

##  Real-World Use Cases of RNN

```
┌────────────────────────────────────────────────────────┐
│                  RNN USE CASES                         │
├────────────────────────┬───────────────────────────────┤
│   Chatbots             │  "What's the weather today?"  │
│                        │  RNN understands full context │
├────────────────────────┼───────────────────────────────┤
│   Translation          │  English → Hindi              │
│                        │  Word order matters a lot!    │
├────────────────────────┼───────────────────────────────┤
│   Speech Recognition   │  "OK Google..." / Siri        │
│                        │  Audio is a time sequence     │
├────────────────────────┼───────────────────────────────┤
│   Stock Prediction     │  Price today depends on       │
│                        │  yesterday's prices           │
├────────────────────────┼───────────────────────────────┤
│   Text Generation      │  Auto-complete, GPT-like bots │
│                        │  Next word depends on past    │
└────────────────────────┴───────────────────────────────┘
```

---

##  Key Components of RNN (Summary)

### 1. Input — Word Embedding Vectors
- Words can't be fed as raw text. They are converted to **number vectors**.
- Tools: Word2Vec, Bag of Words, TF-IDF
- Example: `"food"` → `[0.2, 0.8, 0.1, ...]`

### 2. Hidden Layer — The Memory
- Contains neurons (e.g., 100 neurons)
- Takes **current input + previous output** together
- Passes result to next time step

### 3. Activation Function — Non-linearity
- Applied at each hidden neuron
- Common: `tanh`, `ReLU`, `sigmoid`
- Helps the network learn complex patterns

### 4. Output Layer
- After the last time step, output goes through `softmax`
- Gives probabilities for each class (Positive / Negative / Neutral)

### 5. Weight Matrices
| Matrix | Role |
|--------|------|
| `W`    | Applied to current input |
| `W'`   | Applied to previous hidden state (memory) |
| `W''`  | Applied at output layer for final prediction |

> **Same weights are used at every time step** — the model doesn't learn different weights per step!

---

## RNN Forward Propagation — Full Flow Diagram

```
START
  │
  ▼
Initialize o(0) = [0, 0, 0, ...] ← zero vector (no memory yet)
  │
  ▼
For each word at time step t = 1, 2, 3, ..., T:
  │
  ├── Take input word x(t)  ──────────────────────────┐
  │                                                    │
  ├── Take previous output o(t-1)  ───────────────┐   │
  │                                               ▼   ▼
  │                               o(t) = f( x(t)×W + o(t-1)×W' )
  │
  ▼
After last word at t = T:
  │
  ▼
Final Prediction = softmax( o(T) × W'' )
  │
  ▼
Output: Class Label (e.g., Negative = 0, Positive = 1)
```

---

## Limitations of Basic RNN

> Even though RNN is powerful, it has some weaknesses:

- **Vanishing Gradient Problem:** When sentences are very long, the memory from early words "fades away" during training.
- **Slow to train** on very long sequences.

> **Solution:** Advanced versions like **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) fix these problems — we'll cover those later!

---

## Quick Summary — What You Learned

| Concept | In Simple Words |
|--------|----------------|
| Why RNN? | ANN forgets the past; RNN remembers it |
| Hidden State | RNN's memory passed from step to step |
| Time Steps | Processing one word at a time, in order |
| Forward Propagation | Compute output step-by-step using formula |
| Softmax | Converts last output to a final prediction |
| Weight Sharing | Same W, W', W'' used at every time step |
| Use Cases | Chatbots, translation, speech, stock prediction |

---

## One-Line Definitions to Remember

- **RNN** = A neural network with a loop that gives it memory
- **Hidden State** = The "memory" passed from one time step to the next
- **Time Step** = One moment of processing (one word in a sentence)
- **Embedding** = Converting words into number vectors
- **Softmax** = Turns numbers into percentages (probabilities)
- **Activation Function** = Adds non-linearity (tanh, ReLU, sigmoid)

---