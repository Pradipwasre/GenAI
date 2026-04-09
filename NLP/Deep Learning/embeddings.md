# Word2Vec and Word Embeddings

> You already know ANN. This guide uses that knowledge to explain how words are converted into numbers before feeding into any neural network.

---

## First: Vector vs Embedding, What is the Difference?

### Vector
A vector is just a list of numbers. That is it.

```
[1, 0, 0, 0, 0]   <-- this is a vector
[0.25, 0.8, 0.1]  <-- this is also a vector
```

A vector has no meaning on its own. It is just a data structure.

### Embedding
An embedding is a vector that has been created in a smart way, so that the numbers actually carry meaning.

```
"cat"  --> [0.2, 0.9, 0.1]
"dog"  --> [0.3, 0.8, 0.2]   <-- close to cat (both are animals)
"car"  --> [0.9, 0.1, 0.7]   <-- far from cat
```

The numbers here are not random. They were learned by a neural network.

### One-Line Difference

| Term      | Simple Meaning                                      |
|-----------|-----------------------------------------------------|
| Vector    | Just a list of numbers (no guarantee of meaning)   |
| Embedding | A smart vector where similar things are closer together |

> Think of it this way: all embeddings are vectors, but not all vectors are embeddings.

---

## The Problem: Machines Cannot Read Words

Before anything: ANN, RNN, or any model,  you need to convert words into numbers.

A machine cannot process this:

```
"I love cricket"
```

It can only process this:

```
[0.2, 0.5, 0.1, 0.9, ...]
```

So the question is: how do we convert words to numbers?

---

## Old Methods and Their Problems

### Method 1: Bag of Words

Each word gets a slot. 1 if present, 0 if not.

```
Vocabulary: [I, love, hate, cricket, football]

"I love cricket"  --> [1, 1, 0, 1, 0]
"I hate football" --> [1, 0, 1, 0, 1]
```

Problem: vectors are huge and mostly zeros (sparse). No meaning captured.
"love" and "hate" look equally unrelated to "cricket".

### Method 2: TF-IDF

Gives importance score to each word based on frequency.

Problem: still sparse vectors. Still does not understand that "happy" and "joyful" mean the same thing.

### The Core Problem with Both

```
"happy"  --> [0, 1, 0, 0, 0, 0, ...]
"joyful" --> [0, 0, 0, 1, 0, 0, ...]
```

These two vectors look completely different, but the words mean the same thing.
The model has no idea they are related.

---

## Word2Vec — The Solution

Word2Vec was introduced by Google in 2013. It is a neural network that learns word meaning from context.

The idea is simple:

> Words that appear in similar sentences will have similar vectors.

### Simple Example

Consider these sentences:

```
"I drink tea in the morning."
"I drink coffee in the morning."
```

The words "tea" and "coffee" appear in the exact same context.
So Word2Vec will place them close together in vector space.

After training:

```
"tea"    --> [0.71, 0.32, 0.88]
"coffee" --> [0.69, 0.35, 0.85]   <-- very close to tea
"cricket"--> [0.12, 0.91, 0.21]   <-- far from tea
```

---

## How Word2Vec Works Internally (Using ANN)

Word2Vec is a shallow ANN — just one hidden layer.

```
INPUT LAYER         HIDDEN LAYER        OUTPUT LAYER
(context words)     (learns vectors)    (predicts target word)

  "I"    ──┐
           ├──> [hidden neurons] ──> "drink"
  "morning"──┘
```

### Training Process

```
Step 1: Take a sentence
        "I drink tea in the morning"

Step 2: Slide a window over it (window size = 2)
        Context: ["I", "tea"]  --> Target: "drink"
        Context: ["drink","in"]--> Target: "tea"

Step 3: Feed context words into ANN
        ANN tries to predict the target word

Step 4: Calculate loss (how wrong was the prediction)

Step 5: Backpropagation -- adjust weights

Step 6: After training, the weights = word vectors
```

The weights learned by the hidden layer become the word embeddings.

---

## Two Architectures of Word2Vec

### CBOW (Continuous Bag of Words)
- Give context words, predict the middle (target) word
- Faster to train

```
["I", "tea"] --> predict --> "drink"
```

### Skip-gram
- Give one word, predict surrounding context words
- Better for rare words

```
"drink" --> predict --> ["I", "tea"]
```

---

## The Famous Vector Math

Once trained, you can do arithmetic with words:

```
"king" - "man" + "woman" = "queen"
```

This works because the model learned that the difference between king and man is "royalty applied to male", so applying that same difference to "woman" gives "queen".

This is proof that the vectors carry real meaning.

---

## Dense vs Sparse — Why Word2Vec is Better

| Feature         | Bag of Words / TF-IDF   | Word2Vec               |
|-----------------|-------------------------|------------------------|
| Vector size     | Very large (sparse)     | Small, fixed (e.g. 300)|
| Zero values     | Mostly zeros            | Almost no zeros        |
| Captures meaning| No                      | Yes                    |
| Similar words   | Look completely different| Are close together     |
| Good for ANN/RNN| No                      | Yes                    |

---

## Key Parameters When Training Word2Vec

| Parameter      | What it Controls                                       |
|----------------|--------------------------------------------------------|
| Vector size    | How many numbers represent each word (e.g. 100, 300)  |
| Window size    | How many surrounding words to consider as context     |
| Training data  | More data = better, more meaningful vectors           |

---

## Pre-trained Word2Vec by Google

Google trained Word2Vec on 3 billion words (Google News).

- 3 million words covered
- Each word = 300-dimensional vector
- Ready to use without training yourself

You just load it and use it directly in your project.

---



## Connection to RNN

This is why Word2Vec is used with RNN for tasks like sentiment analysis:

```
Raw Text
   |
   v
Word2Vec (convert each word to a dense vector)
   |
   v
RNN (process vectors one by one, time step by time step)
   |
   v
Final Output (Positive / Negative)
```

TF-IDF or Bag of Words vectors are sparse and do not carry meaning, so they perform worse as RNN input. Word2Vec vectors are dense and meaningful, perfect for RNN.

---

## Summary

| Concept         | One-Line Meaning                                           |
|-----------------|------------------------------------------------------------|
| Vector          | A list of numbers                                          |
| Embedding       | A smart vector where meaning is encoded in the numbers     |
| Bag of Words    | Simple word counting, no meaning, sparse                   |
| TF-IDF          | Weighted word counts, no meaning, sparse                   |
| Word2Vec        | Neural network trained to give meaningful dense vectors    |
| CBOW            | Predict middle word from surrounding words                 |
| Skip-gram       | Predict surrounding words from middle word                 |
| Dense vector    | Most values are non-zero and meaningful                    |
| Sparse vector   | Mostly zeros, wasteful and weak                            |


---