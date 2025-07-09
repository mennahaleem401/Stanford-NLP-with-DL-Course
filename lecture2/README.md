**Word Embeddings and Neural NLP: Summary for Repo**

---

## 1. Gradient Descent in NLP Models

### ⭐ Problem:

Cost function $J(\theta)$ depends on all windows in the corpus (billions!)

* Calculating full gradient is **very expensive**
* Neural networks need **many updates**, so this approach is impractical

### ✅ Solution: **Stochastic Gradient Descent (SGD)**

* Instead of full gradient, **sample one window** at a time and update
* Faster and works well in practice

### Mini-batch SGD:

* Take small batches of windows
* Update weights per batch, not per entire corpus

---

## 2. Word2Vec: Architecture & Computation

### Two main types:

1. **Skip-gram (SG)**: Predict context words from center word
2. **CBOW (Continuous Bag of Words)**: Predict center word from context words

### Two vectors per word:

* Center word vector $v \in V$
* Context word vector $u \in U$

### Computation:

* Dot product $u^T v$ ➞ measures similarity
* Softmax over all vocabulary: expensive!

---

## 3. Negative Sampling (Word2Vec Optimization)

* Instead of softmax, do **binary classification**:

  * Predict if a (center, context) pair is **real** or **noise**
* Sample $k$ negative words
* Use **sigmoid function** instead of softmax

### Objective Function:

$J_{\text{neg-sample}} = -\log \sigma(u_o^T v_c) - \sum_{k \in K} \log \sigma(-u_k^T v_c)$

* Only a few vectors updated per step: **very sparse gradients**
* Efficient for large vocabulary

---

## 4. Co-occurrence Matrix & GloVe

### Motivation:

Why not just store how often each word appears with others?

### Matrix X:

* Each entry $X_{ij}$: how many times word $j$ appears near $i$
* Size: $V \times V$ (sparse but large)

### Dimensionality Reduction (e.g., SVD):

* Turn sparse co-occurrence matrix into **dense low-dimensional embeddings**
* Preprocessing: log counts, ignore function words, scale by distance

---

## 5. GloVe (Global Vectors for Word Representation)

### Key Insight:

Capture **ratios of co-occurrence probabilities** as **linear vector differences**:
$w_x \cdot (w_a - w_b) \approx \log \frac{P(x|a)}{P(x|b)}$

### GloVe Loss Function:


![image](https://github.com/user-attachments/assets/dc8e8bc8-90c2-49ae-b72b-20f43c00b842)


* $f(X_{ij})$: weighting function (usually increases then caps)
* Fast to train
* Scalable to large corpora

---

## 6. Evaluation of Word Vectors

### Intrinsic Evaluation:

* Task: Analogies (e.g., man\:woman :: king\:queen)
* Test: $\text{vec}(king) - \text{vec}(man) + \text{vec}(woman) \approx \text{vec}(queen)$
* Word similarity: Compare cosine similarity with human judgment datasets (e.g., WordSim353)

### Extrinsic Evaluation:

* Use word vectors in real tasks (e.g., NER, sentiment analysis)
* If model improves, embedding is helpful

---

## 7. Word Sense Ambiguity

### Problem:

* One word = multiple meanings ("pike": fish, weapon, road...)

### Solutions:

1. Train **multiple vectors per word** (cluster contexts)
2. Use linear combination of sense vectors:
   $v_{\text{pike}} = \alpha_1 v_1 + \alpha_2 v_2 + \dots$

* Weights $\alpha$ based on frequency of each sense

---

## 8. Neural Classification (NER Example)

### Task:

Label each word with a tag: PERSON, LOCATION, DATE, etc.

### Simple Approach:

* Input: window of word vectors
* Apply binary logistic classifier to center word:
  $s = u^T h \quad \text{where} \quad h = f(Wx + b)$
* Use sigmoid function for binary prediction (location or not)

### Loss Function:

* **Cross entropy**: penalizes incorrect predictions

---

## 9. Neural Computation

### Logistic Regression Unit:

$h_{w,b}(x) = f(w^T x + b)$

* Activation function $f$: sigmoid, ReLU, tanh

### Neural Network:

* Multiple units working in parallel
* Outputs of one layer = inputs to next layer
* Add non-linearities to allow complex function approximation

### Matrix Form:

$z = Wx + b \quad \text{then} \quad a = f(z)$

* Apply activation element-wise

---

## Summary Table: GloVe vs Word2Vec

| Feature            | GloVe                         | Word2Vec                   |     |                     |
| ------------------ | ----------------------------- | -------------------------- | --- | ------------------- |
| Type               | Count-based                   | Predictive                 |     |                     |
| Training Objective | Reconstruct co-occurrence log | Predict context or center  |     |                     |
| Optimization       | Weighted least squares        | SGD with negative sampling |     |                     |
| Interpretability   | Captures ratios (P(x          | a)/P(x                     | b)) | Harder to interpret |
| Scalability        | Very scalable                 | Also scalable              |     |                     |
| Evaluation         | Performs well                 | Performs well              |     |                     |

---

Prepared by: Menna Haleem
