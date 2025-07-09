---
# CS224N - Lecture 1: Word Meaning & Word Vectors

##  How Do We Represent the Meaning of a Word?

###  What is “meaning”

1. The idea represented by a word or phrase.
2. The idea someone wants to express using words or signs.
3. The concept communicated through writing or art.

So, **words are just symbols** — what matters is the **idea they point to**.

---

###  Signifier ⟷ Signified

A common linguistic model says:

* **Signifier** = the word (like "tree")
* **Signified** = the concept or object ()

This approach is called **denotational semantics** — it connects a word to its literal meaning.

Example:

```
“tree” → {branches, leaves, green, trunk...}
```

---

## How Do We Have Usable Meaning in a Computer?

### WordNet: A Traditional Approach

Before neural networks, NLP used resources like **WordNet**, which:

* Organizes words into **synonym sets** (words with similar meanings).
* Stores **hypernyms** (“is-a” relationships), e.g., a *panda* is a *mammal*.

#### Example – Synonyms of “good”:

```python
from nltk.corpus import wordnet as wn

for synset in wn.synsets("good"):
    print(synset.lemmas())
```

Output includes:

* `good, goodness`
* `honorable, respectable`
* `well, thoroughly`

#### Example – Hypernyms of “panda”:

```python
panda = wn.synset("panda.n.01")
panda.closure(lambda s: s.hypernyms())
```

This returns:

```
panda → carnivore → mammal → animal → organism → entity
```

---

## Problems with WordNet

1. **Missing nuance**:
   WordNet may list "proficient" as a synonym for "good", but that’s not always true.

2. **No connotation control**:
   It includes offensive or inappropriate synonyms without warnings.

3. **No slang or new meanings**:
   Words like "wizard" (meaning genius) or "bombest" are missing.

4. **Hard to update**:
   Requires human input and doesn’t scale.

5. **Can’t measure similarity well**:
   It’s not enough to compute how “close” two words are in meaning.

---

##  Words as Discrete Symbols (One-Hot Encoding)

In early NLP:

* Each word was treated as a **unique symbol**.
* Represented using a **one-hot vector** (all 0s, one 1).

Example:

```
"hotel" = [0 0 0 0 1 0 0 ...]
"motel" = [0 0 0 1 0 0 0 ...]
```

But this has a big problem:

> These vectors are **orthogonal** — they have **no sense of similarity** between words like "hotel" and "motel".

---

##  Why Discrete Representations Don’t Work

In tasks like **web search**, if someone types:

```
Seattle motel
```

You’d want to match documents that say:

```
Seattle hotel
```

But with one-hot vectors, they appear totally unrelated.
You need a better representation that **captures meaning**.

---

## Better Idea: Use Context (Distributional Semantics)

> “You shall know a word by the company it keeps.”
> — J.R. Firth (1957)

The meaning of a word can be inferred from:

* **Words that often appear around it** (its context).
* For example, if “banking” often appears near “regulation”, “money”, or “finance”, that tells us about its meaning.

This is called **distributional semantics**.

---

## Word Vectors

We can now **represent each word as a dense vector** of real numbers (not just 0s and 1s).
Words that appear in similar contexts get **similar vectors**.

 Also called:

* **Word embeddings**
* **Neural word representations**
* A **distributed representation** of meaning

Example:

```text
banking = [0.28, 0.79, -0.17, ..., 0.27]
monetary = [0.41, 0.58, -0.00, ..., 0.05]
```

These vectors can now be used to:

* Measure similarity
* Train models
* Feed into neural networks

---

##  Word2Vec: Introduction

Word2Vec is a method to **learn word vectors from data** (Mikolov et al., 2013).

###  Main idea:

1. Take a **large text corpus**.
2. For each word in a sentence (called **center word**), look at the words around it (called **context words**).
3. Try to **predict context words given the center word**.
4. **Adjust the vectors** to get better predictions.
5. maximize probability by keep updating word vectors.

This is called the **Skip-gram model**.

---

### Skip-gram Training Example

Given the sentence:

```
…problems turning into banking crises…
```

If the center word is `"banking"`, and the window size = 2, then the context words are:

```
["into", "turning", "crises", "problems"]
```

The model learns to **maximize** the probability of seeing these context words, given the center word.

---

##  Word2Vec: Objective Function

We want to **maximize** the probability of predicting the correct context words for each center word.

We define a **loss function** that computes this for the whole corpus.

Then we want to **minimize** this loss function.


![image](https://github.com/user-attachments/assets/c59972e2-30f1-4ac2-8110-e48f7426d94a)

---

##  How Is Probability Calculated?

For a center word `c` and a context word `o`, we compute:

$$
P(o | c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}
$$

Where:

* $v_c$: vector of the center word
* $u_o$: vector of the context word
* The denominator is a softmax over the whole vocabulary

This uses:

* **Dot product** to measure similarity between vectors
* **Softmax** to convert scores into probabilities

---

## Training the Model (Gradient Descent)

To **train** the word vectors, we:

* Start with random values
* Use **gradient descent** to adjust them
* Repeatedly update the vectors to **minimize the loss function**

---

## Optimization: Stochastic Gradient Descent (SGD)

The full loss function is calculated over **millions or billions of words**, which is too slow.

Instead, we use **Stochastic Gradient Descent (SGD)**:

* Sample small batches (or even one example at a time)
* Update the vectors more frequently
* Much faster and more scalable

---
