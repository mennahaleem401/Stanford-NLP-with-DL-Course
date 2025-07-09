
## 🧠 Stanford NLP with Deep Learning - Word Embeddings (GloVe)

### 🔹 تحميل نموذج GloVe

```python
import gensim.downloader as api

def load_embedding_model():
    """Load GloVe Vectors (200-dimensional)"""
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(wv_from_bin.index_to_key))
    return wv_from_bin

wv_from_bin = load_embedding_model()
```

---

### 🔹 تكوين المصفوفة من الكلمات المطلوبة

```python
import numpy as np
import random

def get_matrix_of_vectors(wv_from_bin, required_words):
    words = list(wv_from_bin.index_to_key)
    random.seed(225)
    random.shuffle(words)

    word2ind = {}
    M = []
    curInd = 0

    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue

    for w in required_words:
        if w not in words:
            try:
                M.append(wv_from_bin.get_vector(w))
                word2ind[w] = curInd
                curInd += 1
            except KeyError:
                continue

    M = np.stack(M)
    return M, word2ind
```

---

### 🔹 تقليل الأبعاد للتمثيل البصري

```python
from sklearn.decomposition import PCA

def reduce_to_k_dim(M, k=2):
    pca = PCA(n_components=k)
    M_reduced = pca.fit_transform(M)
    return M_reduced

M, word2ind = get_matrix_of_vectors(wv_from_bin, words)
M_reduced = reduce_to_k_dim(M, k=2)

M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis]
```

---

### 🔹 حساب Cosine Similarity

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```

---

### 🔹 التمثيل العكسي للفكتور (Vector Inversion)

```python
neg_vec = -wv_from_bin.get_vector("happy")  # for example
```

---

### 🔹 حل Analogies

```python
import pprint
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'grandfather'], negative=['man']))
# Expected: grandmother
```

معادلة الـ analogy:

$$
\text{x} = \text{grandfather} - \text{man} + \text{woman}
$$

---

### 🔹 أمثلة على الأخطاء في analogies

#### مثال:

```python
pprint.pprint(wv_from_bin.most_similar(positive=['foot', 'glove'], negative=['hand']))
```

النتيجة كانت:

```python
[('45,000-square', 0.49), ('10,000-square', ...)]
```

**السبب:** لأن النموذج تعلم من بيانات تحتوي على استخدامات شائعة لكلمة "foot" كوحدة قياس وليس كجزء من الجسم فقط.

---

### 🔹 تجربة Bias في النموذج

```python
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'profession'], negative=['woman']))
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'profession'], negative=['man']))
```

**النتيجة:**
غالبًا الكلمات المرتبطة بـ "man" فيها وظائف ذات مكانة عالية، بينما المرتبطة بـ "woman" فيها وظائف تقليدية أكتر.

#### ✔️ شرح الانحياز (Bias)

Bias في word vectors ممكن يحصل بسبب:

* البيانات اللي تم التدريب عليها كانت مليانة بالتمييزات الموجودة في المجتمع (زي النصوص الإخبارية أو مقالات الإنترنت).

#### ✔️ طريقة تقليل الانحياز:

طريقة "Hard Debiasing" زي ما شرحتها Bolukbasi et al. (2016):

* بتحدد اتجاه الـ gender (مثلاً: he - she).
* بعدين بتحاول تزيل هذا الاتجاه من الكلمات اللي المفروض تكون محايدة.
