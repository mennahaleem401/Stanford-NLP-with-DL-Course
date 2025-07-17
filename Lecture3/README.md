
# ⚡ Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

| Activation     | Formula                         | Derivative            | Notes                                   |
| -------------- | ------------------------------- | --------------------- | --------------------------------------- |
| **Sigmoid**    | σ(x) = 1 / (1 + e⁻ˣ)            | σ(x)(1 - σ(x))        | Squashes input to (0, 1); can saturate  |
| **Tanh**       | tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | 1 - tanh²(x)          | Output in (-1, 1); zero-centered        |
| **ReLU**       | max(0, x)                       | 1 if x > 0 else 0     | Efficient; sparse activation            |
| **Leaky ReLU** | max(αx, x), α ≈ 0.01            | 1 if x > 0 else α     | Prevents dead neurons                   |
| **Softmax**    | eˣⁱ / Σeˣʲ                      | Gradient more complex | Used in output layer for classification |

---

## 🔁 Forward Pass

The forward pass computes the output of the network given an input.

For a layer:

```
z = W·x + b
a = activation(z)
```

---

## 🔙 Backward Pass (Backpropagation)

Backpropagation computes the gradient of the loss with respect to each parameter using the chain rule.

Chain Rule:

```
∂L/∂x = ∂L/∂z · ∂z/∂x
```

Backprop through a layer:

```
Given:
z = W·x + b
a = activation(z)
Loss = L(a)

Then:
∂L/∂z = ∂L/∂a · activation'(z)
∂L/∂W = ∂L/∂z · xᵀ
∂L/∂x = Wᵀ · ∂L/∂z
∂L/∂b = ∂L/∂z
```

---

## 🔗 Computational Graph

Neural networks can be seen as computational graphs (DAGs) where:

* **Nodes** = operations
* **Edges** = data (tensors) flowing between operations

### Why it's useful:

* Enables **automatic differentiation**
* Tracks all operations during the forward pass
* Enables efficient backward gradient computation

---

## 🌊 Gradient Flow Through Common Operations

| Operation                      | Gradient Behavior                        |
| ------------------------------ | ---------------------------------------- |
| **Addition:** z = x + y        | ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z             |
| **Multiplication:** z = x \* y | ∂L/∂x = ∂L/∂z · y, ∂L/∂y = ∂L/∂z · x     |
| **Max(x, y)**                  | Pass gradient to max input; other gets 0 |

---

## 🧠 Efficient Backpropagation

Key ideas:

1. **Cache forward values** for reuse during backward pass.
2. **Re-use gradients** when possible to avoid recomputation.

---

## 🔄 General Backpropagation Algorithm

1. **Topological sort**: Visit nodes in order from input to output.
2. **Forward pass**: Compute outputs of each node.
3. **Backward pass**:

   * Start with ∂L/∂output = 1
   * Use chain rule to compute gradients backward

🧠 Cost: Same order of complexity as the forward pass.

---

## ⚙️ Partial Derivatives in Neural Networks

For a layer:

```
z = W·x + b
```

* ∂z/∂x = W
* ∂z/∂W = xᵀ
* ∂z/∂b = I (identity matrix)

---

## 🔲 Jacobians and Shapes

* The Jacobian of a vector-valued function is a matrix of all partial derivatives.
* For a scalar loss `S` and weight matrix `W`:

  ```
  ∂S/∂W is shaped like W
  ```

---

## 🧮 Outer Product Gradient (for weights)

For a layer:

```
z = W·x + b
```

If:

* `Δ` = ∂L/∂z (upstream gradient)
* `x` is the input

Then:

```
∂L/∂W = Δᵀ · xᵀ
```

This is an **outer product** of the two vectors, resulting in a matrix of shape `W`.

---

## 🤖 Automatic Differentiation (Autodiff)

Frameworks like PyTorch, TensorFlow use **autodiff** to compute gradients automatically.

How it works:

1. Each operation knows how to compute its **local gradient**.
2. The framework constructs a **computation graph** during forward pass.
3. On `.backward()`, it automatically applies the **chain rule** through the graph.

---

## ✅ Manual Gradient Checking

Used to validate your backpropagation implementation.

### Finite Difference Method:

```
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
```

Steps:

1. Choose small h = 1e-4
2. Compare to backprop result
3. If the difference is < 1e-2, you're usually safe.

⛔ Very slow (not used in training), but essential for debugging.

---

## ✅ Summary Table

| Concept                 | Description                               |
| ----------------------- | ----------------------------------------- |
| **Activation**          | Introduces non-linearity                  |
| **Forward Pass**        | Compute network output                    |
| **Backward Pass**       | Compute gradients using chain rule        |
| **Gradient Flow**       | Follows dependencies in computation graph |
| **Partial Derivatives** | Local derivatives needed for chain rule   |
| **Jacobian**            | Matrix of partial derivatives             |
| **AutoDiff**            | Automates gradient computation            |
| **Manual Checking**     | Finite difference to validate gradients   |


