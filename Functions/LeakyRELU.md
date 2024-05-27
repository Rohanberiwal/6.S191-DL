# Leaky ReLU

## What is ReLU?

ReLU, or Rectified Linear Unit, is an activation function commonly used in artificial neural networks. It introduces non-linearity to the network by outputting the input directly if it is positive, and outputting zero otherwise. The mathematical expression for ReLU is:

\[ f(x) = \max(0, x) \]

## What is Leaky ReLU?

Leaky ReLU is a variation of the ReLU activation function. While ReLU sets all negative values to zero, Leaky ReLU allows a small, positive gradient for negative input values, instead of zero. The mathematical expression for Leaky ReLU is:

$$ f(x) = \begin{cases}
    x & \text{if } x \geq 0 \\
    \alpha x & \text{if } x < 0
\end{cases} $$



Here, $$\( \alpha \) is a small constant, typically a small fraction like 0.01.

## Advantages of Leaky ReLU:

- **Solves "dying ReLU" problem:** In regular ReLU, neurons that output zero for all positive inputs are considered "dead" and do not contribute to the learning process. Leaky ReLU helps mitigate this issue by allowing a small gradient for negative inputs, preventing neurons from becoming inactive.
- **Preserves negative values:** Unlike ReLU, Leaky ReLU does not entirely set negative values to zero, which can be beneficial in certain cases where preserving negative information is important.

## Disadvantages of Leaky ReLU:

- **Adds complexity:** Introducing a parameter like \( \alpha \) adds complexity to the model, which can make training more challenging.
- **Choice of \( \alpha \):** The choice of the leakage coefficient \( \alpha \) is empirical and can affect model performance. It needs to be carefully tuned.

## Usage:

- Leaky ReLU is widely used in deep learning models, particularly when the "dying ReLU" problem is observed.
- It is often used as an alternative to ReLU in convolutional neural networks (CNNs) and other deep learning architectures.

## Comparison with Other Activation Functions:

- Leaky ReLU is one of several variations of the ReLU activation function, including Parametric ReLU (PReLU), Exponential Linear Unit (ELU), and Randomized ReLU (RReLU).
- Each variation has its advantages and disadvantages, and the choice often depends on empirical performance and the specific characteristics of the dataset and model architecture.

In summary, Leaky ReLU addresses the "dying ReLU" problem by allowing a small, non-zero gradient for negative inputs, which helps improve the robustness and training efficiency of deep neural networks.
