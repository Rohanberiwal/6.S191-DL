# Categorical Cross-Entropy Loss

Categorical Cross-Entropy Loss is a commonly used loss function in multi-class classification tasks, where there are more than two classes. It measures the dissimilarity between the true class labels and the predicted class probabilities output by the model.

## Meaning and Calculation:

- **Definition:** Categorical Cross-Entropy Loss measures the cross-entropy loss between the true labels and the predicted probabilities output by the model.
  
- **Formula:** The formula for Categorical Cross-Entropy Loss is:
   L(y, \hat{y}= sum_{i=1}^{C} y_i \log(\hat{y}_i) \
  where \( C \) is the number of classes, \( y_i \) is the true probability of class \( i \), and \( \hat{y}_i \) is the predicted probability of class \( i \).
  
- **Interpretation:** The loss is minimized when the predicted probabilities approach the true probabilities of the classes.

## Properties:

- **Supported Tasks:** Categorical Cross-Entropy Loss is suitable for multi-class classification tasks where the output is categorical (e.g., "cat," "dog," "horse").
  
- **Softmax Activation:** It is commonly used in conjunction with the softmax activation function in the output layer of the model. The softmax function transforms the raw outputs (logits) into a probability distribution over the classes.

- **Gradient Descent:** Categorical Cross-Entropy Loss is differentiable, making it compatible with gradient-based optimization algorithms like stochastic gradient descent (SGD).

## Usage:

- **Loss Function:** Categorical Cross-Entropy Loss is typically used as the loss function during the training phase of the model. The goal is to minimize this loss function to improve the model's performance on the multi-class classification task.
  
- **Evaluation Metric:** It is also commonly used as an evaluation metric to assess the performance of the trained model on a multi-class classification task.

Categorical Cross-Entropy Loss is a fundamental component in multi-class classification tasks and is widely used in various applications, including image classification, natural language processing, and speech recognition.
