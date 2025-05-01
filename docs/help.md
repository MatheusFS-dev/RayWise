# Description

This document is a concise compilation of tips, theoretical explanations, and practical guidance for regularization techniques and multi-objective optimization—especially using Optuna—in machine learning workflows. It covers when and why to use different regularizers, their benefits and drawbacks, and how to set up and interpret multi-objective studies in Optuna.

# Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
  - [Regularizers](#regularizers)
  - [Multi-objective Optimization](#multi-objective-optimization)
    - [How Optuna handles multi-objective studies](#how-optuna-handles-multi-objective-studies)

---

## Regularizers

| Regularizer             | What It Does                                                                                              | When to Use                                                                                   | Benefits                                                                                                  | Disadvantages                                                              |
|-------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **L1**                  | Adds the sum of the absolute values of weights (‖w‖₁) to the loss, encouraging many weights to become zero.  | Use when sparsity is desired, especially in high-dimensional settings with many irrelevant features.  | Promotes sparsity, acts as feature selection, and can lead to more interpretable models.                 | Non-smooth gradients at zero; may cause optimization instability.          |
| **L2**                  | Adds the sum of squared weights (‖w‖₂²) to the loss, discouraging large weights via quadratic penalization.  | Common default in neural networks to control model complexity and ensure smooth optimization.      | Provides smooth gradients, improves generalization by keeping weights small, and is computationally efficient. | Does not yield sparse solutions; models remain dense.                      |
| **L1L2 (Elastic Net)**  | Combines L1 and L2 penalties to balance sparsity and weight decay.                                         | Use when you need both sparsity and stability, particularly with correlated features.             | Balances feature selection and smooth optimization; hyperparameters allow flexible tuning.               | Increases complexity in hyperparameter tuning; requires balancing two penalties. |
| **Orthogonal**          | Adds a penalty that encourages weight matrices to be orthogonal (penalizing the deviation of \\(W^T W\\) from the identity). | Ideal for deep or recurrent networks where diverse features and stable gradient flow are crucial. | Promotes diversity among neurons, reduces redundancy, and improves gradient flow in deep architectures.    | Computationally more expensive and adds extra hyperparameter tuning requirements.  |

---

## Multi-objective Optimization

Multi-objective optimization tackles problems where you must optimize two or more conflicting criteria at once—say, minimizing validation loss *and* model size. Unlike single-objective search, you don’t get one “best” trial but rather a **Pareto front**: the set of non-dominated solutions where improving one objective always degrades another.

### How Optuna handles multi-objective studies

1. **Define multiple directions.**  
   When you create the study, pass a list of directions—one per objective:
   ```python
   study = optuna.create_study(
       directions=["minimize", "minimize"]  # e.g., (loss, size) both to minimize
   )
   ```  

2. **Return a sequence of values.**  
   Your `objective` must return a tuple (or list) of floats, matching the number and order of `directions`. For example:  
   ```python
   def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> Sequence[float]:
       # Hyperparameters…
       model = build_model(trial)
       loss = train_and_evaluate(model)
       param_count = model.count_params()
       return loss, float(param_count)
   ```  
   Optuna automatically treats the first return as objective 1 and the second as objective 2.