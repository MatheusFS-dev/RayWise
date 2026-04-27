# Description

This document is a concise compilation of tips, theoretical explanations, and practical guidance aiding developers in setting up and managing their development environments, optimizing workflows, and understanding key concepts in machine learning and software engineering.


# Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
  - [Useful links](#useful-links)
  - [Regularizers](#regularizers)
  - [Initializing neural networks](#initializing-neural-networks)
    - [Activation Function ⇄ Initializer Reference](#activation-function--initializer-reference)
  - [Initializing the bias](#initializing-the-bias)
    - [Bias Initializer](#bias-initializer)
    - [When to Use Which Bias Initializer](#when-to-use-which-bias-initializer)
  - [Vanishing \& Exploding Gradients](#vanishing--exploding-gradients)
  - [Multi-objective Optimization](#multi-objective-optimization)
    - [How Optuna handles multi-objective studies](#how-optuna-handles-multi-objective-studies)
  - [Multi-class classification vs multi-label classification](#multi-class-classification-vs-multi-label-classification)
  - [Fine-tuning a trained model with SGD](#fine-tuning-a-trained-model-with-sgd)
  - [About Batch Normalization in DNNs](#about-batch-normalization-in-dnns)
    - [When BN may be overkill or even harmful](#when-bn-may-be-overkill-or-even-harmful)
  - [CNN Parameters](#cnn-parameters)
  - [LSTM Parameters](#lstm-parameters)

---

## Useful links

- [DeepLearning.ai - AI Notes](https://www.deeplearning.ai/ai-notes/index.html) is a collection of notes on various topics in deep learning, including optimizers and initialization methods.


---

## Regularizers

| Regularizer             | What It Does                                                                                              | When to Use                                                                                   | Benefits                                                                                                  | Disadvantages                                                              |
|-------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **L1**                  | Adds the sum of the absolute values of weights (‖w‖₁) to the loss, encouraging many weights to become zero.  | Use when sparsity is desired, especially in high-dimensional settings with many irrelevant features.  | Promotes sparsity, acts as feature selection, and can lead to more interpretable models.                 | Non-smooth gradients at zero; may cause optimization instability.          |
| **L2**                  | Adds the sum of squared weights (‖w‖₂²) to the loss, discouraging large weights via quadratic penalization.  | Common default in neural networks to control model complexity and ensure smooth optimization.      | Provides smooth gradients, improves generalization by keeping weights small, and is computationally efficient. | Does not yield sparse solutions; models remain dense.                      |
| **L1L2 (Elastic Net)**  | Combines L1 and L2 penalties to balance sparsity and weight decay.                                         | Use when you need both sparsity and stability, particularly with correlated features.             | Balances feature selection and smooth optimization; hyperparameters allow flexible tuning.               | Increases complexity in hyperparameter tuning; requires balancing two penalties. |
| **Orthogonal**          | Adds a penalty that encourages weight matrices to be orthogonal (penalizing the deviation of \\(W^T W\\) from the identity). | Ideal for deep or recurrent networks where diverse features and stable gradient flow are crucial. | Promotes diversity among neurons, reduces redundancy, and improves gradient flow in deep architectures.    | Computationally more expensive and adds extra hyperparameter tuning requirements.  |

Regularization becomes critical when your model’s capacity (number of parameters) starts to greatly exceed the information content of your data (number of training examples). For example, considering a DNN structure:

| Network / Data Regime                                   | Recommendation                                                                                                                                                                                             |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tiny net**<br/>(< 5 layers, < 100 K parameters)       | • **Skip heavy weight decay**: L2 coefficients (λ) ≈ 0 or very small (1e-6)…1e-5<br/>• **Light dropout**: 0–0.1 only if you see overfitting<br/>• Data ≥ 10× parameters → minimal reg needed               |
| **Small to medium**<br/>(5–10 layers, 100 K–1 M params) | • **L2 weight decay**: λ≈1e-4…1e-3 to combat moderate overfitting<br/>• **Dropout**: 0.1–0.3 on hidden layers<br/>• **Optional L1** (λ≈1e-5) if you need feature sparsity                     |
| **Aggressive regime**<br/>(8–10 layers, > 1 M params)   | • **L2 weight decay**: λ≈1e-5…1e-4 when paired with AdamW-style decay<br/>• **Higher dropout**: 0.3–0.5<br/>• Consider **L1/L2** mix (e.g. L1L2) for both sparsity and smooth weights |
| **Small dataset**<br/>(< 10 K examples)                 | • **Always use regularization**: parameter-to-sample ratio > 1 → strong reg (L2 λ≈1e-3…1e-2)<br/>• **Dropout**: 0.2–0.5<br/>• Early stopping becomes crucial                                               |
| **Large dataset**<br/>(> 100 K examples)                | • You can **dial back** weight decay (λ≈1e-6…1e-4) and dropout (0–0.2)<br/>• Model can “eat” parameters if data supports it—overfitting risk lower                                                         |


* **Capacity vs. data**: If #parameters ≫ #examples, the model can memorize; regularizers (L2/L1/dropout) inject bias/noise to force learning meaningful patterns.
* **Dropout** is a stochastic regularizer: stronger dropout (0.3–0.5) for larger nets or smaller datasets; lighter dropout (0–0.2) when you suspect under-regularization only in final layers.
* **Weight decay (L2)** smooths weight values and penalizes large weights; safer default for most medium-sized DNNs.
* **L1** encourages sparsity—useful if you believe only a subset of features matters or to compress very wide layers.

---

## Initializing neural networks

Proper initialization of network parameters is foundational to stable and efficient training. Without it, practitioners commonly encounter:

* **Symmetry and Redundant Features**
  Initializing all weights to the same constant (e.g. zeros) causes neurons in a layer to produce identical outputs and gradients—effectively collapsing model capacity and preventing diverse feature learning. Randomized schemes break this symmetry, enabling each neuron to learn distinct representations.

* **Vanishing & Exploding Gradients - Check section [Vanishing \& Exploding Gradients](#vanishing--exploding-gradients)**
  If weights are too small, gradients shrink exponentially as they back-propagate through layers, stalling learning; if too large, gradients explode, causing numerical instability and divergence. Initializers like Xavier/Glorot and He/Kaiming are explicitly designed to preserve the variance of activations and gradients across layers, mitigating these issues.

* **Slow Convergence & Poor Optima**
  Improperly scaled initial weights can confine the optimizer to narrow, ill-conditioned regions of the loss landscape, requiring excessively small learning rates or many epochs to converge—and often settling in suboptimal minima. Effective initialization accelerates convergence and increases the likelihood of finding better-performing solutions.

Failing to align initialization with network depth and nonlinearity typically forces extra efforts in tuning learning rates, adding normalization layers, or limiting architectural depth—complicating both experimentation and reproducibility.

> [!TIP]
> A good analysis of this topic can be found in [DeepLearning.AI](https://www.deeplearning.ai/ai-notes/initialization/index.html).


### Activation Function ⇄ Initializer Reference

| Activation Function    | Recommended Initializer              | Why it works                                                                                                 |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Sigmoid                | **GlorotUniform** / **GlorotNormal** | Balances input/output variance to avoid saturation-induced vanishing gradients in bounded activations       |
| Tanh                   | **GlorotUniform** / **GlorotNormal** | Same as Sigmoid but zero-centered, further improving gradient symmetry and flow                             |
| ReLU / Leaky ReLU      | **HeNormal** / **HeUniform**         | Scales variance by 2 to compensate for half of activations being zero, keeping signal magnitude stable      |
| ELU                    | **HeNormal** / **HeUniform**         | Similar variance scaling as ReLU; handles negative saturation while preserving gradient dynamics            |
| SELU                   | **LecunNormal** / **LecunUniform**   | Designed for self-normalizing nets—maintains mean≈0 and var≈1 through each layer                            |
| GELU                   | **TruncatedNormal** (stddev=0.02)    | Small Gaussian spread matches assumptions in transformer-style networks, avoiding large activation variance |
| Softmax (output layer) | **GlorotUniform** / **GlorotNormal** | Ensures stable logit distributions before exponentiation, preventing extreme probability spikes             |
| Linear / Identity      | **VarianceScaling** (scale=1.0)      | Preserves unit variance through identity mapping, so signal neither explodes nor vanishes                   |
| RNNs / LSTMs           | **Orthogonal**                       | Maintains gradient flow in deep or recurrent networks.                     |

---

## Initializing the bias

**Bias**
In any parametric model—from linear regression to neural networks—the **bias** (often called the *intercept*) is an additional parameter that allows the model’s output to shift by a constant amount.  In a linear model $y = w^\top x + b$, the bias $b$ ensures the prediction need not pass through the origin, enabling the model to fit data with nonzero mean or required offsets.  More broadly, this constant term translates decision boundaries or activation functions, expanding the family of functions the model can represent

### Bias Initializer

A **bias initializer** specifies the starting values of $b$ before training.  Common strategies:

* **Zeros**
  All biases start at 0.  This is neutral and simple; symmetry is already broken by randomized weights.
* **Constant ($c$)**
  Every bias starts at a fixed small value $c$, e.g. 0.1.  Helps certain activations (like ReLU) to be “on” at initialization.
* **Data-driven Constant**
  Compute from the target distribution (e.g. log-odds of class priors or mean of regression targets) so that initial predictions are near the data’s central tendency.

Choosing an appropriate initializer can **speed convergence**, **stabilize gradients**, and **avoid dead neurons**.


### When to Use Which Bias Initializer

| **Scenario**                              | **Initializer**                  | **Rationale**                                                                                                                                   |
| ----------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Most hidden layers**<br>(tanh, sigmoid) | Zeros                            | Neutral shift; random weights already break symmetry.                                                                                           |
| **Hidden layers with ReLU/Leaky ReLU**    | Small positive constant (0.01)   | Ensures units receive slightly positive input initially, keeping them active and preserving gradient flow (“dying ReLU” mitigation).       |
| **Binary classification (sigmoid)**       | Constant = log-odds of positives | Offsets model’s initial output to reflect class imbalance; reduces early loss spikes and speeds convergence in imbalanced settings.             |
| **Multi-class classification (softmax)**  | Constant per class = log $p_i$   | Sets each logit bias to the log of its class prior $p_i$; yields initial softmax outputs close to empirical frequencies, stabilizing training.  |
| **Regression (linear output)**            | Constant = target mean           | Centers initial predictions on the data’s mean; network only needs to learn deviations, accelerating convergence on offset-fitting.             |
| **Recurrent forget gates (LSTM/GRU)**     | Constant = 1 (or >0)             | Encourages the network to retain information at start by biasing the forget gate open, improving short-term memory retention early in training. |


## Vanishing & Exploding Gradients

Back-propagation relies on the chain rule, so the gradient at any given layer is a product of many Jacobian matrices. When these matrices have singular values consistently **< 1**, gradients **shrink exponentially** as they propagate toward earlier layers—ultimately becoming so small that those layers learn virtually nothing (vanishing gradients). Conversely, if singular values are **> 1**, gradients **grow exponentially**, leading to numerical overflow or wildly oscillating updates (exploding gradients).

* **Vanishing gradients** stall learning in deep networks: only top layers adapt, and lower layers remain near their initial values—limiting model expressivity and hindering convergence.
* **Exploding gradients** cause unstable training: parameter updates become erratic, model loss may diverge, and gradients can overflow to NaN. Common mitigations include gradient clipping and reduced learning rates, but these address symptoms rather than the root cause.
* **Role of initialization:** If initial weights are poorly scaled—too small or too large—their singular values deviate substantially from 1, directly exacerbating gradient decay or growth. Initialization schemes such as Glorot and He are explicitly designed to keep the scale of activations and gradients roughly constant across layers, thereby preemptively countering vanishing/exploding dynamics.

Even with normalization layers or residual connections, a well-chosen initializer remains crucial: it reduces the need for extreme hyperparameter tuning and promotes faster, more reliable convergence across architectures.

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

---

## Multi-class classification vs multi-label classification

- **Multi-class classification**: Each instance belongs to one and only one class. For example, classifying images of animals into categories like "cat," "dog," or "bird." The model outputs a single label for each instance.
- **Multi-label classification**: Each instance can belong to multiple classes simultaneously. For example, tagging an image with multiple labels like "cat," "cute," and "pet." The model outputs a set of labels for each instance.
- **Key Differences**:
  - **Output Layer**: Multi-class uses softmax for a single label, while multi-label uses sigmoid for independent probabilities.
  - **Loss Function**: Multi-class typically uses categorical cross-entropy, while multi-label uses binary cross-entropy.
  - **Evaluation Metrics**: Multi-class often uses accuracy or F1-score, while multi-label may use hamming loss or subset accuracy.
- **Use Cases**: Multi-class is for exclusive categories, while multi-label is for overlapping categories.

---

## Fine-tuning a trained model with SGD

Usually SGD does not work well with the first version of the model, but it could help to fine-tune a trained model. The SGD is very noisy and can help to escape local minima.


---

## About Batch Normalization in DNNs

Batch normalization (BN) remains a valuable tool for deep feed-forward nets—rarely “redundant” in practice—but its benefit depends on depth, batch size, and data preprocessing:

* **Accelerated convergence & smoother optimization**
  BN standardizes each layer’s inputs (zero mean, unit variance) per mini-batch, which was originally motivated as reducing “internal covariate shift” but is now understood to **smooth the loss landscape**, yielding more stable, predictable gradients and allowing higher learning rates.

* **Implicit regularization**
  The noise in batch statistics injects stochasticity, which often **improves generalization** and can reduce reliance on dropout.

* **Stability in deeper nets**
  In networks beyond ∼5–10 layers, BN **mitigates vanishing/exploding gradients** and sensitivity to initialization, making very deep stacks trainable.

### When BN may be overkill or even harmful

| Scenario                              | Why BN adds little or hurts                                        | Alternative                                       |
| ------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| **Shallow nets (≤ 5 layers)**         | Few internal transforms → small covariate-shift issues             | Simple input standardization (zero-mean/unit-var) |
| **Small/variable batch sizes (< 16)** | Noisy mean/var estimates degrade performance ([Stack Overflow][3]) | LayerNorm or GroupNorm (batch-independent)        |
| **Ultra-low-latency inference**       | Maintains running-mean/var overhead; two‐mode logic                | Remove BN; rely on fixed preprocessing            |


* **Use BN** if you train with batches ≥ 32 and your network is deeper than ∼5 layers—it will speed convergence and add mild regularization.
* **Skip BN** (or switch to LayerNorm) if your batches are very small/irregular, or if you only have a shallow (≤ 5) stack on already normalized inputs.

---

## CNN Parameters

Below is a unified, dimension-agnostic breakdown of every core argument in ConvND layers (Conv1D/2D/3D), what it does, how it impacts the layer’s behavior, when to adjust it, and typical value choices.

* **`kernel_size` (int or tuple of ints)**

  > Size of the convolution window along each spatial/temporal dimension.

  * **Effect:** Larger kernels capture broader context; smaller ones focus on local detail.
  * **When to use:** Small (3×3 or 3) for fine features; larger (5–7) when you need more context per layer.
  * **Typical:** 3 or 5 in most architectures.

* **`strides` (int or tuple of ints, default 1)**

  > Step size of the convolution movement.

  * **Effect:** Stride > 1 downsamples feature maps (reduces resolution).
  * **When to use:** To reduce spatial/temporal size without separate pooling; e.g. stride 2 for halving.
  * **Constraint:** Don’t combine strides > 1 with dilation\_rate > 1.
  * **Typical:** 1 for feature extraction; 2 for occasional downsampling.

* **`padding` (`"valid"`, `"same"`, `"causal"`, default `"valid"`)**

  > How borders are handled.

  * **Effect:**

    * `valid`: no padding → shrinks output.
    * `same`: pads so output size ≈ input size.
    * `causal`: only for sequence models (no future leakage).
  * **When to use:**

    * `same` to preserve dimensions.
    * `valid` to strictly convolve without padding.
    * `causal` in autoregressive/time-series tasks.

* **`data_format` (`"channels_last"` or `"channels_first"`)**

  > Dimension ordering of input/output.

  * **Effect:** Slight performance/memory differences.
  * **When to use:**

    * `"channels_last"` is TensorFlow default.
    * `"channels_first"` if migrating code or optimizing for certain backends.

* **`dilation_rate` (int or tuple of ints, default 1)**

  > Spacing between kernel taps (“holes”) for dilated convolution.

  * **Effect:** Enlarges receptive field without extra parameters.
  * **When to use:** Stacked dilations (2, 4, 8, …) in sequence or image modeling to cover large context.

* **`groups` (int, default 1)**

  > Number of channel groups for grouped convolution.

  * **Effect:** Splits input channels into groups, each convolved separately, then concatenated.
  * **When to use:**

    * `1` for standard convolution.
    * `channels` (i.e. one per input channel) for depthwise conv.
    * Intermediate values in ResNeXt-style blocks.
  * **Constraint:** `filters` % `groups` == 0 and `input_channels` % `groups` == 0.


---

## LSTM Parameters

Below is a topic-by-topic deep dive into every `LSTM` constructor argument, detailing its effect, guidance on when to tweak it, and suggested value ranges.

* **`activation`**

  **Definition & effect:**
  Element-wise nonlinearity for the candidate cell state (default: `tanh`). Shapes how new inputs are integrated and scaled.

  **When to adjust & suggested values:**

  * **Default (`tanh`)** suits most sequence tasks—keeps outputs in (–1,1).
  * **`relu` or variants** may accelerate convergence but risk “dead” cells and gradient explosion; use only if you have very large datasets and stable normalization.
  * **`None` (linear)** rarely used, only in custom RNN variants.

* **`recurrent_activation`**

  **Definition & effect:**
  Activation for the gating functions (default: `sigmoid`), controlling information flow from the previous hidden state.

  **When to adjust & suggested values:**

  * **Default (`sigmoid`)** provides smooth gating.
  * **`hard_sigmoid`** can speed up GPU kernels at a small accuracy trade-off—use in latency-critical deployments.
  * **`None` (linear gates)** effectively disables gating; generally not recommended.

> [!NOTE]
> An LSTM’s **`activation`** (default `tanh`) is applied to the candidate cell state and to the final output, squashing values into (–1, 1) to regulate how new information is written into and read from the cell. In contrast, **`recurrent_activation`** (default `sigmoid`) is applied inside the gating functions—input, forget, and output gates—mapping gate pre-activations into (0, 1) so the network can decide what to retain, forget, or emit at each time step.

* **`unit_forget_bias`**

  **Definition & effect:**
  When `True`, initializes the forget-gate bias to +1 (and forces `bias_initializer="zeros"`), biasing the LSTM to **remember** at start of training.

  **When to adjust:**

  * **Keep `True`** in virtually all cases—stabilizes learning on long sequences.
  * **Set `False`** only when you have a very specific gating scheme or want to learn forget-bias from scratch (rare).

  > *Empirical support:* Gers et al. (2000) and Jozefowicz et al. (2015) demonstrated that positive forget-gate bias improved gradient flow and speed of convergence.


* **`use_bias`, `kernel_initializer`, `recurrent_initializer`, `bias_initializer`**

  **Definition & effect:**

  * **`use_bias`** toggles learnable bias terms per gate.
  * **`kernel_initializer`** (default `glorot_uniform`) seeds input-to-hidden weights.
  * **`recurrent_initializer`** (default `orthogonal`) seeds hidden-to-hidden weights, preserving gradient norms over time.
  * **`bias_initializer`** (default `zeros`, or “zeros with +1 on forget gate” when `unit_forget_bias=True`). 

  **When to adjust:**

  * **Initializers**: stick with defaults unless using very deep or custom RNN cells; e.g. use He initialization (`he_normal`) if you switch `activation` to `relu`.
  * **Disable `use_bias`** only if you want a purely zero-centered linear mapping (rare).


* **Dropout** & **`recurrent_dropout`**
  
  **Definition & effect:**

  * **`dropout`**: fraction of inputs to drop at each time step.
  * **`recurrent_dropout`**: fraction of recurrent state connections to drop.
    Both regularize against co-adaptation but can destabilize long-term memory if overused.

  **When & how to use:**

  * **Language models & translation**: Zaremba et al. (2014) applied 0.5 dropout on inputs/outputs and 0.25 on recurrent connections to achieve state-of-the-art perplexity on Penn Treebank.
  * **General practice**: start with `dropout=0.1–0.3`, `recurrent_dropout=0.0–0.2`; monitor stability and switch off `recurrent_dropout` if using GPU-optimized kernels (common cuDNN restriction).
---

* **`return_sequences`**

  **Definition & effect:**
  When `True`, outputs the full hidden‐state sequence (shape `(batch, timesteps, units)`); when `False`, only the last output (`(batch, units)`).

  **When to use:**

  * **Sequence-to-sequence** or **stacked RNNs**: set `True` on intermediate layers so the next layer sees all time steps.
  * **Many-to-one tasks** (classification/regression on whole sequence): leave `False`.

* **`return_state`**

  **Definition & effect:**
  When `True`, also returns the final hidden and cell states (2 additional outputs) alongside the main output.

  **When to use:**

  * **Encoder–decoder architectures**: capture final states of encoder to feed into decoder.
  * **Stateful prediction loops**: manually step through time.

* **`go_backwards`**

  **Definition & effect:**
  If `True`, processes the input sequence in reverse order (but still returns outputs in forward time).

  **When to use:**

  * **Bidirectional RNNs**: wrap an LSTM with `Bidirectional` using one forward and one backward pass.
  * **Reverse context**: e.g. when future context is known.

* **`stateful`**

  **Definition & effect:**
  When `True`, the final states of each batch become the initial states of the next batch (requires fixed batch size).

  **When to use:**

  * **Streaming/online prediction**: continuing sequences longer than you can fit in one batch.
  * **Careful batch management**: you must manually reset states between independent sequences.

---

* **`unroll`**

  **Definition & effect:**
  When `True`, unrolls the RNN loop into a static graph (faster for small, fixed-length sequences; uses more memory).

  **When to use:**

  * **Very short sequences** (e.g. `< 20` timesteps) with tight latency constraints.
  * **Otherwise** leave `False` to leverage symbolic looping and lower memory use.

---