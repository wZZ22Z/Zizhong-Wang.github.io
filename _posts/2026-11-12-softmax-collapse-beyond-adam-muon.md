---
layout: single
title: "Softmax Collapse Beyond Adam: What Happens to Muon?"
excerpt: "How a numerical error changes gradient geometry, momentum spectra, and Muon training dynamics."
date: 2026-11-12
permalink: /blog/2026/11/12/softmax-collapse-beyond-adam-muon/
author_profile: true
read_time: true
comments: false
share: false
related: false
tags:
  - Softmax Collapse
  - Muon Optimizer
  - Numerical Stability
---

*How a numerical error changes gradient geometry, momentum spectra, and Muon training dynamics.*

**Zizhong Wang · November 2026**

**Keywords**: Softmax Collapse, Muon Optimizer, Numerical Stability


## 1. Overview

A recent numerical analysis of softmax and cross-entropy [1] showed that standard implementations can enter a floating-point absorption regime during training. This phenomenon, called **Softmax Collapse**, can corrupt the gradient, trigger "slingshot" behavior (loss spikes), and reduce optimization efficiency.

As training progresses, a model can become increasingly confident in its predictions(large logit margins). However, large logit margins can push softmax into the floating-point absorption regime: the target probability is rounded to exactly 1, causing the corresponding target-logit gradient to become exactly zero even though its mathematically correct value is still nonzero.

For Adam, the consequence is intuitive: corrupted gradients contaminate first- and second-moment estimates($$m_t$$ and $$v_t$$). But Muon is different.  It operates on matrices and transforms its momentum matrix toward its polar factor $$UV^\top$$, which makes the matrix spectrum and singular values relevant to the update.

This raises a natural question:

**What happens when Softmax Collapse corrupts the gradient seen by Muon?**

### TL;DR

- Softmax Collapse also affects Muon. Fixing it improves training stability and optimization behavior, although through a different mechanism from Adam.

- Softmax Collapse does more than shrink gradients in Muon: it can dramatically concentrate the spectrum of gradient and momentum matrices, producing much lower effective rank.
- Muon can turn this transient gradient corruption into a long-lived unstable training regime: momentum preserves the abnormal state, while normalization prevents the update magnitude from naturally vanishing. Fixing cross-entropy strongly suppresses this regime.

<figure class="blog-figure">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_Loss.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_Loss.png' | relative_url }}" alt="Training dynamics of Adam, Muon, and SGD with Native cross-entropy." loading="lazy" decoding="async"></a>
  <figcaption>Training dynamics of Adam, Muon, and SGD with Native cross-entropy.</figcaption>
</figure>



<figure class="blog-figure">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_Fixed_Loss.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_Fixed_Loss.png' | relative_url }}" alt="Training dynamics of Adam, Muon, and SGD with Fixed cross-entropy." loading="lazy" decoding="async"></a>
  <figcaption>Training dynamics of Adam, Muon, and SGD with Fixed cross-entropy.</figcaption>
</figure>

## 2. Cause of the Collapse

### 2.1 Background Knowledge: Softmax and Cross-Entropy

#### Softmax

Softmax is commonly used to convert a vector of logits into probabilities. Given logits $$z=[z_1,z_2,\dots,z_n]$$, the softmax probability of class $$i$$ is

$$
q_i=\operatorname{softmax}(z_i) =\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$

where $$q_i>0$$ and $$\sum_i q_i=1$$.

For any constant $$c$$,

$$
\operatorname{softmax}(z+c)=\operatorname{softmax}(z). 
$$

This property is called **shift invariance**. For example, $$[1,2,3]$$ and $$[101,102,103]$$ produce exactly the same softmax distribution.

#### Cross-Entropy

For classification, softmax is usually combined with cross-entropy loss. One important property of softmax is **shift invariance**: adding the same constant to every logit does not change the output $$\operatorname{softmax}(z+c)=\operatorname{softmax}(z).$$

Therefore, if all logits are shifted by the same amount, the loss should not change. There should be **no gradient** in this shared-shift direction. 

#### Gradient

For softmax followed by cross-entropy, the gradient with respect to each logit is $$g_i=q_i-p_i$$, where $$q_i$$ is the predicted probability and $$p_i$$ is the target probability.

Therefore, $$\sum_i g_i=\sum_i(q_i-p_i)=\sum_i q_i-\sum_i p_i=1-1=0$$

### 2.2 Softmax Collapse: Why it happens?

#### 2.2.1 Floating-Point Absorption

Softmax Collapse comes from a basic limitation of floating-point arithmetic: computers can only represent a finite set of real numbers.

In the IEEE 754 format, an FP32 number contains 1 sign bit, 8 exponent bits, and 23 fraction bits:

<figure class="blog-figure blog-figure--compact">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/FP32_IEEE_Format.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/FP32_IEEE_Format.png' | relative_url }}" alt="IEEE 754 single-precision floating-point format: 1 sign bit, 8 exponent bits, and 23 fraction bits." loading="lazy" decoding="async"></a>
  <figcaption>IEEE 754 single-precision floating-point format: 1 sign bit, 8 exponent bits, and 23 fraction bits.</figcaption>
</figure>


Because the number of fraction bits is limited, two representable floating-point numbers always have a finite spacing between them. Around 1, the spacing between adjacent FP32 numbers is approximately $$2^{-23}\approx1.19\times10^{-7}$$. This means that FP32 cannot represent every real number between 1 and $$1+1.19\times10^{-7}$$.

For example, mathematically, $$1+10^{-9}=1.000000001$$. But there is no FP32 number close enough to represent this tiny increment around 1. After rounding, the result becomes $$1+10^{-9}=1$$.

When two numbers have very different magnitudes, the smaller value can therefore be completely lost during addition. This phenomenon is called **floating-point absorption**.

<figure class="blog-figure blog-figure--compact">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/numerical_joke_EN.jpeg' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/numerical_joke_EN.jpeg' | relative_url }}" alt="Floating-point arithmetic, in one meme." loading="lazy" decoding="async"></a>
  <figcaption>Floating-point arithmetic, in one meme.</figcaption>
</figure>


This may look like a small numerical detail, but it becomes important for softmax because its normalization can eventually involve this type of computation.

#### 2.2.2 Softmax Overflow

Another important factor behind Softmax Collapse is the way softmax is implemented to avoid numerical overflow.

In practice, directly computing $$q_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$$ can be numerically unsafe when some logits are large, because the exponential function grows quickly and may overflow in finite precision.

A standard solution is to subtract the maximum logit before exponentiation: $$q_i= \frac{e^{z_i-m}} {\sum_j e^{z_j-m}},m=\max_j z_j$$.(This transformation is
because **shift-invariance**). After this transformation, the largest shifted logit is always 0, so its exponential becomes exactly 1.

However, it also means that the denominator now has the form $$1+\sum_{j\neq \operatorname{argmax}(z)} e^{z_j-m}$$, which is exactly the format we talked about(Floating Point Absorption).

In other words, **subtract-max solves the overflow problem, but it does not solve the absorption problem**.

#### 2.2.3 The Inevitable Collapse

Floating-point absorption and the subtract-max implementation provide the numerical conditions for Softmax Collapse. But what makes this problem especially important is that **training itself continuously pushes softmax toward this regime**.

As a model learns, it becomes more confident in its predictions. For a correctly classified sample, this usually means that the target logit becomes increasingly larger than the competing logits. In other words, the **logit margin** grows during training.

The key insight is that **Softmax Collapse is not caused by an unusual input or an isolated numerical accident**. Cross-entropy training rewards increasingly confident predictions, which naturally increases logit margins and pushes the computation toward the absorption regime.

<figure class="blog-figure blog-figure--compact">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse.png' | relative_url }}" alt="Target-gradient collapse near the FP32 absorption cliff." loading="lazy" decoding="async"></a>
  <figcaption>Target-gradient collapse near the FP32 absorption cliff.</figcaption>
</figure>

This does not mean that every pretraining run inevitably enters Softmax Collapse: sufficiently large logit margins place the computation in this numerical regime. The experiments here deliberately drive training into that regime.

### 2.3 Softmax Collapse: An Example

This is a simple example on softmax collapse.

Assume the logits are $$z=[20,0]$$, and the first class($$z_1$$) is the correct target. After subtracting the maximum logit, we get $$[0,-20]$$. Let $$\delta=e^{-20}\approx2.06\times10^{-9}$$, the correct softmax probabilities are $$q_1=\frac{1}{1+\delta}, q_2=\frac{\delta}{1+\delta}$$. 

$$\because g_1=q_1-1,g_2=q_2$$

$$\therefore g_1 = \frac{1}{1+\delta}-1 = -\frac{\delta}{1+\delta}$$, $$g_2 = \frac{\delta}{1+\delta}$$

This satisfies $$\sum_j g_j=0$$

However,  $$\delta\approx2.06\times10^{-9}$$ is too small to change 1 at FP32 precision($$1+\delta \rightarrow 1$$). This is where softmax collapse occurs: $$\tilde{g_1}=1-1=0$$, $$\tilde{g_2}=\delta$$

Compared with the mathematically correct gradient, two things have changed:

- First, the target-logit gradient disappears. In training, different optimizers can respond very differently to the same small gradient error. We will return to this when discussing SGD, Adam, and Muon.

- The second consequence is that the gradient violates the zero-sum constraint $$\sum_j g_j=0$$. 

These two changes have very different implications for optimization, which we discuss next.

### 2.4 Softmax Collapse: When the Gradient Goes Wrong

#### 2.4.1 The Gradient That Disappears

The first problem caused by Softmax Collapse is straightforward: a gradient component that should exist is numerically rounded to zero. In the example above, the target-logit gradient should be a small but nonzero negative value. After collapse, however, it becomes exactly zero. 

By itself, this does not tell us how serious the training impact will be. The missing gradient can be extremely small, and different optimizers may react very differently to the same error. SGD directly follows the current gradient, Adam keeps historical first- and second-moment states, while Muon processes matrix-valued momentum through normalization and a spectral transformation. 

We will return to this optimizer-dependent effect later.

#### 2.4.2 The Direction That Shouldn't Exist

The second problem is that after softmax collapse, the optimizer is trying to update towards a direction that actually doesn't help to reduce loss.

Since we know softmax and cross-entropy have shift-invariance, which means: $$\mathcal{L}(z+c\mathbf{1})=\mathcal{L}(z)$$. Therefore, the true loss has no gradient with respect to this shared shift: $$\frac{\partial\mathcal{L}}{\partial c} = \sum_i \frac{\partial\mathcal{L}}{\partial z_i} = \sum_i g_i = 0$$

However, after softmax collapse, the computed gradient no longer sums to zero. In this case, the optimizer receives a spurious gradient component that actually won't help reduce the loss.

This example shows that Softmax Collapse does not merely reduce the gradient magnitude. **It changes the gradient direction by introducing a spurious shared-shift component.** Part of the computed update can therefore be spent following a direction that cannot reduce the true cross-entropy loss, reducing the efficiency of each optimization step.

## 3. When Softmax Collapse Meets Optimizer

So far, we have only looked at the gradient itself. Softmax Collapse removes a mathematically valid component and introduces a spurious one. But both errors can initially be extremely small.

***Why can such a tiny numerical error eventually produce visible training instability?***

The answer depends on the optimizer.

### Experimental Setup

We use the same 99.26M-parameter GPT-style Transformer and modular-addition task across all runs, with identical initialization and training data. Training is performed in FP32 with full-batch updates for 10,000 optimizer steps at a constant learning rate of 0.001, with weight decay disabled. Native and Fixed CE differ only in the cross-entropy implementation. The Adam baseline uses $$\beta_1=0.9$$ and $$\beta_2=0.98$$, while SGD uses no momentum; for Muon runs, matrix-valued Transformer weights use quintic Newton–Schulz Muon and the remaining parameters use Adam with the same moment coefficients.

### 3.1 SGD

#### 3.1.1 SGD Background

SGD updates the parameters directly using the current gradient: $$\theta_{t+1}=\theta_t-\eta g_t$$, where $$\eta$$ is learning rate.

Unlike Adam, SGD does not maintain first- or second-moment estimates. The gradient observed at the current step directly determines the update at that step.

This makes SGD a useful baseline for understanding Softmax Collapse.

#### 3.1.2 Collapse in SGD

Recall the example from Section 2.3. Before collapse, the correct logit gradient is approximately $$g= \begin{bmatrix} -\delta\\ \delta \end{bmatrix}$$where $$\delta$$ is extremely small. After Softmax Collapse, $$\tilde g= \begin{bmatrix} 0\\ \delta \end{bmatrix}$$. Under SGD, the corresponding updates are $$\Delta\theta=-\eta g,\widetilde{\Delta\theta}=-\eta\tilde g$$ . The difference introduced by Softmax Collapse is therefore $$\begin{bmatrix} \eta \delta\\ 0 \end{bmatrix}$$. Since Softmax Collapse occurs only when $$\delta$$ is already extremely small, and the learning rate $$\eta$$ is usually small, the immediate difference between the correct and collapsed SGD updates can also be extremely small.

More importantly, SGD has no internal moment state that directly remembers this error. A corrupted gradient changes the current update, and therefore may still alter the future optimization trajectory, but there is no additional optimizer state such as $$m_t$$ or $$v_t$$ that stores the collapse and later amplifies it.

This does not mean that SGD is immune to Softmax Collapse. The gradient is still incorrect, and the spurious direction discussed in Section 2.4 still reduces the quality of the optimization signal. However, SGD does not have the state-dependent amplification mechanism that produces the characteristic delayed loss spikes in Adam.

In the run shown above, SGD itself does not reach the same low-loss FP32 precision cliff as Adam or Muon. The curve should therefore be read as a baseline for optimizer dynamics, not as evidence that SGD remains unaffected after entering the same collapse regime.

### 3.2 Adam

#### 3.2.1 Adam Background

Unlike SGD, Adam maintains two running statistics of the gradient:$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$, $$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$. Here, $$m_t$$ tracks the first moment of the gradient, while $$v_t$$ tracks its squared magnitude. After bias correction: $$\hat m_t=\frac{m_t}{1-\beta_1^t},\hat v_t=\frac{v_t}{1-\beta_2^t}$$ and Adam updates the parameters using $$\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$.

#### 3.2.2 Wrong Memory

Now consider what happens when Softmax Collapse makes a target-related gradient numerically zero for many consecutive steps. We know softmax collapse will make the target-logit's gradient zero, and if it last for many steps, its $$m_t$$ and $$v_t$$ will $$\approx 0$$. In this case, Softmax Collapse does not only make the current right gradient disappear, but also make history state near zero.

At a moment, when gradient suddenly not zero(no collapse), $$m_{\mathrm{new}}\approx(1-\beta_1)g$$, $$v_{\mathrm{new}}\approx(1-\beta_2)g^2$$. The Adam update becomes: $$\frac{m_{\mathrm{new}}}{\sqrt{v_{\mathrm{new}}}} \approx \frac{1-\beta_1}{\sqrt{1-\beta_2}} \operatorname{sign}(g)$$. For the Adam configuration used in our experiments, $$\beta_1=0.9$$ and $$\beta_2=0.98$$, $$\frac{1-\beta_1}{\sqrt{1-\beta_2}}=\frac{0.1}{\sqrt{0.02}} \approx0.707$$. So even if the returning gradient is very small in absolute magnitude, Adam can normalize it into an update on the scale of the learning rate.

The point is not that 0.707 is unusually large for an ordinary Adam step, but that a signal that should be extremely small in absolute magnitude is renormalized to the learning-rate scale.

This creates a delayed effect:

$$\text{Softmax Collapse}\rightarrow\text{moment decay}\rightarrow\text{gradient recovery}\rightarrow\text{large normalized update}\rightarrow\text{loss spike}.$$

When this process repeats during training, it can produce the characteristic periodic loss spikes often referred to as **slingshot behavior**.

#### 3.2.3 Slingshot Cycle

Softmax collapse can explain one loss spike. But why does the same spike appear again and again?

The key is that a spike also **resets the training state**.

Before the spike, many examples are confident and have entered the Softmax Collapse regime. Their gradients become extremely small, and Adam's second-moment estimate gradually decays.

When some gradients recover, Adam produces a much larger update, causing the loss spike. This large update pushes the model away from the highly confident collapsed regime. At the same time, the newly enlarged gradients increase Adam's second moment again.

The model can train normally for a while, and repeat the pattern: The loss decreases, predictions become more confident, and cross-entropy once again pushes the logit margins upward. Eventually, some examples cross the floating-point precision cliff **again** and Softmax Collapse returns.

<figure class="blog-figure">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/slingshot_cycle.svg' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/slingshot_cycle.svg' | relative_url }}" alt="Slingshot Cycle Demonstration" loading="lazy" decoding="async"></a>
  <figcaption>Slingshot Cycle Demonstration</figcaption>
</figure>

Previous work suggests that examples can move back below the precision cliff through **weight decay or residual non-target updates** [1].

Our experiment is interesting because **weight decay is disabled**, yet the repeated slingshot pattern still appears. This shows that weight decay is not necessary for the cycle in our setting. Residual gradients and continued updates through shared model parameters are sufficient candidates for bringing some examples back across the precision cliff.



## 4. Softmax Collapse in Muon

Adam gives us one mechanism: Softmax Collapse corrupts coordinate-wise gradient history. But Muon has no element-wise second moment. Instead, it builds and transforms matrix-valued momentum. If Softmax Collapse also affects Muon, the failure must appear through a different path.

### 4.1 Muon Background

#### 4.1.1 Geometry Update

Muon was motivated by an empirical observation about matrix-valued optimizer updates. In Transformer layers, momentum and Adam updates for 2D weight matrices often have highly concentrated spectra: a few singular directions dominate, while many other directions have much smaller singular values. Jordan et al. hypothesized that these weak “rare directions” may still contain useful learning signal, and proposed orthogonalizing the matrix update so that they are no longer dominated by only a few large directions [2].

Given a matrix-valued momentum $$M=U\Sigma V^\top$$,  the idealized Muon transformation replaces its singular values with 1: $$U\Sigma V^\top \longrightarrow UV^\top$$. The matrix $$UV^\top$$ is the polar factor of $$M$$. Geometrically, this is steepest descent under the spectral norm[3].

#### 4.1.2 Newton-Schulz

Computing an exact SVD at every optimization step would be expensive. Practical Muon therefore approximates the polar transformation using **Newton-Schulz-style polynomial iterations**, which require only matrix multiplications(GEMM) and are well suited to GPUs [2].

If $$X=U\Sigma V^\top$$, a quintic Newton-Schulz step $$aX+b(XX^\top)X+c(XX^\top)^2X$$ preserves the singular vectors while transforming each singular value according to $$\sigma \longrightarrow a\sigma+b\sigma^3+c\sigma^5$$.

There are several variants for Newton–Schulz iteration. For example, Polar Express [4] chooses a different quintic polynomial aiming at approaching the ideal polar factor. These variants are already appearing in large-scale pretraining(Qwen3.8-Flash-Next, DeepSeek-V4)[5,6]. 

### 4.2 Collapse in Matrix

Adam sees Softmax Collapse as missing gradient components. Muon sees a change in the shape of the whole gradient matrix.

When many examples collapse, their target components are rounded to zero. This removes part of the gradient information contributed by those examples, while the remaining non-zero components continue to accumulate across the batch. As fewer independent directions contribute significant energy, the matrix can become dominated by only a few singular directions.

The resulting spectrum is not merely low-rank: a small number of singular directions carry most of the meaningful energy, followed by a long tail of much weaker directions. With Fixed CE, this extreme concentration is greatly reduced and the spectrum remains distributed across more directions.

The three snapshots correspond to step 1872 (before the event), step 1873 (the first loss spike, where one previously confident example becomes misclassified), and step 1907 (after the event).



<figure class="blog-figure">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Muon_Spetrum_Spike.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Muon_Spetrum_Spike.png' | relative_url }}" alt="Gradient and momentum spectra around a Muon loss spike." loading="lazy" decoding="async"></a>
  <figcaption>Softmax Collapse in Quintic NS and Exact SVD. Even in exact SVD, loss spike still occurs.</figcaption>
</figure>

A natural question is whether this instability is caused by Softmax Collapse itself or by Muon's approximate orthogonalization. To separate the two, we replaced the finite-step Newton–Schulz iteration with an exact SVD-based polar factor.

<figure class="blog-figure">
  <a href="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_NS_vs_SVD.png' | relative_url }}"><img src="{{ '/images/blog/softmax-collapse-beyond-adam-muon/Softmax_Collapse_NS_vs_SVD.png' | relative_url }}" alt="Softmax Collapse in Quintic NS and Exact SVD. Even in exact SVD, loss spike still occurs." loading="lazy" decoding="async"></a>
  <figcaption>Softmax Collapse in Quintic NS and Exact SVD. Even in exact SVD, loss spike still occurs.</figcaption>
</figure>

The instability survives exact SVD and disappears under both orthogonalizers once cross-entropy is fixed, indicating that Softmax Collapse is the upstream trigger rather than Newton–Schulz approximation itself.

The orthogonalizer still shapes what follows. Newton–Schulz applies only a few polynomial iterations, so very weak singular directions may remain far from where an exact polar factor would place them. When collapse produces a highly head-dominated spectrum with a long weak tail, this difference can become especially pronounced.

### 4.3 Why Does the Instability Persist?

#### Momentum stores it

Muon accumulates matrix-valued gradient history in momentum:

$$
M_t=\mu M_{t-1}+(1-\mu)G_t.
$$

In the collapse regime, the accumulated state can already be extremely small. A gradient event that is large relative to that state can therefore dominate the new momentum, changing its magnitude, direction, and spectrum in a single step. Even if the event itself lasts only one step, its contribution to momentum remains afterward: when subsequent gradients are negligible, $$M_{t+k}\approx\mu^k M_t$$. The magnitude decays, but the abnormal geometry can persist.

#### Normalization keeps it alive

Before applying its spectral transformation, Muon normalizes the matrix $$H$$ supplied to the orthogonalizer, which is the momentum or its Nesterov-adjusted form:

$$
X_0=\frac{H}{\max(\|H\|_F,\epsilon)}.
$$

As long as $$\|H\|_F$$ stays above the normalization floor $$\epsilon$$, a uniform decay in $$H$$ is divided out. If $$H$$ becomes $$\alpha H$$ without changing direction or relative singular values, the normalized input remains the same while $$\alpha\|H\|_F>\epsilon$$. The subsequent spectral transformation can therefore keep producing an update of comparable magnitude even as the raw momentum norm falls.

Below the floor, the denominator stops tracking the shrinking state, so $$X_0=H/\epsilon$$ can shrink with it. For a fixed number of polynomial iterations, the transformed update then tends to zero as $$H$$ tends to zero. This gives Fixed CE a route to fading updates as its uncorrupted state decays sufficiently far; crossing the floor does not itself make the update instantly zero.

**Momentum stores the abnormal geometry; normalization prevents the update amplitude from fading with it.** Together, these mechanisms explain how a transient corruption can sustain an unstable regime long after the original event.

## 5. Gradient Quantization

Softmax Collapse also changes how Muon interacts with gradient quantization.

Consider a collapse event in which a previously confident example suddenly becomes misclassified. At this point, the matrix gradient becomes extremely concentrated. For example, there are only 3 main directions. Then we quantize this gradient. Due to quantization error, we introduce many small noises to the gradient(adding the gradient spectra a few weak singular values). In Muon Newton-Schulz iteration, it naturally amplifies small singular values and tries to make it near 1. As a result, weak directions introduced by quantization error becomes larger and larger and eventually, same magnitude as ground-truth gradient. 

The situation changes after fixing Softmax Collapse. With Fixed CE, meaningful gradient signal remains distributed across a broader set of singular directions instead of collapsing into only a few dominant modes. Quantization noise therefore has less opportunity to create new directions inside an almost empty spectral tail.  As a result, the same small quantization perturbations are less likely to become competitive with the true gradient signal after Newton-Schulz iteration.

## References

[1] **Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.** “A Fix for Softmax and Cross-Entropy.” *arXiv preprint arXiv:2610.31642*, 2026. [arXiv](https://arxiv.org/abs/2610.31642)

[2] **Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein.** “Muon: An Optimizer for Hidden Layers in Neural Networks.” 2024. [Blog post](https://kellerjordan.github.io/posts/muon)

[3] **Jeremy Bernstein and Laker Newhouse.** “Old Optimizer, New Norm: An Anthology.” *NeurIPS 2024 Workshop on Optimization for Machine Learning*, 2024. arXiv:2409.20325. [Paper](https://arxiv.org/abs/2409.20325)

[4] **Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower.** “The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm.” *International Conference on Learning Representations (ICLR)*, 2026. arXiv:2505.16932. [Paper](https://arxiv.org/abs/2505.16932)

[5] **Zihan Qiu et al.** “On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability.” *arXiv preprint arXiv:2608.30320*, 2026. [arXiv](https://arxiv.org/abs/2608.30320) 

[6] **DeepSeek-AI.** “DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.” *arXiv preprint arXiv:2606.19348*, 2026.  [Paper](https://arxiv.org/abs/2606.19348)
