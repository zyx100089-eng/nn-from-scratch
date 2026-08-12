# Neural Networks from Scratch, and What They Taught Me About Adversarial Examples

*A project report. Written for anyone reading the repo — including, I
hope, an admissions tutor.*

---

## Summary

I built a neural network framework entirely from scratch — my own
reverse-mode autodiff engine, dense and convolutional layers, batch
norm, dropout, three optimisers, and a training loop — and then used
that framework to study **adversarial examples**: perturbations
invisible to the human eye that flip a model's predictions.

The project has two halves that share one engine:

1. **The framework.** Every layer is a graph op on the autodiff engine.
   There is no PyTorch, no TensorFlow, no autograd library — every
   gradient is either the chain rule applied by the engine or a
   hand-derived backward rule (convolution, max pooling, batch norm)
   verified against numerical differentiation.
2. **The study.** I trained an MLP on MNIST, attacked it with FGSM and
   PGD, measured *why* the attacks work (Goodfellow's linearity
   argument, verified numerically rather than asserted), tested
   targeted and transfer attacks, and measured the standard defence
   (adversarial training).

The headline results, all verified by the test suite:

- **89.4%** test accuracy on MNIST with a 784→128→64→10 MLP trained
  from scratch.
- A perturbation bounded by **0.1 per pixel** flips **~76%** of the
  model's correct predictions (FGSM), and **~84%** with an iterated
  attack (PGD).
- The model is **locally linear in its input** at small scales: the
  measured logit change from a perturbation matches the gradient
  prediction to a ratio of **1.00** — the premise of Goodfellow's 2014
  explanation, verified numerically.
- **Adversarial training** raises survival under attack from **16% to
  54%** at the same perturbation budget.
- Adversarial examples **transfer** between different architectures: a
  perturbation crafted against one model flips ~99% of another model's
  predictions at eps=0.3.
- A from-scratch CNN beats the best MLP on MNIST (97.8% vs 95.9%)
  with **8.5× fewer parameters** — and, honestly, does *not* beat it
  on Fashion-MNIST (see §4.5).

---

## 1. Why build a neural network from scratch?

A neural network is a function that maps an image (784 pixel values)
to class probabilities. Intuitively, if a "4" is classified as a "4"
with 89% confidence, a tiny change to the image should not change the
answer. It does. FGSM, a single gradient step bounded by 0.1 per pixel
— invisible to the eye — flips ~76% of predictions.

Why should anyone care? Because these examples are not a bug of my
implementation: they are a property of high-dimensional piecewise-linear
functions, and they have real-world consequences (a stop sign misread
as a speed limit, a medical image misread after imperceptible noise).
Understanding them requires understanding the geometry of the loss
landscape — which requires a network you can take apart. That is why I
built everything from first principles.

## 2. The autodiff engine

The core building block is a reverse-mode autodiff engine: a `Tensor`
holds a value, a gradient buffer, and a backward function implementing
the chain rule for the operation that produced it. A forward pass
builds a computation graph; `backward()` walks it in reverse
topological order, accumulating gradients. This is the same machinery
as PyTorch's autograd, written from scratch in ~300 lines.

Two implementation details were correctness-critical:

- **Broadcasting**: adding a bias `(3,)` to a batch `(N,3)` requires
  summing the gradient back over the batch axis — easy to get wrong,
  and wrong gradients train nothing.
- **Persistent parameters**: the optimizer must step on the *same*
  tensor objects the graph recorded gradients into, or training sits
  at chance accuracy forever. (This was a real bug — see §5.)

Every gradient was verified against numeric differentiation (finite
differences) to ~1e-9 before any training was trusted.

## 3. The framework

### 3.1 Layers as graph ops

Everything is a graph op on the engine:

- `Linear` — matmul + bias, He initialisation.
- `Conv2D` — im2col-style sliding windows; the backward is a
  col2im scatter, hand-derived and gradient-checked.
- `MaxPool2D` — backward routes each gradient to the argmax position.
- `Dropout` — inverted dropout; identity at inference.
- `BatchNorm` — the full backward through dvar and dmean, the
  trickiest hand-derived gradient in the project, verified against
  finite differences.
- Activations: ReLU, Sigmoid, Tanh, Softmax (full Jacobian-vector
  product — see §5 for why the naive shortcut is wrong).

### 3.2 Optimisers

SGD, SGD with momentum, and Adam (first/second moment estimates with
bias correction, Kingma & Ba 2015). All step the same parameter list
and zero gradient buffers.

### 3.3 The training loop

One step: build the graph for a batch, compute the loss, `backward()`
to accumulate gradients into every parameter, then `optimiser.step()`.
Minibatch SGD with weight decay, validation tracking, reproducible
seeds.

## 4. Experiments

### 4.1 Activation comparison (MNIST, 784→128→10, Adam, 12k samples)

| Activation | Test Accuracy |
|---|---|
| ReLU | 93.9% |
| Tanh | 92.8% |
| Sigmoid | 91.1% |

ReLU converges fastest and wins by a small margin; sigmoid is slowest
due to vanishing gradients in its saturation regions.

### 4.2 Optimiser comparison (MNIST, 784→128→10, 12k samples)

| Optimiser | Test Accuracy |
|---|---|
| SGD (lr=0.1) | 92.0% |
| SGD + Momentum (lr=0.1, μ=0.9) | 95.3% |
| Adam (lr=0.001) | 93.9% |

Momentum and Adam converge faster than vanilla SGD; momentum wins here
because the loss surface is smooth enough that a well-tuned velocity
term accelerates without overshooting.

### 4.3 Deliberate overfitting, then fixing it (1000 Fashion-MNIST samples)

A large network (784→256→128→10) trained on only 1000 samples:

| Config | Test Accuracy | Train–Val Loss Gap |
|---|---|---|
| No regularisation | 85.6% | 0.63 |
| Dropout (p=0.5) | 87.6% | 0.50 |
| L2 (λ=0.005) | 84.2% | 0.46 |

Without regularisation the training loss collapses to ~0 while
validation loss stays high — classic overfitting. Dropout improves
test accuracy by 2 points and tightens the gap; L2 tightens the gap
but slightly hurts accuracy here (the λ=0.005 penalty is strong for
this small network).

### 4.4 MNIST and Fashion-MNIST (full framework check)

All configurations reach 91%+ on MNIST (12k-sample subset),
confirming the implementation is correct. Fashion-MNIST is harder
across the board — the classes (shirts, pullovers, coats) share
silhouettes, so the task rewards local features more than global
ones.

### 4.5 MLP vs CNN

| Dataset | MLP (256→128) | CNN (2 conv + 2 pool) |
|---|---|---|
| MNIST | 95.9% (~235k params) | **97.8%** (~28k params) |
| Fashion-MNIST | 85.5% | 85.6% |

The CNN beats the MLP on MNIST by ~2 points despite using **8.5×
fewer parameters** — weight sharing is a better inductive bias for
images than global dense connections.

**Honest caveat:** the Fashion-MNIST margin (+0.1 points) is *not*
real. A 3-seed sweep (committed as `results/cnn_seed_sweep.csv`)
shows the CNN *behind* the MLP on Fashion-MNIST: 83.3% ± 0.8 vs
84.7% ± 0.5. The single-run "win" was seed noise. The
parameter-efficiency result is confirmed on both datasets; the
accuracy advantage is MNIST-only at this scale. That inconclusive
result is itself instructive: it shows *when* weight sharing pays
off — digits have consistent local structure (strokes, loops), while
garments share silhouettes that defeat even local feature detectors.

### 4.6 The adversarial study

#### Attacks

Flip rates (fraction of correctly-classified test images whose
prediction changes), 300-image sample:

| eps | FGSM | PGD (10 steps) |
|----:|-----:|---------------:|
| 0.05 | 25.7% | 29.5% |
| 0.10 | 75.7% | 84.3% |
| 0.30 | 100% | 100% |

Targeted PGD steered ~63% of images into a chosen class at eps=0.1.
All perturbations strictly respect the L∞ budget (verified).

#### Why the attacks work — measured

The standard explanation (Goodfellow et al. 2014): a ReLU network is
**piecewise-linear** in its input, so locally `logits(x+δ) ≈
logits(x) + g·δ` where `g` is the logit's gradient. A perturbation
with per-pixel bound `eps` has total norm `‖δ‖₂ ≈ eps·√784`, so the
logits change by an amount ~28× larger than any single pixel's change
— an invisible perturbation is, in aggregate, large.

Rather than assert this, I measured it. For the predicted class, I
computed the logit gradient `g`, applied an FGSM-sized step `δ`, and
compared the *actual* logit change with the linear prediction `g·δ`:

| eps | actual / predicted logit change |
|----:|--------------------------------:|
| 1e-5 | 1.00 |
| 1e-3 | 1.00 |
| 0.01 | 0.99 |
| 0.1 | 0.87 |

At small perturbation scales the ratio is exactly 1.00: the model
really is locally linear, so the gradient genuinely points across a
nearby decision boundary. At eps=0.1 the ratio drifts to 0.87 — the
perturbation has crossed ReLU kinks — which is precisely *why* FGSM
saturates at large eps and PGD (which re-computes the gradient each
step) keeps improving.

#### The defence

Accuracy after PGD-10 attack:

| eps | plain model | adversarially trained |
|----:|------------:|----------------------:|
| 0.05 | 70.5% | 79.4% |
| 0.10 | 15.7% | 54.2% |
| 0.30 | 0.0% | 0.0% |

The defence roughly triples survival at eps=0.1, but it is not
magic: at eps=0.3 both models collapse. The robustness/accuracy
trade-off is visible — the adversarially trained model trades a little
clean accuracy for robustness.

#### Transfer

Perturbations crafted against model A (784→128→64→10) were tested on
model B (784→256→10, independently trained). At eps=0.3 the transfer
flip rate was ~99% — nearly matching white-box. Attacks transfer
because both models approximate the same decision boundary.

## 5. Bugs I found and fixed

1. **Gradients vanished into throwaway tensors.** `Linear.forward`
   originally created fresh `Tensor`s for W and b on every call, so
   `parameters()` returned different objects than the ones the graph
   recorded gradients into. Training accuracy sat at 11.8% (chance is
   10%) with loss flat — the classic "no learning" smell. Fixed by
   making the parameters persistent attributes.
2. **Broadcasting broke the backward pass.** `(4,3) + (3,)` raised a
   broadcast error on the backward pass. Fixed with `_reduce_grad`
   (sum over broadcast axes) — the PyTorch behaviour, reimplemented.
3. **The Softmax backward shortcut is wrong in general.** The common
   trick of returning `dout` unchanged is only valid when Softmax is
   paired with cross-entropy loss (the combined gradient simplifies to
   `(p − y)/n`); it silently produces wrong gradients with any other
   loss. The current implementation computes the full Jacobian-vector
   product, `dX_i = s_i (dout_i − Σ_j dout_j s_j)`. Numerical gradient
   checking caught this coupling.
4. **Missing `__pow__`** — needed by the test harness for the `x²`
   loss used in gradient checks.

The meta-lesson: silent training failure (flat loss) is usually a
gradient-flow bug; verify gradients numerically before trusting any
training loop.

## 6. Limitations

1. **MLP and small CNN, not a modern architecture.** The phenomena
   studied are architectural — adversarial examples hold for CNNs and
   are famously *worse* there — but I did not re-run the linearity
   measurement on a CNN to claim generality.
2. **The linearity ratio is an interpretive metric.** "Locally linear"
   is made precise by the measurement at small eps (ratio ≈ 1.00), but
   the threshold for "linear enough" is my judgement, not a
   statistical test.
3. **Adversarial training is the standard defence.** Its value here is
   verification, not novelty: a textbook defence, implemented from
   scratch and shown to work on my own model.
4. **L∞-norm attacks only.** FGSM and PGD operate in the L∞ ball; L2
   attacks are a natural extension.
5. **Pure-NumPy convolution is slow** on full datasets (~minutes per
   epoch), which is why the CNN experiments use subsets or few epochs.

## 7. What I'd do next

- L2-bounded PGD, and a comparison of which norm is the stronger
  constraint.
- Universal perturbations (Moosavi-Dezfooli et al.): a single
  perturbation that fools most images.
- Certified robustness bounds (e.g. interval propagation) — a nice fit
  for the from-scratch engine.
- Re-run the linearity measurement on the CNN.
- Benchmark the framework against a reference PyTorch implementation
  to quantify the overhead of doing it from scratch.

## 8. Conclusion

I built a neural network framework from scratch and used it to study
adversarial examples end-to-end: attack (FGSM, PGD, targeted), explain
(the linearity argument, verified at ratio 1.00), defend (adversarial
training, survival 16% → 54%), and generalise (transfer between
architectures). The project's value is not in any single number but in
the discipline of verifying every claim — gradients to 1e-9, budgets
to machine precision, and the field's central explanation to a
measured ratio.

---

*Reference: Goodfellow, Shlens, Szegedy, "Explaining and Harnessing
Adversarial Examples" (ICLR, 2015).*
