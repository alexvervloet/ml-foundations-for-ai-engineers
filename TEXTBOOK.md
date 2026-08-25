# Chapter 24: The Numeric Contracts Beneath the Model

This text explains the mechanics behind the runnable lessons. It assumes you build AI
applications and need enough model knowledge to debug a bad result, review an ML claim,
or estimate a deployment. It does not try to turn one repository into a graduate
curriculum.

## 1. Start with axes, not arrays

A tensor is an array plus meaning assigned to each axis. The same shape `(32, 128, 768)`
could mean batch, sequence, hidden width, or something completely different. A matrix
multiplication can be numerically legal and semantically wrong if two equal lengths
represent different things.

Write the axis contract before the operation. For an affine layer:

```text
inputs   [batch, input_width]
weights  [input_width, output_width]
bias     [output_width]
output   [batch, output_width]
```

The operation is `Y = XW + b`. Bias broadcasting is deliberate because one output bias
is applied to every batch row. Broadcasting an input-width vector over the result would
be a bug, even if a particular square layer made the lengths equal.

### Direction and magnitude

For vectors `a` and `b`, cosine similarity is:

```text
cos(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Multiplying either vector by a positive scalar leaves its direction unchanged. A zero
vector has no direction, so cosine similarity is undefined. Returning zero for that case
would turn an invalid input into a plausible similarity score.

Embedding similarity inherits these limits. A high cosine says two vectors point in a
similar direction under one embedding model. It does not prove factual equivalence,
relevance to a user, or stability across model revisions.

## 2. Logits and probabilities answer different questions

A logit is an unnormalized score. For a vector `z`, softmax produces:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

The obvious implementation can overflow. If the largest logit is 1,000, `exp(1000)` is
outside ordinary floating-point range. Subtract the maximum `m`:

```text
p_i = exp(z_i - m) / sum_j exp(z_j - m)
```

The result is identical because the shared factor `exp(-m)` cancels. Every exponent is
now at most zero.

Softmax is invariant to a shared shift. It is not invariant to scale. Multiplying logits
by two sharpens the distribution. Dividing by a temperature greater than one flattens it.
That distinction later powers temperature scaling and sampling.

Do not read a softmax number as a measured chance of correctness without calibration
evidence. It is normalized model preference for one input. Confidence is a behavioral
claim across many labelled outcomes.

## 3. Cross-entropy turns supervision into a loss

For one target class `y`, categorical cross-entropy is:

```text
loss = -log(p_y)
```

If the model assigns the target probability one, the loss approaches zero. If target
probability approaches zero, the loss grows without bound. Using the stable log-sum-exp
form avoids materializing a fragile softmax:

```text
loss = log(sum_j exp(z_j)) - z_y
```

Framework losses combine these operations. PyTorch documents that
[`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
accepts unnormalized logits. Applying softmax first changes the calculation and costs
stability.

### What lower loss establishes

Lower training loss establishes that the optimizer fit the training objective better.
It does not establish:

- generalization to held-out data;
- calibrated confidence;
- acceptable subgroup behavior;
- robustness to changed input distributions;
- useful generation under the chosen sampling policy.

Loss is also unitless. Comparing values across different tokenizations, label smoothing,
class weighting, or reduction rules needs care.

## 4. A gradient is local information

The derivative of a scalar loss with respect to a parameter gives the local slope. For
`f(x) = (x - 3)^2 + 1`, the analytic derivative is `2(x - 3)`.

A centered finite difference estimates the same quantity:

```text
f'(x) approximately equals (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
```

Large epsilon measures a secant over too wide a region. Tiny epsilon subtracts nearly
equal floating-point numbers and can lose the signal to cancellation. The useful range
depends on dtype, scale, and the function.

Autograd records operations and applies the chain rule backward through the graph. A
three-way agreement among analytic, numeric, and autograd gradients is strong evidence
for that local path. It says nothing about data correctness or whether the objective
represents the product goal.

Useful gradient debugging questions include:

- Is the parameter connected to the loss graph?
- Did an accidental detach cut the graph?
- Are gradients finite before the optimizer step?
- Are they zero because the optimum is near, or because an activation saturated?
- Did gradient accumulation include the intended number of batches?

## 5. Gradient descent is a state machine

The basic update is:

```text
parameter_next = parameter_now - learning_rate * gradient_now
```

The minus sign moves against the local slope. Learning rate converts slope into step
size. A rate that works for one scale can overshoot on another.

A training loop should distinguish why it stopped. This course uses three terminal
states:

```text
converged   declared tolerance passed
max_steps   budget ended first
diverged    arithmetic became nonfinite
```

A finite loss that grows for 12 steps is bad evidence, but it is not the same state as
NaN. The remedies differ. Budget exhaustion may need more steps, a new rate, or a better
stop rule. Nonfinite arithmetic may need lower precision controls, loss scaling, clipping,
or investigation of invalid data.

Real training adds minibatches, vector parameters, momentum, adaptive moments, schedules,
regularization, and distributed reduction. Those change the update rule, not the need for
explicit observations and terminal semantics.

## 6. Attention is weighted retrieval under a permission rule

Scaled dot-product attention calculates:

```text
scores  = Q @ K.T / sqrt(key_width)
weights = softmax(mask(scores))
output  = weights @ V
```

Queries describe what each position seeks. Keys describe what each stored position offers.
Values contain the information to mix. A row of attention weights sums to one across
allowed keys.

The square-root scale matters because dot-product variance grows with vector width. Without
scaling, wide random query and key vectors create large score gaps that can push softmax
into saturated regions. The original transformer paper specifies this operation in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

### Causal masking

At position `i`, a causal decoder may read keys from positions zero through `i`. It must
not read later positions. A lower-triangular permission matrix expresses that rule.

Mask before softmax. Zeroing forbidden weights after softmax without renormalizing changes
row mass. Giving a forbidden score zero does not remove it because zero may exceed allowed
negative scores. The usual additive implementation uses negative infinity before softmax.

Shape checks are not enough to validate causality. Change a future value or future token
and verify earlier outputs do not move. That counterfactual tests the property you care
about.

## 7. What a transformer block adds

A decoder block in this course uses pre-normalization:

```text
h1 = x + attention(layer_norm(x))
h2 = h1 + feed_forward(layer_norm(h1))
```

Residual additions require the update and residual stream to share shape. Multi-head
attention splits model width into `heads * head_width`, computes attention per head, then
merges heads back to model width. Construction must reject a model width that cannot divide
evenly.

The feed-forward network acts independently at each sequence position. Attention mixes
positions. These are different operations even though both return the residual shape.

PyTorch's [transformer building-block tutorial](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
shows production-oriented use of scaled-dot-product attention and nested tensors. The
course block stays small enough to read. It omits performance work and modern architecture
choices that would distract from the tensor path.

### From block to language model

A tiny causal language model adds:

1. token embeddings;
2. position embeddings;
3. one or more causal blocks;
4. final normalization;
5. a projection to one logit per vocabulary token.

For a training row `[t0, t1, t2, t3]`, the inputs are `[t0, t1, t2]` and targets are
`[t1, t2, t3]`. Flattening batch and sequence for cross-entropy is valid only after logits
and targets preserve that alignment.

## 8. Sampling is part of system behavior

Greedy decoding chooses argmax. Sampling draws according to a transformed probability
distribution. The model supplies logits; the application supplies decoding policy.

Temperature `T` divides logits before softmax:

```text
p_i(T) = softmax(z / T)_i
```

`T < 1` sharpens. `T > 1` flattens. Temperature must stay positive. At exactly zero the
formula is undefined, so greedy decoding should be a separate branch rather than a numeric
hack.

Top-k keeps only the highest `k` logits and excludes the rest. Ties need a stable rule if
exact reproducibility matters. This course uses token id as the secondary ordering key.

Randomness has state. Injecting a NumPy generator makes that state visible to tests. A
global seed can still be disturbed by an unrelated draw elsewhere in the process.

Sampling tests should cover:

- excluded tokens never appearing;
- the same generator state replaying the same sequence;
- invalid temperature and top-k bounds failing before a draw;
- greedy ties following the documented rule.

None of those establishes generation quality. They establish decoding mechanics.

## 9. Calibration needs a separate split

A classifier is calibrated when predictions made near confidence `c` are correct about a
fraction `c` of the time. Modern neural networks can be accurate and overconfident. Guo and
coauthors measured this behavior and evaluated temperature scaling in
[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

Temperature scaling selects one positive scalar on labelled calibration data. Dividing all
class logits by the same temperature preserves their ordering, so accuracy does not change.
Confidence does.

The split roles are:

```text
training      fit model parameters
calibration   fit temperature or another calibration map
test          judge the fitted model and calibration process
```

Evaluating ECE on the same rows used to choose temperature reuses evidence. It reports fit,
not held-out behavior.

### A grid search only sees its own endpoints

Selecting temperature from a declared grid is honest only if the winner sits inside the
grid. If the smallest or largest candidate wins, loss was still falling when the search
ran out of candidates, and the reported value says more about the grid than the data.

A calibration split the model classifies perfectly guarantees this. With every row
correct and confident, cross-entropy falls monotonically as temperature approaches zero,
so the smallest candidate always wins and any ECE improvement can be driven arbitrarily
close to zero by extending the grid downward. There is no temperature to find. The
correct response is to say the fit is unresolved, not to report the grid floor as a
result. `fit_temperature` returns `on_grid_boundary` for exactly this check.

Expected calibration error groups predictions into confidence bins, compares mean
confidence with empirical accuracy in each bin, and weights the absolute gaps. Its value
depends on bin boundaries. Small datasets leave many bins sparse. Always state the binning
rule and pair the scalar with reliability plots or cohort checks in serious work.

## 10. Quantization changes representation

Symmetric per-tensor quantization chooses a signed integer range and one scale:

```text
scale = max(abs(weights)) / maximum_integer_level
q     = round(weights / scale)
w_hat = q * scale
```

Clipping keeps `q` inside the declared range. Values that map to the same integer level
cannot be separated after dequantization.

More integer levels usually reduce reconstruction error for the same range. Outliers can
make one global scale waste levels near zero, which motivates per-channel scales and other
schemes. The quantization white paper by Nagel and coauthors provides a broader taxonomy:
[A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295). Gholami
and coauthors survey the wider method space in
[A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630).

Keep four claims apart:

1. logical precision, such as int4;
2. packed storage bytes;
3. task or logit drift after reconstruction;
4. runtime latency and throughput on one kernel and device.

The first two do not imply the last two. A NumPy int8 container holding logical int4 values
does not consume int4 memory until code actually packs it.

## 11. Memory follows retained state

For mixed-precision Adam training, a rough per-parameter account may include:

```text
fp16 or bf16 weight       2 bytes
gradient                  2 bytes
fp32 master weight        4 bytes
first Adam moment         4 bytes
second Adam moment        4 bytes
```

That is 16 bytes per parameter before saved activations, temporary workspaces, allocator
behavior, or distributed copies. Optimizer choice and sharding can change the account.

Autoregressive inference often retains lower-precision weights, current activations, and
keys and values for cached tokens. A basic KV element count is proportional to batch,
layers, cached tokens, heads, and head width, with a factor of two for keys and values.
Architectures with grouped-query or multi-query attention change that formula.

A capacity decision needs named components and scopes. Ask:

- Does the figure cover one request, one replica, or a whole host?
- Is the sequence length prompt-only or prompt plus generated tokens?
- Does KV memory include both key and value tensors across every layer?
- Are allocator reserve and temporary kernels measured or omitted?
- Is the number before or after tensor parallel sharding?

The course formulas answer none of those silently. Callers provide derived counts, and the
result labels each byte component.

## 12. Reading the capstone correctly

The cyclic dataset defines one learnable rule: each next token is the current token plus
one modulo vocabulary size. Rows differ by starting token. The split uses starts 0 through
7 for training, 8 and 9 for calibration, and 10 and 11 for testing.

This design avoids an earlier bad corpus idea. If a row alternates arbitrary token pairs,
the first target is unknowable from the first input token. That places a hard ceiling under
accuracy and a floor under loss. A requirement above that ceiling would be fake rigor.

### The calibration step does not work here, and says so

The trained model gets every calibration row right. By the argument above, that makes
the temperature fit unresolved: the grid floor of 0.5 wins, and widening the grid
downward drives held-out ECE to zero without learning anything. The report records
`temperature_on_grid_boundary` and prints `grid floor, unresolved` beside the number.

Leaving this in is deliberate. A tiny deterministic task that the model masters is
exactly the situation where calibration has nothing to measure, and that is worth seeing
once. Removing the step would hide the mechanism; presenting 0.5 as a fitted temperature
would be the kind of quiet overclaim the rest of this chapter argues against.

The experiment stops after 20 AdamW steps. More steps drove training loss closer to numeric
zero but increased quantized-logit drift without changing held-out accuracy. Lower training
loss was not free.

### Independent requirements

The report declares bounds before producing observations:

```text
maximum final training loss       0.20
minimum held-out accuracy         0.95
maximum held-out ECE              0.15
maximum mean int8 logit drift     0.03
```

The verdict validates finite measurements and three distinct split identifiers first. It
then checks training, calibration, and quantization in a fixed order. Equality passes.

Tests mutate each observation across its requirement and change policy while holding
observations fixed. That matters. A check whose expected answer came from the same result it
judges would keep agreeing with its own bug.

### What the verdict does not say

`ready_for_lab_use` is intentionally narrow. It does not say:

- the model understands language;
- the cyclic task represents any production distribution;
- int8 improves runtime speed;
- the memory estimate proves a device fit;
- one ECE value establishes calibration for every class or cohort.

The capstone is valuable because every claim names its evidence and boundary. Its model is
deliberately useless outside the lesson.

## 13. A review checklist for model-facing code

Before accepting a model mechanic or an experiment result, ask:

1. What does each axis mean, and where is that checked?
2. Are logits passed to a stable combined loss?
3. Does a gradient check use a sensible dtype and epsilon?
4. Does training name convergence, budget exhaustion, and numeric failure separately?
5. Does a causal counterfactual prove future information cannot leak backward?
6. Are sampling policy and random state explicit inputs?
7. Are model fitting, calibration fitting, and final judgment on separate evidence?
   Did the fitted value land inside its search grid or on an endpoint?
8. Does quantization report reconstruction or task drift apart from payload bytes?
9. Does a memory figure name every retained component and its scope?
10. Can changing an independent requirement flip the final verdict?

If the answer to the last question is no, inspect the check for circular evidence.

## Source notes

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
- Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
- PyTorch, [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- PyTorch, [`MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- PyTorch, [transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- PyTorch, [installation selector](https://docs.pytorch.org/get-started/locally/)
- NumPy, [2.5 release notes](https://numpy.org/doc/stable/release/2.5.0-notes.html)
