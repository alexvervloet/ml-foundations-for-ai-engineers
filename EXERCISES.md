# Exercises

Write a prediction before each run. After the change, explain the output using shapes,
units, split roles, policy, or retained state. Do not copy the observed answer into the
check that judges it.

## 1. Vectors and shapes

Run:

```bash
python examples/01_vectors.py
```

1. Scale `query` by a negative value. Predict the cosine before running.
2. Make the affine weight input width four while leaving the batch width three. Read the
   validation error and write the failed shape equation.
3. Try a scalar bias and a `(2, 1)` bias. Explain why NumPy could broadcast some shapes
   that the course rejects.
4. Name the axes of a real embedding batch in one application you maintain.

## 2. Stable softmax

Run:

```bash
python examples/02_softmax.py
```

1. Add one million to every logit and check shift invariance.
2. Multiply every logit by two. Predict which probability moves farthest.
3. Implement direct exponentiation in the example only. Observe the warning or nonfinite
   result on large logits, then remove the broken version.
4. Explain why a 0.94 softmax value does not establish 94 percent correctness.

## 3. Cross-entropy

Run:

```bash
python examples/03_cross_entropy.py
```

1. Raise a non-target logit instead. Predict the target probability and loss direction.
2. Add the same constant to all logits and verify the loss is unchanged.
3. Pass probability values to PyTorch cross-entropy as if they were logits. Compare the
   number without claiming either input contract is equivalent.
4. Add label smoothing with PyTorch and state which objective changed.

## 4. Gradient checks

Run:

```bash
python examples/04_gradients.py
```

1. Try finite-difference epsilon values `1e-2`, `1e-6`, and `1e-8`. Compare the gaps.
2. Add a cubic term and derive its analytic gradient before editing the code.
3. Detach the tensor before the loss. Explain why autograd can no longer supply evidence.
4. Design a gradient check for one parameter of a two-layer PyTorch module.

## 5. Gradient descent

Run:

```bash
python examples/05_gradient_descent.py
```

1. Set a moderate rate with a budget of one step. Confirm the terminal status is not
   convergence.
2. Find a learning rate that reaches a nonfinite value under a much larger initial
   parameter. Keep `max_steps` finite.
3. Put the gradient exactly on the tolerance boundary and confirm equality passes.
4. Add a trace printer that shows whether each update moves toward or away from the
   known minimum. Derive that label from consecutive points.

## 6. Attention

Run:

```bash
python examples/06_attention.py
```

1. Remove the square-root scale temporarily and compare the last-row weights.
2. Change the final key instead of the final value. Confirm earlier causal outputs still
   do not move.
3. Pass a mask with one fully forbidden row and explain why returning zeros would hide an
   invalid attention relationship.
4. Build a padding mask for two sequences of unequal length. State how it differs from a
   causal mask.

## 7. Transformer block

Run:

```bash
python examples/07_transformer_block.py
```

1. Construct width ten with three heads and predict where validation fails.
2. Set `causal=False`, change the last token, and observe whether earlier outputs move.
3. Zero every learned parameter. Confirm the residual path becomes an identity map.
4. Add a second block and trace the shape at the input and output of each block.

## 8. Sampling

Run:

```bash
python examples/08_sampling.py
```

1. Draw 1,000 tokens at temperatures 0.5 and 2.0 with fresh generators. Compare counts.
2. Set `top_k=1` and explain its relationship to greedy selection, including ties.
3. Reuse one generator for two calls, then recreate it from the seed for a third. Predict
   which sequences match.
4. Write a policy table for temperature, top-p, repetition penalty, and a fixed random
   seed in one application.

## 9. Calibration

Run:

```bash
python examples/09_calibration.py
```

1. Verify argmax predictions stay unchanged at every candidate temperature.
2. Fit and evaluate on the same rows. Explain why the number is training evidence for the
   calibrator, not held-out evidence.
3. Change from two to four ECE bins. Record the changed metric and its unchanged inputs.
4. Construct an underconfident calibration set that chooses a temperature below one.
5. Drop `8.0` from the grid so `4.0` becomes the largest candidate. Watch
   `on_grid_boundary` flip and explain what the fitted number now fails to establish.
6. Build a calibration set the model classifies perfectly. Extend the grid downward
   until held-out ECE reaches zero, then argue why that number is worthless.

## 10. Quantization

Run:

```bash
python examples/10_quantization.py
```

1. Add one large outlier and compare int8 reconstruction error for the other values.
2. Quantize an all-zero tensor and explain the scale convention.
3. Pack two logical int4 values into each byte. Test odd and even element counts.
4. Benchmarking is outside the current example. Write the device, kernel, batch, warmup,
   and repetition fields a valid latency result would need.

## 11. Training and inference memory

Run:

```bash
python examples/11_training_vs_inference_memory.py
```

1. Hand-calculate each training component and the total before changing the example.
2. Double parameters while holding activation and KV element counts fixed. Identify which
   components double.
3. Model gradient checkpointing by reducing saved activation elements. State which extra
   compute the byte estimate omits.
4. Derive KV elements for a declared batch, layer count, token count, head count, and head
   width. Include the key and value factor explicitly.

## 12. Capstone

Run:

```bash
python hands_on/train_tiny_transformer.py
```

1. Read `training-report.json` and map every observation to the requirement that consumes
   it. Split identifiers are validity evidence, not quality thresholds.
2. Set `maximum_mean_logit_drift` exactly to the observed drift and confirm equality
   passes. Lower it by `1e-6` and confirm the verdict changes.
3. Reuse the calibration split identifier as the test identifier. Confirm the report is
   invalid before quality checks run.
4. Train for 40 steps while leaving requirements fixed. Measure loss, accuracy, ECE, and
   int8 drift. Explain why the extra optimization is not automatically an improvement.
5. Replace cyclic rotations with arbitrary alternating pairs. Calculate the share of
   targets that cannot be inferred from the available prefix before running anything.
6. Add quantized test accuracy as a new observation and a separately declared minimum.
   Write the failing counterfactual test before changing the verdict function.
7. The report marks its temperature fit unresolved. Change the calibration split to rows
   the model gets wrong, then state whether the fit becomes interior and what that costs
   the held-out accuracy requirement.

## Review exercise

Pick one model claim in another repository. Write four lines:

```text
claim:
independent requirement:
observed evidence:
validity boundary:
```

Then change only the requirement. If the conclusion cannot change, the check may be
circular or the requirement may not be connected to the decision.
