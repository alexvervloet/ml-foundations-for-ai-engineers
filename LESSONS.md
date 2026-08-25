# Lessons learned

## Keep patch wrappers separate from lesson markup

- **Expected:** One patch call would create the first four concept modules.
- **Actual:** Markdown backticks inside Python docstrings terminated the JavaScript
  template used to pass the patch. The wrapper failed before it changed any file.
- **Next time:** Split large patches by file and avoid an interpolation-sensitive
  wrapper for text that contains its delimiter.

## Do not combine environment creation with a network upgrade

- **Expected:** Creating the virtual environment and upgrading pip would finish as one
  setup step.
- **Actual:** The environment was created, but sandboxed DNS blocked the index lookup.
  Pip already matched the requested package and exited zero after five noisy retries.
- **Next time:** Create the environment with no network command attached. Run required
  downloads separately so a network failure has an unambiguous exit status and can be
  retried with the narrow package-install approval.

## Build tensors from the array, not a list around it

- **Expected:** The cross-entropy example would print only its five measured results.
- **Actual:** Wrapping a NumPy array in a Python list before calling `torch.tensor`
  triggered PyTorch's slow-construction warning.
- **Next time:** Add the batch dimension in NumPy, then use `torch.from_numpy`. Run
  every example alone and treat warnings as failures before its first commit.

## Use terminal-state names exactly as the code defines them

- **Expected:** The large learning-rate example would demonstrate divergence.
- **Actual:** Its finite loss grew until the budget ended, so the optimizer correctly
  returned `MAX_STEPS`. The prediction used "diverges" in the looser mathematical
  sense and contradicted the program's explicit status.
- **Next time:** Write the prediction with the decision table open. Compare its nouns
  and verbs against the observed enum and printed output before committing the file.

## Resolve the Python floor from the complete dependency set

- **Expected:** Python 3.11 would match the neighboring courses while current NumPy
  and PyTorch pins supplied the tensor runtime.
- **Actual:** NumPy 2.5 dropped Python 3.11. The package metadata promised a runtime
  that its own dependency could not install on.
- **Next time:** Check every pinned package's Python classifiers before choosing the
  course floor. Make the CI minimum cell match the intersection, not a house default.

## Match unittest discovery to the test directory shape

- **Expected:** Supplying the repository as `top_level_dir` would make setup discovery
  match the command-line test run.
- **Actual:** `unittest` then required the plain `tests` directory to be an importable
  package and stopped before counting tests.
- **Next time:** Either add `tests/__init__.py` deliberately or omit `top_level_dir`.
  Copy the exact verified discovery call into setup and CI instead of approximating it.

## A scaling relation needs a nonzero anchor

- **Expected:** The test that doubled KV elements would kill an implementation that
  dropped the KV cache from memory accounting.
- **Actual:** Both zero-byte results still satisfied `long == 2 * short`, so the
  mutation survived.
- **Next time:** Pair metamorphic scaling checks with one independently calculated
  baseline. The KV test now requires 40 fp16 elements to consume exactly 80 bytes.

## Measure bytes before the convenience conversion

- **Expected:** `quantize_symmetric` reported the source size of whatever the caller
  handed it, so the capstone's payload comparison was a fair one.
- **Actual:** The function converted its input to float64 on the first line and then
  read `nbytes` from that copy. Every float32 caller was told it held twice the bytes
  it really did. The capstone advertised 45504 source bytes against a 5824-byte int8
  payload when the model's fp32 weights occupy 22752, turning a 3.9x reduction into a
  claimed 7.8x. The unit tests missed it because they all passed float64 arrays, where
  the wrong reading and the right one are identical.
- **Next time:** When a function reports a property of its input, read that property
  before any internal conversion. Test the accounting with a dtype that differs from
  the working dtype, or the test cannot tell the two apart.

## A separable calibration split cannot choose a temperature

- **Expected:** Fitting temperature on a held-out calibration split would demonstrate
  calibration, with the reported ECE drop as the result.
- **Actual:** The trained model classifies every calibration row correctly, so
  cross-entropy falls monotonically as temperature approaches zero and the smallest
  grid candidate always wins. The reported `T=0.50` was the grid floor, not a fitted
  value, and extending the grid downward drives held-out ECE to 0.0 while meaning
  nothing. The lesson's own example had the mirror-image problem: its winner sat on
  the top end of the grid, which happened to be a real minimum, but the grid could not
  have shown that.
- **Next time:** Check whether a grid-search winner sits on an endpoint before
  reporting it. `fit_temperature` now returns `on_grid_boundary`, the example grid
  extends past its winner, and the capstone prints `grid floor, unresolved` instead of
  presenting a truncation artifact as a calibration result.

## Match the series chapter title convention

- **Expected:** A descriptive TEXTBOOK.md title was enough, since the parent series
  table supplies the chapter number and name.
- **Actual:** Every other dive titles the file `# Chapter N: <Title>`. This one used a
  bare name, so the parent linked readers to chapter 24, "The Numeric Contracts
  Beneath the Model", and they landed on a page whose heading said something else.
- **Next time:** Copy the heading straight from the parent TEXTBOOK.md row when
  creating a dive, before writing any body text.

## A conversion can swallow the value a type check was meant to catch

- **Expected:** Adding an explicit Boolean guard to each temperature parameter closed
  the gap where `True` was read as a temperature of 1.0.
- **Actual:** It closed two of the three entry points. `fit_temperature` normalizes its
  candidate grid with `float(value)` before validating anything, so `(True, 2.0)`
  became `(1.0, 2.0)` and passed every later check. A guard placed after a conversion
  cannot see what the conversion consumed. The same review also found that
  `isinstance(value, bool)` misses `numpy.bool_`, which matters in a course whose
  arrays are all NumPy.
- **Next time:** Validate types at the boundary, before any normalization step, and
  when a fix applies to a family of parameters, grep for every place that family is
  accepted rather than fixing the ones that came to mind.

## Do not identify a NumPy type by its name

- **Expected:** Checking `type(value).__name__ == "bool_"` would catch `numpy.bool_`
  in the scalar optimizer module without importing NumPy into it.
- **Actual:** NumPy 2 renamed the type, so `type(np.True_).__name__` is now `"bool"`
  and the guard matched nothing. The test caught it, but only because it checked
  `np.True_` alongside the Python `True`.
- **Next time:** Identify a type with `isinstance`, not its name. Keeping one module
  import-free was not worth a check that silently changes meaning with a dependency
  release. `optimization.py` now imports NumPy for the same guard the other modules
  use.

