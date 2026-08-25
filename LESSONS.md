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
