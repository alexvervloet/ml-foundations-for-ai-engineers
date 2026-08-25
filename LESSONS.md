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
