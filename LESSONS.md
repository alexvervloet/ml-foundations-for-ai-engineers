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
