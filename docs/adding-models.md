# Adding the first model

No model is included yet. When the protocol is ready:

1. Create one module under `src/aisource/models/`.
2. Implement `BaseGenerator.fit` and `BaseGenerator.sample`.
3. Add the model configuration to a copied benchmark YAML file.
4. Connect the model to `benchmark.py`.
5. Add a small synthetic test.
6. Document dependencies and checkpoint format.

Do not add placeholder model names before implementations exist.
