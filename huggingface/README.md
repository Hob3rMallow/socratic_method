# Hugging Face export templates

`socratic-export` fills the model-card template that sits beside the recipe it
exports: `huggingface/<recipe directory>/README.md`. Each directory also keeps
example `config.json` and `preprocessor_config.json` files showing what the
exporter writes for that recipe.

- `f0/`: the current frozen model, C3-F0 at 200,000 exposures, shipped with
  eight-way mirror TTA at T=0.30 (`recipes/f0/recipe.json`; inference evidence in
  `recipes/f0_inference_20260909/`).
- `v31/`: the historical v31 8,192-sample model, T=0.45
  (`recipes/v31/recipe.json`).
- `dataset/`: dataset-card scaffold for the replay bundles.

Generated bundles land in `huggingface/export/` (gitignored). Weights are never
committed to Git, and publishing remains a separate, explicit `hf upload`
action that still requires a license and resolution of the upstream artifact
terms.
