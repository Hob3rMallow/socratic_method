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

Generated bundles land in `huggingface/export/` (gitignored). The frozen F0
checkpoint is committed to this repository under `releases/` via Git LFS and is
Apache-2.0 licensed; pushing a bundle to the Hugging Face Hub remains a
separate, explicit `hf upload` action. The upstream data, teacher and M7 terms
still govern anything rebuilt from those artifacts - see the repository
[NOTICE](../NOTICE).
