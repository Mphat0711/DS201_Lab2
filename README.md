# DS211 Lab 2 — reproducible CNN experiments

The project is designed for Kaggle/Colab notebooks that orchestrate code from this repository. Modules in `src/` are atomic: every data/output path is supplied by the caller via config or explicit function argument.

## Workflow

1. Select the explicit Kaggle config for the assignment task: `lenet_64.yaml`, `googlenet_64.yaml`, `resnet18_96.yaml`, or `resnet50_224.yaml`.
2. Freeze the stratified split once, generate EDA/report artifacts, then materialize a required image size.
3. Run a no-training smoke test before launching any training config.
4. Train configs sequentially for their full configured epoch count; save the checkpoint with the lowest validation loss and release VRAM after each run.
5. Evaluate all saved best checkpoints only after training ends, then package the report.

For Kaggle, use `notebooks/assignments.ipynb`. Its sections mirror Bài 1–4 and call the versioned YAML files directly.

`prepare.py --materialize` creates every resized image named by the frozen manifest. It is intentionally an explicit operation and can take time/disk space.

## Kaggle convention

- Original data: `/kaggle/input/<dataset>/VinaFood21` (read-only)
- Generated resized data/metadata: `/kaggle/working/data/`
- Runs: `/kaggle/working/runs/`
- Report and ZIP: `/kaggle/working/report/` and `/kaggle/working/report.zip`

Set these only in the notebook/config passed into the scripts; do not hard-code them in `src/`.

## Config on Kaggle

The four versioned YAML files already contain the agreed Kaggle paths. If the mounted input slug changes, edit only `paths.source_root` in those YAML files and commit before cloning. `train_config.py` copies the exact YAML to `<run_dir>/config.yaml`, so each checkpoint keeps its immutable configuration snapshot.
