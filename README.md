# DS211 Lab 2 — reproducible CNN experiments

The project is designed for Kaggle/Colab notebooks that orchestrate code from this repository. Modules in `src/` are atomic: every data/output path is supplied by the caller via config or explicit function argument.

## Workflow

1. Set paths in a config. `configs/local.yaml` is local-only; do not commit it. Use `configs/local.example.yaml` as a portable template.
2. Freeze the stratified split once, generate EDA/report artifacts, then materialize a required image size.
3. Run a no-training smoke test before launching any training config.
4. Train configs sequentially, saving best checkpoints; release VRAM after each run.
5. Evaluate all saved best checkpoints only after training ends, then package the report.

```powershell
python scripts/prepare.py --config configs/local.yaml --create-split
python scripts/smoke_test.py --config configs/local.yaml
python scripts/prepare.py --config configs/local.yaml --materialize
python scripts/smoke_test.py --config configs/local.yaml --use-materialized
python scripts/package_report.py --report-dir report --zip-path report.zip
```

`prepare.py --materialize` creates every resized image named by the frozen manifest. It is intentionally an explicit operation and can take time/disk space.

## Kaggle convention

- Original data: `/kaggle/input/<dataset>/VinaFood21` (read-only)
- Generated resized data/metadata: `/kaggle/working/data/`
- Runs: `/kaggle/working/runs/`
- Report and ZIP: `/kaggle/working/report/` and `/kaggle/working/report.zip`

Set these only in the notebook/config passed into the scripts; do not hard-code them in `src/`.
