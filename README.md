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

## Runtime YAML on Kaggle

After cloning, YAML files are normal writable files under `/kaggle/working/<repo>/configs/`.
Keep versioned templates in `configs/*.example.yaml`; create the actual per-session configuration in `configs/runtime/` from a notebook cell. That directory is intentionally ignored by Git. `train_config.py` copies the exact runtime YAML to `<run_dir>/config.yaml`, so each checkpoint keeps its immutable configuration snapshot.

```python
from pathlib import Path
import yaml

template = Path("configs/resnet50_224.example.yaml")
runtime = Path("configs/runtime/resnet50_224_t4.yaml")
runtime.parent.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(template.read_text())
cfg["paths"]["source_root"] = "/kaggle/input/<dataset>/VinaFood21"
cfg["paths"]["resized_root"] = "/kaggle/working/data/resized_224"
cfg["paths"]["resized_manifest"] = "/kaggle/working/data/resized_224/manifest.csv"
cfg["paths"]["run_dir"] = "/kaggle/working/runs/resnet50_224_seed42"
runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
```
