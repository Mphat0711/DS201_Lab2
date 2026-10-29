# Lab thực hành 2 — DS201

**Tên:** Trần Minh Phát

**MSSV:** 24521317

Phân loại ảnh món ăn VinaFood21 bằng LeNet-5, GoogLeNet, ResNet-18 và ResNet-50 pretrained.

## Cây thư mục

```text
DS201_Lab2/
├── configs/                    # Cấu hình Kaggle của từng bài
│   ├── lenet_64_avg/max.yaml   # Bài 1: AvgPool và MaxPool, ảnh 64×64
│   ├── googlenet_64.yaml       # Bài 2: GoogLeNet, ảnh 64×64
│   ├── resnet18_96_noaug/aug.yaml # Bài 3: không/có augmentation, ảnh 96×96
│   └── resnet50_224_feature/finetune.yaml # Bài 4: feature extraction/fine-tuning
├── notebooks/
│   └── assignments.ipynb       # Notebook chạy chính, giữ hình minh họa của đề
├── scripts/                    # prepare.py, train.py, evaluate.py, package_report.py
├── src/                        # data.py, model.py, engine.py và các helper cần thiết
├── data/                       # Thư mục dữ liệu tạo cục bộ (không đưa lên Git)
├── report/                     # Report tạo cục bộ (không đưa lên Git)
├── requirements.txt            # Thư viện cần cài
└── README.md
```

## Hướng dẫn

1. Mở `notebooks/assignments.ipynb`, chạy các cell chuẩn bị chung ở đầu notebook.
2. Chạy cell `run_all()` dưới Bài 1.

```python
run_all()
```

3. Sau khi train xong, chạy block report cuối notebook:

```python
# Chạy toàn bộ cell "Evaluation - Report"; cell dùng chính danh sách RUNS.
```

Cell đầu notebook pull code mới từ GitHub vào Kaggle session.
