# Lab thực hành 2 — DS201

**Tên:** Trần Minh Phát

Phân loại ảnh món ăn VinaFood21 bằng LeNet-5, GoogLeNet, ResNet-18 và ResNet-50 pretrained.

## Cây thư mục

```text
DS201_Lab2/
├── configs/                    # Cấu hình Kaggle của từng bài
│   ├── lenet_64.yaml           # Bài 1: LeNet-5, ảnh 64×64
│   ├── googlenet_64.yaml       # Bài 2: GoogLeNet, ảnh 64×64
│   ├── resnet18_96.yaml        # Bài 3: ResNet-18, ảnh 96×96
│   └── resnet50_224.yaml       # Bài 4: ResNet-50 pretrained, ảnh 224×224
├── notebooks/
│   └── assignments.ipynb       # Notebook chạy chính, giữ hình minh họa của đề
├── scripts/                    # Các lệnh chuẩn bị, train, đánh giá và đóng gói report
├── src/                        # Code xử lý dữ liệu, model, training và reporting
├── data/                       # Thư mục dữ liệu tạo cục bộ (không đưa lên Git)
├── report/                     # Report tạo cục bộ (không đưa lên Git)
├── requirements.txt            # Thư viện cần cài
└── README.md
```

## Hướng dẫn chạy trên Kaggle

1. Mở `notebooks/assignments.ipynb`, bật GPU T4 và chạy các cell chuẩn bị chung ở đầu notebook.
2. Chạy bài cần train:

```python
run_task('lenet_64')
run_task('googlenet_64')
run_task('resnet18_96')
run_task('resnet50_224')
```

3. Sau khi train xong, chạy block xuất report cuối notebook:

```python
SELECTED = ['lenet_64', 'googlenet_64', 'resnet18_96', 'resnet50_224']
REPO = '/kaggle/working/DS201_Lab2'

for name in SELECTED:
    config = f'{REPO}/configs/{name}.yaml'
    checkpoint = f'/kaggle/working/runs/{name}_seed42/best.pt'
    output_dir = f'/kaggle/working/report/03_evaluation/{name}_seed42'
    !python {REPO}/scripts/evaluate_checkpoint.py --config {config} --checkpoint {checkpoint} --output-dir {output_dir}

!python {REPO}/scripts/package_report.py --report-dir /kaggle/working/report --zip-path /kaggle/working/report.zip
```

Cell đầu notebook tự pull code mới từ GitHub vào Kaggle session. Report cuối nằm tại `/kaggle/working/report.zip`.
