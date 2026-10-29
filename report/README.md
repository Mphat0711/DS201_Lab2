# Hướng dẫn đọc kết quả

- `00_eda/`: phân bố nhãn và mapping class-to-index.
- `01_preprocessing_split/`: split cố định và thông tin tiền xử lý.
- `02_training/<run>/`: `training_curves.png`, history, `model_summary.json` (tham số, padding, augmentation) và `layer_summary.csv` khi bài yêu cầu.
- `03_evaluation/<run>/`: bốn metric test, classification report và confusion matrix.
- `tables/model_comparison.csv`: bảng tổng hợp resize, tham số, giây/epoch, accuracy và macro-F1.

`divergence_epoch` là epoch đầu tiên trong hai epoch liên tiếp có `val_loss > train_loss`; dùng để nhận xét overfitting.
