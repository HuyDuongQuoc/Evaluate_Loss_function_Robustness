# Đánh giá độ bền vững của hàm mất mát dưới nhiễu nhãn

Dự án xây dựng một benchmark PyTorch có kiểm soát để khảo sát ảnh hưởng của hàm mất mát đến hiệu năng và độ bền vững của mô hình khi **nhãn huấn luyện chứa nhiễu**. Hai bài toán được nghiên cứu độc lập:

- **Hồi quy:** so sánh MSE, MAE và Huber dưới Gaussian target noise.
- **Phân loại:** so sánh Cross-Entropy (CE), Generalized Cross-Entropy (GCE) và Symmetric Cross-Entropy (SCE) dưới symmetric label noise.

Mục tiêu của dự án không phải đề xuất một loss mới, mà là so sánh công bằng các loss hiện có và làm rõ sự đánh đổi giữa hiệu năng trên dữ liệu sạch, hiệu năng dưới nhiễu và độ ổn định giữa các lần chạy.

## Câu hỏi nghiên cứu

1. MSE, MAE và Huber thay đổi như thế nào khi tỷ lệ target bị nhiễu tăng từ 0% lên 10% và 20%?
2. CE, GCE và SCE thay đổi như thế nào khi tỷ lệ nhãn phân loại bị nhiễu tăng từ 0% lên 20% và 40%?
3. Loss tốt nhất trên dữ liệu sạch có còn là loss bền vững nhất khi nhiễu tăng hay không?

## Thiết kế thí nghiệm

| Thành phần | Hồi quy | Phân loại |
|---|---|---|
| Dữ liệu | `make_regression` | `make_classification` |
| Số mẫu / đặc trưng | 5.000 / 20 | 5.000 / 20 |
| Đầu ra | Giá trị liên tục | 3 lớp cân bằng |
| Mô hình | MLP `20 → 64 → 32 → 1` | MLP `20 → 64 → 32 → 3` |
| Loss | MSE, MAE, Huber | CE, GCE, SCE |
| Mức nhiễu | 0%, 10%, 20% | 0%, 20%, 40% |
| Metric | MAE, RMSE | Accuracy, Macro-F1, confusion matrix |
| Early stopping | Validation MAE | Validation Macro-F1 |

Mỗi tổ hợp được lặp lại với ba seed `42`, `123` và `2026`:

```text
2 task × 3 loss × 3 mức nhiễu × 3 seed = 54 main-study runs
```

Các nguyên tắc kiểm soát chính:

- Dữ liệu được sinh sạch trước, sau đó chia train/validation/test theo tỷ lệ 70/15/15.
- Chỉ nhãn của tập train bị gây nhiễu; validation và test luôn giữ sạch.
- Với cùng task, seed và mức nhiễu, các loss dùng cùng dữ liệu, noise mask, nhãn đã corrupt, trọng số khởi tạo và seed của DataLoader.
- Kiến trúc, optimizer Adam, learning rate `0.001`, batch size `64` và tối đa `300` epoch được giữ cố định.
- Checkpoint tốt nhất được chọn bằng validation set. Test set chỉ được dùng khi đánh giá cuối cùng.
- Robustness được tính theo paired seed, sau đó mới tổng hợp mean và sample standard deviation.

Siêu tham số đặc thù của loss đã được chọn bằng validation set và khóa trong cấu hình chính:

- Huber: `delta = 0.1 × std(y_train)`.
- GCE: `q = 0.5`.
- SCE: `alpha = 0.1`, `beta = 0.1`.

## Cấu trúc dự án

```text
.
├── configs/             # Cấu hình dữ liệu, loss, training và output
├── notebooks/           # Notebook kiểm tra dữ liệu và phân tích kết quả
├── results/             # Logs, checkpoints, metrics, noise archives và figures
├── scripts/             # Entry point chạy, tuning và hậu xử lý thí nghiệm
├── src/
│   ├── datasets/        # Sinh và chia dữ liệu tổng hợp
│   ├── experiments/     # Điều phối run và lưu artefact
│   ├── losses/          # Các loss hồi quy và phân loại
│   ├── models/          # MLP cho hai task
│   ├── noise/           # Gaussian target noise và symmetric label noise
│   └── training/        # DataLoader, training loop và evaluator
├── requirements.txt
└── README.md
```

## Yêu cầu môi trường

- Python 3.13 đã được dùng để xác minh repository này. Đây là phiên bản đã kiểm thử, không phải tuyên bố về phiên bản tối thiểu.
- Dependency được khóa trong `requirements.txt`.
- Không cần tải dataset bên ngoài; cả hai dataset đều được sinh trong code.
- Nên chạy mọi lệnh dưới đây từ thư mục gốc của repository.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS (Bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tái lập thí nghiệm

### 1. Kiểm tra nhanh loss và gradient

```bash
python -m scripts.loss_function_check
```

Lệnh này kiểm tra shape đầu ra, giá trị loss và gradient hữu hạn cho cả sáu loss mà không chạy toàn bộ thí nghiệm.

### 2. Chạy thử một điều kiện

```bash
python -m scripts.run_regression --mode single --seed 42 --noise 0.1
python -m scripts.run_classification --mode single --seed 42 --noise 0.2
```

Chế độ `single` phù hợp để kiểm tra pipeline bằng validation metrics trước khi chạy ma trận đầy đủ. Có thể thay `--seed`, `--noise` hoặc cung cấp config khác qua `--config`.

### 3. Chạy đủ 54 main-study runs

```bash
python -m scripts.run_regression --mode full --final
python -m scripts.run_classification --mode full --final
```

Flag `--final` là bắt buộc khi tái lập kết quả báo cáo: nó kích hoạt đánh giá trên clean test set và ghi test metrics vào JSON logs. Script tạo master result sẽ bỏ qua log không có final test evaluation.

Hai lệnh trên lần lượt chạy:

- 27 regression runs: 3 loss × 3 noise levels × 3 seed.
- 27 classification runs: 3 loss × 3 noise levels × 3 seed.

> **Lưu ý:** tên artefact được xác định bởi `task + loss + noise level + seed`. Chạy lại cùng tổ hợp sẽ ghi lại log, checkpoint, noise archive hoặc CSV có cùng tên. Nếu cần giữ một lần chạy cũ, hãy sao lưu thư mục `results/` trước khi chạy lại.

### 4. Tổng hợp metrics và tính robustness

Chạy theo đúng thứ tự:

```bash
python -m scripts.create_master_result
python -m scripts.create_summary
python -m scripts.calculate_robustness
```

Các bước này:

1. Lọc đúng các final logs thuộc 54 main-study combinations và tạo `results/metrics/results.csv`.
2. Tổng hợp mean ± sample standard deviation cho 18 nhóm task–loss–noise vào `results/metrics/summary.csv`.
3. Ghép clean/noisy result theo cùng loss và seed để tạo 36 paired comparisons và 12 nhóm robustness.

Robustness được định nghĩa như sau:

```text
Regression:     ΔMAE = MAE(noisy) - MAE(clean)
Classification: ΔAccuracy = Accuracy(clean) - Accuracy(noisy)
```

Delta nhỏ hơn biểu thị mức suy giảm thấp hơn, nhưng phải được đọc cùng hiệu năng tuyệt đối.

### 5. Tạo sơ đồ từ kết quả thí nghiệm

```bash
python -m scripts.create_report_figures
python -m scripts.create_robustness_figures
```

Mặc định, figures được xuất ở cả PNG và PDF vào `results/figures/`. Có thể xem toàn bộ tùy chọn bằng `--help`, ví dụ:

```bash
python -m scripts.create_report_figures --help
```

### 6. Tuning lại loss hyperparameters (tùy chọn)

Main study sử dụng các giá trị đã khóa trong `configs/regression.yaml` và `configs/classification.yaml`. Chỉ chạy lại tuning khi muốn kiểm tra quá trình lựa chọn tham số:

```bash
python -m scripts.tune_regression
python -m scripts.tune_classification
```

Tuning dùng validation performance tại 10% regression noise và 20% classification noise; test set không tham gia lựa chọn. Các script tuning ghi kết quả vào `results/tuning/` nhưng không tự sửa config. Các lượt tuning không thuộc 54 main-study runs.

## Output mong đợi

Sau một lần chạy đầy đủ và hậu xử lý thành công, các đầu ra chính gồm:

```text
results/
├── checkpoints/                         # 54 model checkpoints
├── logs/                                # 54 JSON logs
├── noise/                               # Clean/noisy labels và noise mask theo task/seed/noise
├── metrics/
│   ├── results.csv                      # 54 run-level rows
│   ├── summary.csv                      # 18 nhóm task–loss–noise
│   ├── robustness_results.csv           # 36 paired-seed rows
│   └── robustness_summary.csv           # 12 nhóm robustness
└── figures/                             # Performance, robustness, confusion matrix, learning curves
```

