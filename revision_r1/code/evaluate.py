import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_ece(y_true, y_prob, n_bins=15):
    """
    Expected Calibration Error.

    다중분류(단일 라벨)에서는 관례대로 **top-1 신뢰도 기준**으로 계산한다:
    예측 확률의 최댓값을 신뢰도로 보고 신뢰도 구간별로 (정확도 - 평균신뢰도)의
    절댓값을 표본 비율로 가중합한다.

        ECE = sum_b (|B_b| / N) * | acc(B_b) - conf(B_b) |

    멀티라벨(y_true 가 2차원 지시행렬)일 때만 클래스별 one-vs-rest 로 계산하고,
    각 구간의 acc 는 **양성 실제 빈도**를 쓴다.

    [이전 구현의 버그] one-vs-rest 구간에서 acc 를 `y_true == (p > 0.5)` 즉
    0.5 임계값과의 일치율로 계산했다. CIFAR-100 처럼 클래스가 많으면 대부분의
    표본이 최저 구간에 몰리고 거기서 acc ~ 1 - 1/K, conf ~ 0 이 되어 ECE 가
    모델과 무관하게 (1 - 1/K) 로 고정된다. 실제로 전 런에서 0.98 이 나왔다.
    """
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    if y_true.ndim == 1:
        # ---- 단일 라벨 다중분류: top-1 신뢰도 기준 ----
        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)
        return _binned_ece(conf, correct, n_bins)

    # ---- 멀티라벨: 클래스별 one-vs-rest, acc = 양성 실제 빈도 ----
    eces = []
    for c in range(y_true.shape[1]):
        eces.append(_binned_ece(y_prob[:, c], y_true[:, c].astype(float), n_bins))
    return float(np.mean(eces))


def _binned_ece(conf, correct, n_bins=15):
    """conf(신뢰도)와 correct(0/1 정답 여부)를 구간별로 묶어 ECE 를 계산한다."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += np.abs(correct[mask].mean() - conf[mask].mean()) * (mask.sum() / n)
    return float(ece)


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            is_multilabel = target.ndim > 1 and target.size(1) > 1

            if is_multilabel:
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(target.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    labels_np = all_labels.numpy()
    preds_np = all_preds.numpy()
    probs_np = all_probs.numpy()

    is_multilabel = len(labels_np.shape) == 2 and labels_np.shape[1] > 1

    if is_multilabel:
        acc = (all_preds == all_labels).all(dim=1).float().mean().item() * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro')
        f1_micro = f1_score(labels_np, preds_np, average='micro')
        auc = roc_auc_score(labels_np, probs_np)
    else:
        acc = accuracy_score(labels_np, preds_np) * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro')
        f1_micro = f1_score(labels_np, preds_np, average='micro')
        # 이진 분류(waterbirds)는 양성 확률 1열만 넘겨야 한다 — (N,2) 를 주면 sklearn 이 죽는다.
        if probs_np.ndim == 2 and probs_np.shape[1] == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            auc = roc_auc_score(labels_np, probs_np, multi_class='ovr')

    ece = compute_ece(labels_np, probs_np)

    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc': auc,
        'ece': ece
    }
