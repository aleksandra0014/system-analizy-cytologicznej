import math
import numpy as np
from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import os

CLASS_NAMES = ["cell", "HSIL_group"]   # 0 -> cell, 1 -> HSIL_group
BG_NAME = "background"
EPS = 1e-9

from typing import List, Tuple, Optional, Any, Dict

def load_yolo_label_file(path: str) -> np.ndarray:
    """
    Load YOLO format label file.
    Args:
        path: Path to the label file.
    Returns:
        np.ndarray: Array of shape (N, 5) where each row is [class_id, cx, cy, w, h].
    """
    if not os.path.exists(path):
        return np.zeros((0,5), float)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) == 5:
                rows.append([float(x) for x in p])
    return np.array(rows, float) if rows else np.zeros((0,5), float) 


def yolo_to_xyxy(lbl_arr: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert YOLO label array to (x1, y1, x2, y2) format.
    Args:
        lbl_arr: Array of shape (N, 5) in YOLO format.
        W: Image width.
        H: Image height.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of shape (N, 4) with bounding boxes in xyxy format.
            - Array of class indices (N,).
    """
    if lbl_arr.size == 0:
        return np.zeros((0,4), float), np.zeros((0,), int)
    cls = lbl_arr[:,0].astype(int)
    cx, cy = lbl_arr[:,1]*W, lbl_arr[:,2]*H
    w,  h  = lbl_arr[:,3]*W, lbl_arr[:,4]*H
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return np.stack([x1,y1,x2,y2], axis=1), cls

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    Args:
        A: Array of shape (N, 4) for GT boxes.
        B: Array of shape (M, 4) for predicted boxes.
    Returns:
        np.ndarray: IoU matrix of shape (N, M).
    """
    if A.shape[0]==0 or B.shape[0]==0:
        return np.zeros((A.shape[0], B.shape[0]), float)
    ax1, ay1, ax2, ay2 = A[:,0:1], A[:,1:2], A[:,2:3], A[:,3:4]
    bx1, by1, bx2, by2 = B[:,0], B[:,1], B[:,2], B[:,3]
    ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0, ix2-ix1); ih = np.maximum(0, iy2-iy1)
    inter = iw*ih
    areaA = (ax2-ax1)*(ay2-ay1)
    areaB = (bx2-bx1)*(by2-by1)
    union = areaA + areaB - inter
    return (inter/np.maximum(union, 1e-9)).squeeze()

def greedy_match(iou_mat: np.ndarray, thr: float = 0.5) -> List[Tuple[int, int]]:
    """
    Class-agnostic 1-1 matching by IoU.

    (Kopiuje macierz IoU,

    Szuka największego elementu; jeśli < thr, koniec,

    Dodaje parę (wiersz, kolumna) do wyników,

    Wyklucza wiersz i kolumnę (ustawia na -1), żeby nie łączyć ponownie tych samych obiektów,

    Powtarza aż nie ma sensownych kandydatów.)
    
    Args:
        iou_mat: IoU matrix (N, M).
        thr: IoU threshold for matching.
    Returns:
        List of (gi, pi) tuples for matched GT and prediction indices.
    """
    matches: List[Tuple[int, int]] = []
    if iou_mat.size == 0:
        return matches
    if len(iou_mat.shape) < 2 or iou_mat.shape[0] == 0 or iou_mat.shape[1] == 0:
        return matches
    M = iou_mat.copy()
    while True:
        if np.all(M < thr):
            break
        gi, pi = np.unravel_index(np.argmax(M), M.shape)
        if M[gi, pi] < thr: break
        matches.append((gi, pi))
        M[gi, :] = -1.0
        M[:, pi] = -1.0
    return matches

def evaluate_multiclass_detector(
    model: Any,
    image_dir: str,
    label_dir: Optional[str] = None,
    conf: float = 0.25,
    iou_thresh: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate multiclass detector with confusion matrix and metrics.
    Args:
        model: YOLO model object.
        image_dir: Directory with images.
        label_dir: Directory with label files (if None, uses image_dir).
        conf: Confidence threshold for predictions.
        iou_thresh: IoU threshold for matching.
    Returns:
        Dict with confusion_matrix, labels, per_class metrics, macro_f1, accuracy.
    """
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(img_exts)]
    nC = len(CLASS_NAMES); BG = nC
    C = np.zeros((nC+1, nC+1), dtype=int)

    for fn in images:
        img_path = os.path.join(image_dir, fn)
        base, _ = os.path.splitext(fn)
        lbl_path = os.path.join(label_dir or image_dir, base + ".txt")

        # GT
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_raw = load_yolo_label_file(lbl_path)
        gt_boxes, gt_cls = yolo_to_xyxy(gt_raw, W, H)

        # Pred
        r = model(img_path, conf=conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes)==0:
            pred_boxes = np.zeros((0,4), float)
            pred_cls   = np.zeros((0,), int)
        else:
            pred_boxes = r.boxes.xyxy.cpu().numpy()
            pred_cls   = r.boxes.cls.cpu().numpy().astype(int)

        # Diagnostyka
        print(f"\nObraz: {fn}")
        print(f"  Liczba GT: {len(gt_boxes)}  |  Liczba pred: {len(pred_boxes)}")
        if len(gt_boxes) > 0:
            print(f"  Przykład GT: {gt_boxes[0]}, klasa: {gt_cls[0]}")
        if len(pred_boxes) > 0:
            print(f"  Przykład pred: {pred_boxes[0]}, klasa: {pred_cls[0]}")

        # Dopasowanie class-agnostic
        IoU = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(IoU, thr=iou_thresh)
        print(f"  Liczba dopasowań (IoU>{iou_thresh}): {len(matches)}")
        matched_gt = {gi for gi,_ in matches}
        matched_pd = {pi for _,pi in matches}

        # Zliczenia do CM
        # 1) dopasowane: GT vs Pred wg klas
        for gi, pi in matches:
            gk = int(gt_cls[gi]); pk = int(pred_cls[pi])
            if gk >= nC: gk = BG
            if pk >= nC: pk = BG
            C[gk, pk] += 1

        # 2) GT bez dopasowania -> FN (GT klasa -> Pred BG)
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                gk = int(gt_cls[gi]); 
                if gk >= nC: gk = BG
                C[gk, BG] += 1

        # 3) Pred bez dopasowania -> FP (GT BG -> Pred klasa)
        for pi in range(len(pred_boxes)):
            if pi not in matched_pd:
                pk = int(pred_cls[pi]); 
                if pk >= nC: pk = BG
                C[BG, pk] += 1

    # Metryki per klasa (bez tła)
    per_class = []
    for k in range(nC):
        TP = C[k,k]
        FP = C[:,k].sum() - TP  
        FN = C[k,:].sum() - TP    
        P = TP / max(TP + FP, EPS)
        R = TP / max(TP + FN, EPS)
        F1 = 2 * P * R / max(P + R, EPS)
        per_class.append({
            "class_id": k, "class_name": CLASS_NAMES[k],
            "TP": int(TP), "FP": int(FP), "FN": int(FN),
            "precision": P, "recall": R, "f1": F1
        })

    macro_f1 = float(np.mean([x["f1"] for x in per_class])) if per_class else 0.0
    total_effective = C.sum() - C[-1,-1]
    total_correct   = sum(C[k,k] for k in range(nC))
    accuracy = total_correct / max(total_effective, EPS)

    return {
        "confusion_matrix": C,
        "labels": CLASS_NAMES + [BG_NAME],
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": accuracy
    }

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> Any:
    """
    Plot confusion matrix with counts or normalized values.
    Args:
        cm: Confusion matrix (array).
        labels: List of class labels.
        normalize: Whether to normalize rows.
        title: Title for the plot.
    Returns:
        Matplotlib figure object.
    """
    import numpy as np, matplotlib.pyplot as plt
    M = cm.astype(float)
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(M, interpolation="nearest", cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}" if normalize else str(int(cm[i,j])),
                    ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


models_paths = [
    r'yolo_models\models\yolo_cells_detector_hybrid2\weights\best.pt',
    r'yolo_models\models\yolo_cells_detector_hybrid3\weights\best.pt',
    r'yolo_models\models\yolo_detector_2107_100_10_16_7683\weights\best.pt',
    r'yolo_models\models\yolo_detector_2107_100_20_16_7682\weights\best.pt',
]

test_folders = [
    r"data\LBC_slides\HSIL\pow 10",
    r"data\LBC_slides\LSIL\pow. 10",
    r"data\LBC_slides\NSIL\pow. 10"
]

metric_folder = r'data_yolo\syntetic_and_mine_val'
num_samples = 15

output_root = r"C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\tests"

test_samples_per_folder = {}

for folder in test_folders:
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    test_samples_per_folder[folder] = sample_files

for model_path in models_paths:
    print(f"\n>>> Testowanie modelu: {model_path}")
    model = YOLO(model_path)

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(os.path.dirname(model_dir))

    for folder in test_folders:
        print(f"\nFolder testowy: {folder}")
        sample_files = test_samples_per_folder[folder]

        n_cols = 5
        n_rows = math.ceil(len(sample_files) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, file_name in enumerate(sample_files):
            img_path = os.path.join(folder, file_name)
            print(f"  Przetwarzanie: {img_path}")
            results = model(img_path, conf=0.2)
            result_img = results[0].plot()

            axes[idx].imshow(result_img)
            axes[idx].set_title(file_name, fontsize=10)
            axes[idx].axis("off")

        for j in range(len(sample_files), len(axes)):
            axes[j].axis("off")

        folder_parts = os.path.normpath(folder).split(os.sep)
        last_two = "_".join(folder_parts[-2:]).replace(".", "").replace(" ", "_")
        save_dir = os.path.join(output_root, model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{last_two}.png")

        plt.suptitle(f"Model: {model_name}\nFolder: {last_two}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Zapisano do: {save_path}")
    
    print(">>> Ewaluacja (cell, HSIL_group) na:", metric_folder)
    metrics = evaluate_multiclass_detector(
        model=model,
        image_dir=os.path.join(metric_folder, "images"),
        label_dir=os.path.join(metric_folder, "labels"),
        conf=0.25,
        iou_thresh=0.5
    )

    print(f"Accuracy={metrics['accuracy']:.4f}   Macro-F1={metrics['macro_f1']:.4f}")
    for r in metrics["per_class"]:
        print(f"[{r['class_name']}]  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
        f"TP={r['TP']} FP={r['FP']} FN={r['FN']}")

    save_dir = os.path.join(output_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                                        normalize=False, title="Confusion Matrix (counts)")
    fig.savefig(os.path.join(save_dir, "cm_counts.png"), dpi=200); plt.close(fig)

    fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                                        normalize=True, title="Confusion Matrix Normalized")
    fig.savefig(os.path.join(save_dir, "cm_normalized.png"), dpi=200); plt.close(fig)

    csv_path = os.path.join(output_root, "metrics_summary.csv")
    import csv
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["model", "accuracy", "macro_f1"])
        writer.writerow([model_name, f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}"])

