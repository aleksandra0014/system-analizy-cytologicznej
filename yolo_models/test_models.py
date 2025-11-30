import math
import cv2
import numpy as np
from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import csv

CLASS_NAMES = ["cell", "HSIL_group"]   # 0 -> cell, 1 -> HSIL_group
BG_NAME = "background"
EPS = 1e-9

# NOWE: Progi confidence dla każdej klasy
CLASS_CONF_THRESHOLDS = {
    0: 0.35,  # cell
    1: 0.35    # HSIL_group
}

from typing import List, Tuple, Optional, Any, Dict

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    use_median: bool = False,
    median_kernel: int = 3
) -> np.ndarray:
    """
    Apply CLAHE to a RGB image.

    Args:
        image (np.ndarray): Input image.
        clip_limit (float): Maximum histogram height (Hmax).
        tile_grid_size (tuple): Block size (e.g., (8,8)).
        use_median (bool): Whether to apply median filter before CLAHE.
        median_kernel (int): Median window size (must be odd).

    Returns:
        np.ndarray: CLAHE-processed image (same shape as input).
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if use_median:
            l = cv2.medianBlur(l, median_kernel)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return result
    return image

def load_yolo_label_file(path: str) -> np.ndarray:
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
    if lbl_arr.size == 0:
        return np.zeros((0,4), float), np.zeros((0,), int)
    cls = lbl_arr[:,0].astype(int)
    cx, cy = lbl_arr[:,1]*W, lbl_arr[:,2]*H
    w,  h  = lbl_arr[:,3]*W, lbl_arr[:,4]*H
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return np.stack([x1,y1,x2,y2], axis=1), cls

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
    matches = []
    if iou_mat.size == 0:
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

def has_labels(image_dir: str, label_dir: Optional[str] = None) -> bool:
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(img_exts)]
    
    if not images:
        return False
    
    label_path = label_dir or image_dir
    for fn in images[:5]:
        base, _ = os.path.splitext(fn)
        lbl_path = os.path.join(label_path, base + ".txt")
        if os.path.exists(lbl_path):
            return True
    return False

def apply_class_specific_conf(pred_cls, pred_conf, conf_thresholds):
    mask = np.zeros(len(pred_cls), dtype=bool)
    for class_id, threshold in conf_thresholds.items():
        class_mask = (pred_cls == class_id) & (pred_conf >= threshold)
        mask |= class_mask
    return mask

def evaluate_multiclass_detector_with_pr(
    model, image_dir, label_dir=None, conf_thresholds=None, iou_thresh=0.5
):
    if conf_thresholds is None:
        conf_thresholds = CLASS_CONF_THRESHOLDS
    
    if not has_labels(image_dir, label_dir):
        print(f"  ⚠️ Brak plików z labelami dla {image_dir} - pomijam ewaluację")
        return None
    
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(img_exts)]
    nC = len(CLASS_NAMES); BG = nC
    C = np.zeros((nC+1, nC+1), dtype=int)
    
    all_predictions = {k: [] for k in range(nC)}

    for fn in images:
        img_path = os.path.join(image_dir, fn)
        base, _ = os.path.splitext(fn)
        lbl_path = os.path.join(label_dir or image_dir, base + ".txt")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_raw = load_yolo_label_file(lbl_path)
        gt_boxes, gt_cls = yolo_to_xyxy(gt_raw, W, H)

        r = model(img_path, conf=0.01, verbose=False)[0]
        if r.boxes is None or len(r.boxes)==0:
            pred_boxes = np.zeros((0,4), float)
            pred_cls   = np.zeros((0,), int)
            pred_conf  = np.zeros((0,), float)
        else:
            pred_boxes = r.boxes.xyxy.cpu().numpy()
            pred_cls   = r.boxes.cls.cpu().numpy().astype(int)
            pred_conf  = r.boxes.conf.cpu().numpy()

        conf_mask = apply_class_specific_conf(pred_cls, pred_conf, conf_thresholds)
        pred_boxes_cm = pred_boxes[conf_mask]
        pred_cls_cm = pred_cls[conf_mask]
        pred_conf_cm = pred_conf[conf_mask]
        
        IoU_cm = iou_matrix(gt_boxes, pred_boxes_cm)
        matches_cm = greedy_match(IoU_cm, thr=iou_thresh)
        matched_gt_cm = {gi for gi,_ in matches_cm}
        matched_pd_cm = {pi for _,pi in matches_cm}

        IoU_pr = iou_matrix(gt_boxes, pred_boxes)
        matches_pr = greedy_match(IoU_pr, thr=iou_thresh)
        matched_pd_pr = {pi for _,pi in matches_pr}
        
        for gi, pi in matches_cm:
            gk = int(gt_cls[gi]); pk = int(pred_cls_cm[pi])
            if gk >= nC: gk = BG
            if pk >= nC: pk = BG
            C[gk, pk] += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt_cm:
                gk = int(gt_cls[gi]); 
                if gk >= nC: gk = BG
                C[gk, BG] += 1

        for pi in range(len(pred_boxes_cm)):
            if pi not in matched_pd_cm:
                pk = int(pred_cls_cm[pi]); 
                if pk >= nC: pk = BG
                C[BG, pk] += 1

        for pi in range(len(pred_boxes)):
            pk = int(pred_cls[pi])
            if pk >= nC:
                continue
            confidence = float(pred_conf[pi])
            is_tp = False
            if pi in matched_pd_pr:
                gi = next(gi for gi, p in matches_pr if p == pi)
                gk = int(gt_cls[gi])
                if gk == pk:
                    is_tp = True
            all_predictions[pk].append((confidence, is_tp))

    gt_counts = {k: 0 for k in range(nC)}
    for fn in images:
        base, _ = os.path.splitext(fn)
        lbl_path = os.path.join(label_dir or image_dir, base + ".txt")
        img_path = os.path.join(image_dir, fn)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_raw = load_yolo_label_file(lbl_path)
        _, gt_cls = yolo_to_xyxy(gt_raw, W, H)
        for gk in gt_cls:
            if gk < nC:
                gt_counts[int(gk)] += 1

    pr_data = {}
    for k in range(nC):
        preds = all_predictions[k]
        n_gt = gt_counts[k]
        
        if len(preds) == 0 or n_gt == 0:
            pr_data[k] = {"precisions": np.array([]),"recalls": np.array([]),"thresholds": np.array([]),"ap": 0.0}
            continue
        
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        
        precisions = []
        recalls = []
        thresholds = []
        
        tp_cumsum = 0
        fp_cumsum = 0
        
        for conf, is_tp in preds_sorted:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / n_gt
            
            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(conf)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        precisions = np.concatenate([[1.0], precisions, [0.0]])
        recalls = np.concatenate([[0.0], recalls, [recalls[-1]]])
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        ap = np.trapz(precisions, recalls)
        
        pr_data[k] = {
            "precisions": precisions[1:-1],
            "recalls": recalls[1:-1],
            "thresholds": np.array(thresholds),
            "ap": float(ap)
        }

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
            "precision": P, "recall": R, "f1": F1,
            "ap": pr_data[k]["ap"],
            "conf_threshold": conf_thresholds.get(k, 0.5)
        })

    macro_f1 = float(np.mean([x["f1"] for x in per_class])) if per_class else 0.0
    mean_ap = float(np.mean([x["ap"] for x in per_class])) if per_class else 0.0
    total_effective = C.sum() - C[-1,-1]
    total_correct   = sum(C[k,k] for k in range(nC))
    accuracy = total_correct / max(total_effective, EPS)

    return {
        "confusion_matrix": C,
        "labels": CLASS_NAMES + [BG_NAME],
        "per_class": per_class,
        "macro_f1": macro_f1,
        "mean_ap": mean_ap,
        "accuracy": accuracy,
        "pr_data": pr_data,
        "conf_thresholds": conf_thresholds
    }

def plot_confusion_matrix(cm, labels, normalize=False, title="Confusion Matrix"):
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

def plot_pr_curve(pr_data, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for k, data in pr_data.items():
        if len(data["recalls"]) == 0:
            continue
        ax.plot(data["recalls"], data["precisions"], linewidth=2, color=colors[k],
                label=f'{class_names[k]} (AP={data["ap"]:.3f})')
    
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    return fig

# ============== MAIN EXECUTION ==============

models_paths = [
    r'C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\models\16.0.001.augmentacja15_test22_adam\weights\best.pt',
]

test_folders = [
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data_yolo\syntetic_and_mine_test\images",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_mendeley\Low squamous intra-epithelial lesion",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_mendeley\Negative for Intraepithelial malignancy",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\HSIL\pow 40",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\LSIL\pow 40",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\NSIL\pow 40",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\HSIL\pow 10",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\LSIL\pow 10",
    r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\NSIL\pow 10",
]

metric_folder = r'data_yolo\syntetic_and_mine_test'
num_samples = 10
output_root = r"C:\Users\aleks\OneDrive\Documents\inzynierka\yolo_models\tests\new"

# Przygotowanie próbek
test_samples_per_folder = {}
for folder in test_folders:
    if not os.path.exists(folder):
        print(f"⚠️ Folder nie istnieje: {folder}")
        continue
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    test_samples_per_folder[folder] = sample_files

# Przetwarzanie modeli
for model_path in models_paths:
    print(f"\n{'='*80}")
    print(f">>> Testowanie modelu: {model_path}")
    print(f"{'='*80}")
    model = YOLO(model_path)

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(os.path.dirname(model_dir))

    min_conf = min(CLASS_CONF_THRESHOLDS.values())
    print(f"  ℹ️  Używane progi confidence: {CLASS_CONF_THRESHOLDS}")
    print(f"  ℹ️  Wizualizacja: cell=czerwony, HSIL_group=żółty")

    # Kolory dla każdej klasy (RGB)
    colors = {
        0: (255, 0, 0),    # cell - czerwony
        1: (255, 255, 0)   # HSIL_group - żółty
    }
    
    # Grubość linii dla każdej klasy
    line_widths = {
        0: 10,   # cell - grubsza linia
        1: 10    # HSIL_group - najgrubsza linia
    }

    for folder in test_folders:
        if folder not in test_samples_per_folder:
            continue
            
        print(f"\n📁 Folder testowy: {folder}")
        sample_files = test_samples_per_folder[folder]

        n_cols = 5
        n_rows = math.ceil(len(sample_files) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, file_name in enumerate(sample_files):
            img_path = os.path.join(folder, file_name)
            print(f"  🖼️  Przetwarzanie: {file_name}")

            # Wczytaj obraz
            img = cv2.imread(img_path)
            
            # Zastosuj CLAHE
            # img = apply_clahe(
            #     img,
            #     clip_limit=2.0,
            #     tile_grid_size=(8, 8),
            #     use_median=True,
            #     median_kernel=3
            # )
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Predykcja
            results = model(img_path, conf=min_conf, verbose=False)

            # Rysuj bounding boxy
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Pobierz kolor i grubość dla klasy
                    color = colors.get(cls, (255, 0, 0))
                    line_width = line_widths.get(cls, 3)
                    
                    # Rysuj bbox
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, line_width)
                    
                    # Label z wyraźniejszym tekstem
                    label = f"{CLASS_NAMES[cls]} {conf:.2f}"
                    font_scale = 1
                    font_thickness = 4
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Tło dla tekstu
                    cv2.rectangle(img_rgb, (x1, y1 - text_h - baseline - 5), (x1 + text_w + 5, y1), color, -1)
                    
                    # Biały tekst z czarnym konturem dla lepszej czytelności
                    cv2.putText(img_rgb, label, (x1 + 2, y1 - baseline - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
                    cv2.putText(img_rgb, label, (x1 + 2, y1 - baseline - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            axes[idx].imshow(img_rgb)
            axes[idx].set_title(file_name, fontsize=12)
            axes[idx].axis("off")

        for j in range(len(sample_files), len(axes)):
            axes[j].axis("off")

        folder_parts = os.path.normpath(folder).split(os.sep)
        last_two = "_".join(folder_parts[-2:]).replace(".", "").replace(" ", "_")
        save_dir = os.path.join(output_root, model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{last_two}.png")

        plt.suptitle(f"Model: {model_name}\nFolder: {last_two}", fontsize=18, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Zapisano do: {save_path}")
    
    print(f"\n{'='*80}")
    print(f">>> Ewaluacja na: {metric_folder}")
    print(f"{'='*80}")
    
    metrics = evaluate_multiclass_detector_with_pr(
        model=model,
        image_dir=os.path.join(metric_folder, "images"),
        label_dir=os.path.join(metric_folder, "labels"),
        conf_thresholds=CLASS_CONF_THRESHOLDS,
        iou_thresh=0.5
    )

    if metrics is None:
        print("⏭️  Pomijam ewaluację - brak plików z labelami\n")
        continue

    print(f"\n📊 WYNIKI:")
    print(f"  Accuracy  = {metrics['accuracy']:.4f}")
    print(f"  Macro-F1  = {metrics['macro_f1']:.4f}")
    print(f"  mAP       = {metrics['mean_ap']:.4f}\n")
    
    for r in metrics["per_class"]:
        print(f"  [{r['class_name']:12s}]  F1={r['f1']:.4f}  P={r['precision']:.4f}  "
              f"R={r['recall']:.4f}  AP={r['ap']:.4f}  TP={r['TP']} FP={r['FP']} FN={r['FN']}")

    save_dir = os.path.join(output_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], normalize=False)
    fig.savefig(os.path.join(save_dir, "cm_counts.png"), dpi=200); plt.close(fig)

    fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], normalize=True)
    fig.savefig(os.path.join(save_dir, "cm_normalized.png"), dpi=200); plt.close(fig)

    fig = plot_pr_curve(metrics["pr_data"], CLASS_NAMES)
    fig.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=200); plt.close(fig)
    print(f"\n✅ Zapisano wykresy do: {save_dir}")

    csv_path = os.path.join(output_root, "metrics_summary.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["model", "accuracy", "macro_f1", "mean_ap", "conf_thresholds"])
        writer.writerow([model_name, f"{metrics['accuracy']:.4f}", 
                        f"{metrics['macro_f1']:.4f}", f"{metrics['mean_ap']:.4f}",
                        str(CLASS_CONF_THRESHOLDS)])

print(f"\n{'='*80}")
print("✅ Wszystkie testy zakończone!")
print(f"{'='*80}")