from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

from temporal_interaction_head import TemporalInteractionHead


def _latent_from_row(row, emb_dim=64):
    target = row.get("y_future_latent")
    if isinstance(target, list) and len(target) > 0:
        arr = np.zeros((emb_dim,), dtype=np.float32)
        m = min(len(target), emb_dim)
        arr[:m] = np.asarray(target[:m], dtype=np.float32)
        return arr
    feat = row.get("feat")
    if isinstance(feat, list) and len(feat) >= emb_dim * 2:
        arr = np.zeros((emb_dim,), dtype=np.float32)
        arr[:] = np.asarray(feat[emb_dim:emb_dim * 2], dtype=np.float32)
        return arr
    return np.zeros((emb_dim,), dtype=np.float32)


def _binary_metrics(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    pred = prob >= threshold
    tgt = target >= 0.5
    tp = int(np.logical_and(pred, tgt).sum())
    fp = int(np.logical_and(pred, ~tgt).sum())
    tn = int(np.logical_and(~pred, ~tgt).sum())
    fn = int(np.logical_and(~pred, tgt).sum())
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_rate": round(float(tgt.mean()) if tgt.size else 0.0, 4),
        "mean_prob": round(float(prob.mean()) if prob.size else 0.0, 4),
    }


def _threshold_sweep(prob: np.ndarray, target: np.ndarray) -> dict:
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    rows = [_binary_metrics(prob, target, threshold=t) for t in thresholds]
    best_f1 = max(rows, key=lambda item: (item["f1"], item["recall"], -item["threshold"]))
    high_recall = [item for item in rows if item["recall"] >= 0.75]
    best_high_recall = max(high_recall, key=lambda item: (item["precision"], item["f1"])) if high_recall else None
    return {
        "best_f1": best_f1,
        "best_precision_at_recall_0_75": best_high_recall,
        "points": rows,
    }


def load_rows(path: str):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r for r in rows if isinstance(r.get("feat"), list)]
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    x = np.asarray([r["feat"] for r in rows], dtype=np.float32)
    yc = np.asarray([float(r.get("y_contact", 0.0)) for r in rows], dtype=np.float32).reshape(-1, 1)
    yp = np.asarray([float(r.get("y_place", 0.0)) for r in rows], dtype=np.float32).reshape(-1, 1)
    yr = np.asarray([float(r.get("y_release", 0.0)) for r in rows], dtype=np.float32).reshape(-1, 1)
    yt = np.asarray([float(r.get("y_target_tray", 0.0)) for r in rows], dtype=np.float32).reshape(-1, 1)
    ym = np.asarray([r.get("y_motion", [0.0, 0.0, 0.0]) for r in rows], dtype=np.float32)
    yz = np.asarray([_latent_from_row(r, 64) for r in rows], dtype=np.float32)
    return rows, x, yc, yp, yr, yt, ym, yz


def evaluate_model(model, x, yc, yp, yr, yt, ym, yz):
    with torch.no_grad():
        out = model(torch.from_numpy(x))
        contact_prob = torch.sigmoid(out["contact_logit"]).cpu().numpy()
        place_prob = torch.sigmoid(out["placement_logit"]).cpu().numpy()
        release_prob = torch.sigmoid(out.get("release_logit", out["placement_logit"])).cpu().numpy()
        target_tray_prob = torch.sigmoid(out.get("target_tray_logit", out["placement_logit"])).cpu().numpy()
        motion = out["motion_delta"].cpu().numpy()
        latent = out.get("future_latent")
        latent = latent.cpu().numpy() if latent is not None else np.zeros_like(yz)
    motion_mae = np.abs(motion - ym).mean(axis=0)
    latent_mse = float(np.mean((latent - yz) ** 2)) if yz.size else 0.0
    denom = np.linalg.norm(latent, axis=1) * np.linalg.norm(yz, axis=1)
    valid = denom > 1e-6
    latent_cos = np.zeros((latent.shape[0],), dtype=np.float32)
    if valid.any():
        latent_cos[valid] = (latent[valid] * yz[valid]).sum(axis=1) / denom[valid]
    contact_flat = contact_prob.reshape(-1)
    place_flat = place_prob.reshape(-1)
    release_flat = release_prob.reshape(-1)
    target_flat = target_tray_prob.reshape(-1)
    yc_flat = yc.reshape(-1)
    yp_flat = yp.reshape(-1)
    yr_flat = yr.reshape(-1)
    yt_flat = yt.reshape(-1)
    return {
        "contact": _binary_metrics(contact_flat, yc_flat),
        "contact_threshold_sweep": _threshold_sweep(contact_flat, yc_flat),
        "placement": _binary_metrics(place_flat, yp_flat),
        "placement_threshold_sweep": _threshold_sweep(place_flat, yp_flat),
        "release": _binary_metrics(release_flat, yr_flat),
        "release_threshold_sweep": _threshold_sweep(release_flat, yr_flat),
        "target_tray": _binary_metrics(target_flat, yt_flat),
        "target_tray_threshold_sweep": _threshold_sweep(target_flat, yt_flat),
        "motion_mae_xyz": [round(float(v), 5) for v in motion_mae.tolist()],
        "motion_mae_mean": round(float(np.abs(motion - ym).mean()), 5),
        "future_latent_mse": round(float(latent_mse), 6),
        "future_latent_cosine_mean": round(float(latent_cos[valid].mean()) if valid.any() else 0.0, 4),
        "future_latent_valid": int(valid.sum()),
    }


def train_reference_fit(x, yc, yp, yr, yt, ym, yz, epochs: int, lr: float):
    model = TemporalInteractionHead(in_dim=x.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    xt = torch.from_numpy(x)
    yct = torch.from_numpy(yc)
    ypt = torch.from_numpy(yp)
    yrt = torch.from_numpy(yr)
    ytt = torch.from_numpy(yt)
    ymt = torch.from_numpy(ym)
    yzt = torch.from_numpy(yz)
    for _ in range(max(1, int(epochs))):
        out = model(xt)
        loss = (
            bce(out["contact_logit"], yct)
            + bce(out["placement_logit"], ypt)
            + bce(out.get("release_logit", out["placement_logit"]), yrt)
            + 0.5 * bce(out["target_tray_logit"], ytt)
            + 0.5 * mse(out["motion_delta"], ymt)
            + 0.2 * mse(out["future_latent"], yzt)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="world_model/data/temporal_head_train_rows_sophie.json")
    parser.add_argument("--model", default="world_model/models/temporal_interaction_head_sophie.pt")
    parser.add_argument("--output", default="world_model/data/temporal_head_eval_sophie_latest.json")
    parser.add_argument("--fit-epochs", type=int, default=80)
    parser.add_argument("--save-fit-model", default="")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required")

    rows, x, yc, yp, yr, yt, ym, yz = load_rows(args.rows)
    report = {
        "rows": len(rows),
        "feature_dim": int(x.shape[1]),
        "contact_positive": int((yc >= 0.5).sum()),
        "placement_positive": int((yp >= 0.5).sum()),
        "release_positive": int((yr >= 0.5).sum()),
        "target_tray_positive": int((yt >= 0.5).sum()),
        "fit_baseline_note": (
            "row_fit_upper_bound is trained and evaluated on these same rows. "
            "It is an upper-bound sanity check for feature/label learnability, not held-out performance."
        ),
    }

    model = TemporalInteractionHead(in_dim=x.shape[1])
    model_path = Path(args.model)
    if model_path.is_file():
        state = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        report["saved_model"] = evaluate_model(model, x, yc, yp, yr, yt, ym, yz)
    else:
        report["saved_model"] = {"missing": str(model_path)}

    reference_fit = train_reference_fit(x, yc, yp, yr, yt, ym, yz, epochs=args.fit_epochs, lr=args.lr)
    reference_fit.eval()
    report["row_fit_upper_bound"] = evaluate_model(reference_fit, x, yc, yp, yr, yt, ym, yz)
    if args.save_fit_model:
        save_path = Path(args.save_fit_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference_fit.state_dict(), str(save_path))
        report["row_fit_upper_bound"]["saved_to"] = str(save_path)
    report["interpretation"] = (
        "If row_fit_upper_bound metrics are strong but saved_model is weak, the rows contain learnable signal "
        "but the deployed temporal head needs more training/tuning. If both are weak, labels/features need work."
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
