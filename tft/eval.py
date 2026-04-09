import numpy as np

def pinball_loss(y_true, y_pred_q, q=0.9):
    diff = y_true - y_pred_q
    return np.mean(np.maximum(q * diff, (q - 1) * diff))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def combined_score(y_true, y_pred, y_pred_p90, peak_mask):
    mae_all = mae(y_true, y_pred)
    mae_peak = mae(y_true[peak_mask], y_pred[peak_mask])
    pinball_p90 = pinball_loss(y_true, y_pred_p90, q=0.9)
    return 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90