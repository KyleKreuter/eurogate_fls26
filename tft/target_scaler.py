import numpy as np

class StandardTargetScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: np.ndarray):
        self.mean = float(np.nanmean(y))
        self.std = float(np.nanstd(y))
        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.std + self.mean