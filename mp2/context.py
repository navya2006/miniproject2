"""The Context object: one place where each stage leaves its products for the next stage."""
import os
import numpy as np
from .config import SEED


class Context:
    """Carries data, fitted models and results between pipeline stages.

    R        dict of every number that ends up in results.json
    raw, df  raw sheet (as read) and the cleaned full data frame
    S, S4    main analysis sample and the (smaller) sample with family-background variables
    MOD/ROB  classical and HC3-robust statsmodels fits for M1..M6
    rng      the single seeded random generator (draw order is fixed, so results are reproducible)
    """

    def __init__(self, src, out):
        self.src = src
        self.out = out
        self.fig_dir = os.path.join(out, "figures")
        self.tab_dir = os.path.join(out, "tables")
        self.img_tab_dir = os.path.join(out, "table_images")
        for p in (out, self.fig_dir, self.tab_dir, self.img_tab_dir):
            os.makedirs(p, exist_ok=True)
        self.rng = np.random.default_rng(SEED)
        self.R = {}
        self.raw = self.df = self.log = None
        self.S = self.S4 = None
        self.N = 0
        self.MOD, self.ROB, self.TBL, self.FIT = {}, {}, {}, {}
        self.TB5F = None
        self.jitter = {}
