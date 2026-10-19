"""Write the machine-readable products: results.json and analysis_sample.csv."""
import json
import os
import sys
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib
from .config import SEED


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    raise TypeError(type(o))


def run(ctx):
    ctx.R["versions"] = dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__,
                             statsmodels=sm.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                             seed=SEED)
    with open(os.path.join(ctx.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(ctx.R, fh, default=_jsonable, indent=1)
    ctx.S.to_csv(os.path.join(ctx.out, "analysis_sample.csv"), index=False)
