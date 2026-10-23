import numpy as np
from scipy.stats import ks_2samp

def diagnosticar(y_train, pred_train, y_test, pred_test):
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    e_tr = y_tr - np.asarray(pred_train)
    e_te = y_te - np.asarray(pred_test)

    ks_error = ks_2samp(e_tr, e_te)
    ks_ventas = ks_2samp(y_tr, y_te)

    def resumen(y, e):
        return {
            "sesgo": float(e.mean()),
            "pbias_pct": float(100 * e.sum() / y.sum()),
            "std_error": float(e.std())
        }

    return {
        "train": resumen(y_tr, e_tr),
        "test": resumen(y_te, e_te),
        "deriva_errores": {
            "ks": float(ks_error.statistic),
            "p_value": float(ks_error.pvalue)
        },
        "deriva_ventas": {
            "ks": float(ks_ventas.statistic),
            "p_value": float(ks_ventas.pvalue)
        }
    }