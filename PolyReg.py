# PolyReg.py
"""
Desktop GUI for Polynomial Regression .

- Load CSV or paste X/Y values
- Manual degree or Auto (BIC/AIC/CV R²)
- Optional Ridge regularization
- Shows equation and metrics (R², adj R², RMSE, MAE, AIC, BIC, CV R²)
- Plots: Fit, Residuals vs Fitted, Residual histogram, Q–Q
- Forecast: predict Y for X and solve for X at target Y
- Export predictions/residuals (CSV) and model (JSON)

Dependencies:
    python -m pip install numpy pandas scikit-learn matplotlib scipy or use standalone PolyReg.exe (github)

Run:
    python PolyReg.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

# PyQt6 / GUI
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSpinBox, QComboBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas


# Core helpers


def _re_split(s: str) -> List[str]:
    import re
    return [t for t in re.split(r"[\s,;]+", s.strip()) if t]

def parse_number_list(text: str) -> List[float]:
    return [float(t) for t in _re_split(text)] if text and text.strip() else []

def load_csv(path: Path, sep: str, header: bool) -> pd.DataFrame:
    header_opt = 0 if header else None
    return pd.read_csv(path, sep=sep, header=header_opt)

def build_design_matrix(x: np.ndarray, degree: int, include_bias: bool = True) -> Tuple[np.ndarray, PolynomialFeatures]:
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    return X_poly, poly

def format_poly_equation(coefs: np.ndarray, precision: int = 6, variable: str = "x") -> str:
    terms = []
    for power, coef in enumerate(coefs):
        if abs(coef) < 10 ** (-precision):
            continue
        sign = " - " if coef < 0 else (" + " if terms else "")
        coef_abs = abs(coef)
        if power == 0:
            term = f"{coef_abs:.{precision}g}"
        elif power == 1:
            term = f"{coef_abs:.{precision}g}{variable}"
        else:
            term = f"{coef_abs:.{precision}g}{variable}^{power}"
        terms.append(sign + term)
    return "y = 0" if not terms else "y =" + "".join(terms)

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if not np.isfinite(r2) or n - p <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(resid)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "ss_res": ss_res, "ss_tot": ss_tot}

def aic_bic_from_gaussian(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> Tuple[float, float]:
    n = len(y_true)
    resid = y_true - y_pred
    sse = float(np.sum(resid ** 2))
    sigma2 = sse / n
    ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    aic = -2 * ll + 2 * p
    bic = -2 * ll + p * math.log(n)
    return float(aic), float(bic)

def kfold_cv_r2(x: np.ndarray, y: np.ndarray, degree: int, k: int = 5, ridge_alpha: Optional[float] = None) -> Tuple[float, List[float]]:
    k = min(len(x), k)
    if k < 2:
        return float("nan"), []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        Xtr, poly = build_design_matrix(x_train, degree)
        Xte = poly.transform(x_test.reshape(-1, 1))
        model = Ridge(alpha=ridge_alpha, fit_intercept=False) if (ridge_alpha and ridge_alpha > 0) else LinearRegression(fit_intercept=False)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_res = np.sum((y_test - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        scores.append(float(r2))
    return float(np.nanmean(scores)) if scores else float("nan"), scores

def solve_for_x_from_y(coefs_inc_order: np.ndarray, y_target: float) -> List[float]:
    c = coefs_inc_order.copy().astype(float)
    c[0] -= y_target
    from numpy.polynomial import polynomial as P
    roots = P.Polynomial(c).roots()
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8)


# Fit wrapper


def fit_degree(x: np.ndarray, y: np.ndarray, degree: int, ridge_alpha: Optional[float], kfold: int) -> Dict:
    X_poly, _ = build_design_matrix(x, degree, include_bias=True)
    model = Ridge(alpha=ridge_alpha, fit_intercept=False) if (ridge_alpha and ridge_alpha > 0) else LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    coefs = np.array(model.coef_, dtype=float)
    y_hat = X_poly @ coefs
    metrics = compute_metrics(y, y_hat)
    p = X_poly.shape[1]
    aic, bic = aic_bic_from_gaussian(y, y_hat, p)
    adj = adjusted_r2(metrics["r2"], len(y), p)
    cv_mean, cv_scores = kfold_cv_r2(x, y, degree, k=kfold, ridge_alpha=ridge_alpha)
    return {
        "degree": degree,
        "coefs": coefs,
        "y_hat": y_hat,
        "metrics": metrics,
        "aic": aic,
        "bic": bic,
        "adj_r2": adj,
        "cv_r2_mean": cv_mean,
        "cv_r2_scores": cv_scores,
    }

def choose_best_degree(x: np.ndarray, y: np.ndarray, max_degree: int, criterion: str, ridge_alpha: Optional[float], kfold: int) -> Dict:
    results: List[Dict] = []
    for d in range(1, max_degree + 1):
        if len(x) < d + 1:
            continue
        results.append(fit_degree(x, y, d, ridge_alpha, kfold))
    if not results:
        raise ValueError("Not enough data for requested max degree")
    if criterion == "BIC":
        return min(results, key=lambda r: (r["bic"], r["degree"]))
    if criterion == "AIC":
        return min(results, key=lambda r: (r["aic"], r["degree"]))
    # CV R2
    return max(results, key=lambda r: (r["cv_r2_mean"], -r["degree"]))


# GUI (PyQt6)


class PolyRegApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolyReg — Polynomial Regression GUI")
        self.resize(1100, 780)
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.best: Optional[Dict] = None
        self._build_ui()

    def _build_ui(self):
        # Central widget with QTabWidget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        # Pages
        self.page_data = QWidget()
        self.page_model = QWidget()
        self.page_plots = QWidget()
        self.page_forecast = QWidget()
        self.tabs.addTab(self.page_data, "Data")
        self.tabs.addTab(self.page_model, "Model")
        self.tabs.addTab(self.page_plots, "Plots")
        self.tabs.addTab(self.page_forecast, "Forecast")
        self._build_data_page()
        self._build_model_page()
        self._build_plots_page()
        self._build_forecast_page()

    # --- Data page ---
    def _build_data_page(self):
        layout = QVBoxLayout(self.page_data)
        # CSV controls
        csv_box = QGroupBox("Load CSV")
        csv_layout = QHBoxLayout(csv_box)
        self.csv_path_edit = QLineEdit()
        csv_layout.addWidget(self.csv_path_edit, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self.browse_csv)
        csv_layout.addWidget(btn_browse)
        layout.addWidget(csv_box)

        # Options grid
        opts_box = QWidget()
        opts_layout = QHBoxLayout(opts_box)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.addWidget(QLabel("X column (name or index)"))
        self.xcol_edit = QLineEdit()
        self.xcol_edit.setFixedWidth(80)
        opts_layout.addWidget(self.xcol_edit)
        opts_layout.addWidget(QLabel("Y column (name or index)"))
        self.ycol_edit = QLineEdit()
        self.ycol_edit.setFixedWidth(80)
        opts_layout.addWidget(self.ycol_edit)
        self.header_check = QCheckBox("Header row")
        self.header_check.setChecked(True)
        opts_layout.addWidget(self.header_check)
        opts_layout.addWidget(QLabel("Separator"))
        self.sep_edit = QLineEdit(",")
        self.sep_edit.setFixedWidth(30)
        opts_layout.addWidget(self.sep_edit)
        btn_load_csv = QPushButton("Load CSV")
        btn_load_csv.clicked.connect(self.load_csv_clicked)
        opts_layout.addWidget(btn_load_csv)
        layout.addWidget(opts_box)

        # Manual entry
        man_box = QGroupBox("Or paste values")
        man_layout = QVBoxLayout(man_box)
        man_layout.addWidget(QLabel("X values (comma/space/newline)"))
        self.txt_x = QTextEdit()
        self.txt_x.setFixedHeight(60)
        man_layout.addWidget(self.txt_x)
        man_layout.addWidget(QLabel("Y values (same length as X)"))
        self.txt_y = QTextEdit()
        self.txt_y.setFixedHeight(60)
        man_layout.addWidget(self.txt_y)
        btn_use_pasted = QPushButton("Use pasted data")
        btn_use_pasted.clicked.connect(self.use_pasted)
        man_layout.addWidget(btn_use_pasted, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(man_box)

        self.data_info_lbl = QLabel("No data loaded.")
        layout.addWidget(self.data_info_lbl)

    # --- Model page ---
    def _build_model_page(self):
        layout = QVBoxLayout(self.page_model)
        ctl_box = QGroupBox("Model Settings")
        ctl_layout = QHBoxLayout(ctl_box)
        # Order
        self.order_mode_combo = QComboBox()
        self.order_mode_combo.addItems(["Manual", "Auto"])
        ctl_layout.addWidget(QLabel("Order:"))
        ctl_layout.addWidget(self.order_mode_combo)
        ctl_layout.addWidget(QLabel("Degree"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 15)
        self.degree_spin.setValue(2)
        ctl_layout.addWidget(self.degree_spin)
        ctl_layout.addWidget(QLabel("Max degree"))
        self.maxdeg_spin = QSpinBox()
        self.maxdeg_spin.setRange(2, 15)
        self.maxdeg_spin.setValue(6)
        ctl_layout.addWidget(self.maxdeg_spin)
        ctl_layout.addWidget(QLabel("Criterion"))
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["BIC", "AIC", "CV R2"])
        ctl_layout.addWidget(self.criterion_combo)
        # Ridge
        self.use_ridge_chk = QCheckBox("Use Ridge")
        ctl_layout.addWidget(self.use_ridge_chk)
        ctl_layout.addWidget(QLabel("alpha"))
        self.alpha_edit = QLineEdit("1.0")
        self.alpha_edit.setFixedWidth(60)
        ctl_layout.addWidget(self.alpha_edit)
        ctl_layout.addWidget(QLabel("K-Fold"))
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        ctl_layout.addWidget(self.kfold_spin)
        btn_fit = QPushButton("Fit Model")
        btn_fit.clicked.connect(self.fit_model)
        ctl_layout.addWidget(btn_fit)
        layout.addWidget(ctl_box)

        # Summary
        sum_box = QGroupBox("Summary")
        sum_layout = QVBoxLayout(sum_box)
        self.eq_txt = QTextEdit()
        self.eq_txt.setReadOnly(True)
        self.eq_txt.setFixedHeight(50)
        sum_layout.addWidget(self.eq_txt)
        grid = QWidget()
        grid_layout = QHBoxLayout(grid)
        self.var_r2 = QLabel(); self.var_adj = QLabel(); self.var_cv = QLabel()
        self.var_rmse = QLabel(); self.var_mae = QLabel(); self.var_aic = QLabel(); self.var_bic = QLabel()
        grid_layout.addWidget(QLabel("R²:")); grid_layout.addWidget(self.var_r2)
        grid_layout.addWidget(QLabel("Adj R²:")); grid_layout.addWidget(self.var_adj)
        grid_layout.addWidget(QLabel("CV R² (mean):")); grid_layout.addWidget(self.var_cv)
        grid_layout.addWidget(QLabel("RMSE:")); grid_layout.addWidget(self.var_rmse)
        grid_layout.addWidget(QLabel("MAE:")); grid_layout.addWidget(self.var_mae)
        grid_layout.addWidget(QLabel("AIC:")); grid_layout.addWidget(self.var_aic)
        grid_layout.addWidget(QLabel("BIC:")); grid_layout.addWidget(self.var_bic)
        sum_layout.addWidget(grid)
        exp_box = QWidget()
        exp_layout = QHBoxLayout(exp_box)
        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.clicked.connect(self.export_csv)
        exp_layout.addWidget(btn_export_csv)
        btn_export_json = QPushButton("Export JSON")
        btn_export_json.clicked.connect(self.export_json)
        exp_layout.addWidget(btn_export_json)
        sum_layout.addWidget(exp_box, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(sum_box)

    # --- Plots page ---
    def _build_plots_page(self):
        layout = QVBoxLayout(self.page_plots)
        self.plot_tabs = QTabWidget()
        self.plot_frames = {}
        for name in ["Fit", "Residuals", "Histogram", "QQ Plot"]:
            tab = QWidget()
            self.plot_tabs.addTab(tab, name)
            self.plot_frames[name] = tab
        layout.addWidget(self.plot_tabs)

    def draw_plots(self):
        if not self.best:
            return
        x, y, y_hat = self.x, self.y, self.best["y_hat"]
        order = np.argsort(x)
        xs, ys, yh = x[order], y[order], y_hat[order]
        def put_plot(tab: QWidget, fig: plt.Figure):
            lyt = tab.layout()
            if lyt is None:
                lyt = QVBoxLayout(tab)
                tab.setLayout(lyt)
            while lyt.count():
                child = lyt.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            canvas = FigureCanvas(fig)
            lyt.addWidget(canvas)
        # Fit
        f1 = plt.Figure(figsize=(6,4))
        ax1 = f1.add_subplot(111)
        ax1.scatter(xs, ys, label="Data")
        ax1.plot(xs, yh, label="Fit")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.legend(); ax1.set_title("Polynomial Fit")
        put_plot(self.plot_frames["Fit"], f1)
        resid = y - y_hat
        # Residuals vs Fitted
        f2 = plt.Figure(figsize=(6,4))
        ax2 = f2.add_subplot(111)
        ax2.scatter(y_hat, resid)
        ax2.axhline(0, linestyle='--')
        ax2.set_xlabel("Fitted values"); ax2.set_ylabel("Residuals"); ax2.set_title("Residuals vs Fitted")
        put_plot(self.plot_frames["Residuals"], f2)
        # Histogram
        f3 = plt.Figure(figsize=(6,4))
        ax3 = f3.add_subplot(111)
        ax3.hist(resid, bins='auto')
        ax3.set_xlabel("Residual"); ax3.set_ylabel("Count"); ax3.set_title("Residual Histogram")
        put_plot(self.plot_frames["Histogram"], f3)
        # QQ plot
        f4 = plt.Figure(figsize=(6,4))
        ax4 = f4.add_subplot(111)
        stats.probplot(resid, dist="norm", plot=ax4)
        ax4.set_title("Q–Q Plot of Residuals")
        put_plot(self.plot_frames["QQ Plot"], f4)

    # --- Forecast page ---
    def _build_forecast_page(self):
        layout = QVBoxLayout(self.page_forecast)
        box = QGroupBox("Forecast")
        box_layout = QHBoxLayout(box)
        box_layout.addWidget(QLabel("Predict Y for X values"))
        self.forecast_x_edit = QLineEdit()
        self.forecast_x_edit.setFixedWidth(300)
        box_layout.addWidget(self.forecast_x_edit)
        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self.do_predict)
        box_layout.addWidget(btn_predict)
        box_layout.addWidget(QLabel("Solve X for target Y"))
        self.solve_y_edit = QLineEdit()
        self.solve_y_edit.setFixedWidth(100)
        box_layout.addWidget(self.solve_y_edit)
        btn_solve = QPushButton("Solve")
        btn_solve.clicked.connect(self.do_solve)
        box_layout.addWidget(btn_solve)
        layout.addWidget(box)
        # Table
        self.pred_table = QTableWidget(0, 2)
        self.pred_table.setHorizontalHeaderLabels(["X", "Predicted Y"])
        layout.addWidget(self.pred_table)

    # --- Actions ---
    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_path_edit.setText(path)

    def load_csv_clicked(self):
        try:
            path = self.csv_path_edit.text()
            if not path:
                QMessageBox.warning(self, "CSV", "Choose a CSV file first.")
                return
            sep = self.sep_edit.text() or ","
            header = self.header_check.isChecked()
            df = load_csv(Path(path), sep=sep, header=header)
            xcol = self.xcol_edit.text()
            ycol = self.ycol_edit.text()
            if xcol == "" or ycol == "":
                QMessageBox.warning(self, "CSV", "Enter X and Y column (name or index).")
                return
            def pick_col(c):
                if c.isdigit():
                    return df.columns[int(c)]
                return c
            xcol = pick_col(xcol)
            ycol = pick_col(ycol)
            x = pd.to_numeric(df[xcol], errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(df[ycol], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            dropped = int(np.sum(~mask))
            self.x, self.y = x[mask], y[mask]
            msg = f"Loaded {len(self.x)} rows" + (f" (dropped {dropped} bad rows)" if dropped else "")
            self.data_info_lbl.setText(msg)
        except Exception as e:
            QMessageBox.critical(self, "CSV error", str(e))

    def use_pasted(self):
        try:
            xs = parse_number_list(self.txt_x.toPlainText())
            ys = parse_number_list(self.txt_y.toPlainText())
            if not xs or not ys or len(xs) != len(ys):
                QMessageBox.warning(self, "Paste", "X and Y must be same nonzero length.")
                return
            self.x = np.array(xs, dtype=float)
            self.y = np.array(ys, dtype=float)
            self.data_info_lbl.setText(f"Using pasted data: {len(self.x)} rows")
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))

    def _ensure_data(self) -> bool:
        if self.x is None or self.y is None:
            QMessageBox.warning(self, "Data", "Load CSV or paste X/Y first.")
            return False
        return True

    def fit_model(self):
        if not self._ensure_data():
            return
        try:
            order_mode = self.order_mode_combo.currentText()
            ridge_alpha = float(self.alpha_edit.text()) if self.use_ridge_chk.isChecked() else None
            kfold = int(self.kfold_spin.value())
            if order_mode == "Manual":
                degree = int(self.degree_spin.value())
                if len(self.x) < degree + 1:
                    QMessageBox.warning(self, "Degree", f"Need at least {degree+1} points for degree {degree}.")
                    return
                self.best = fit_degree(self.x, self.y, degree, ridge_alpha, kfold)
            else:
                maxdeg = int(self.maxdeg_spin.value())
                crit = self.criterion_combo.currentText()
                crit_key = "CV R2" if crit == "CV R2" else crit
                self.best = choose_best_degree(self.x, self.y, maxdeg, crit_key, ridge_alpha, kfold)
            # Update summary
            coefs = self.best["coefs"]
            self.eq_txt.setPlainText(format_poly_equation(coefs))
            self.var_r2.setText(f"{self.best['metrics']['r2']:.4f}")
            self.var_adj.setText(f"{self.best['adj_r2']:.4f}")
            self.var_cv.setText(f"{self.best['cv_r2_mean']:.4f}")
            self.var_rmse.setText(f"{self.best['metrics']['rmse']:.6g}")
            self.var_mae.setText(f"{self.best['metrics']['mae']:.6g}")
            self.var_aic.setText(f"{self.best['aic']:.3f}")
            self.var_bic.setText(f"{self.best['bic']:.3f}")
            self.draw_plots()
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))

    def do_predict(self):
        if not self.best:
            QMessageBox.warning(self, "Predict", "Fit a model first.")
            return
        try:
            xs = parse_number_list(self.forecast_x_edit.text())
            if not xs:
                QMessageBox.warning(self, "Predict", "Enter one or more X values.")
                return
            x_pred = np.array(xs, dtype=float)
            degree = int(self.best["degree"])
            Xp, _ = build_design_matrix(x_pred, degree, include_bias=True)
            y_pred = Xp @ self.best["coefs"]
            self.pred_table.setRowCount(0)
            for xp, yp in zip(x_pred, y_pred):
                row = self.pred_table.rowCount()
                self.pred_table.insertRow(row)
                self.pred_table.setItem(row, 0, QTableWidgetItem(f"{xp:g}"))
                self.pred_table.setItem(row, 1, QTableWidgetItem(f"{yp:g}"))
        except Exception as e:
            QMessageBox.critical(self, "Predict error", str(e))

    def do_solve(self):
        if not self.best:
            QMessageBox.warning(self, "Solve", "Fit a model first.")
            return
        try:
            y_target = float(self.solve_y_edit.text())
            roots = solve_for_x_from_y(self.best["coefs"], y_target)
            if roots:
                QMessageBox.information(self, "Solutions", "\n".join(f"X = {r:g}" for r in roots))
            else:
                QMessageBox.information(self, "Solutions", "No real-valued solutions for the given Y.")
        except Exception as e:
            QMessageBox.critical(self, "Solve error", str(e))

    def export_csv(self):
        if not self.best:
            QMessageBox.warning(self, "Export", "Fit a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        out = pd.DataFrame({
            "X": self.x,
            "Y": self.y,
            "Fitted_Y": self.best["y_hat"],
            "Residual": self.y - self.best["y_hat"],
        })
        out.to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Saved {path}")

    def export_json(self):
        if not self.best:
            QMessageBox.warning(self, "Export", "Fit a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if not path:
            return
        data = {
            "degree": int(self.best["degree"]),
            "coefs_increasing_power": [float(c) for c in self.best["coefs"]],
            "equation": format_poly_equation(self.best["coefs"]),
            "metrics": {
                "r2": float(self.best['metrics']['r2']),
                "adj_r2": float(self.best['adj_r2']),
                "rmse": float(self.best['metrics']['rmse']),
                "mae": float(self.best['metrics']['mae']),
                "aic": float(self.best['aic']),
                "bic": float(self.best['bic']),
                "cv_r2_mean": float(self.best['cv_r2_mean']),
                "cv_r2_scores": [float(s) for s in (self.best['cv_r2_scores'] or [])],
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))
        QMessageBox.information(self, "Export", f"Saved {path}")



if __name__ == "__main__":
    import sys
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        # Optional: nice stylesheet
        app.setStyleSheet("""
        QWidget { font-size: 12pt; }
        QLineEdit, QTextEdit { background: #f9f9fa; }
        QTabWidget::pane { border: 1px solid #bdbdbd; }
        QGroupBox { font-weight: bold; }
        """)
        win = PolyRegApp()
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        print("Error starting GUI:", e)