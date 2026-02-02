from __future__ import annotations

import math
import struct
import traceback
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes


# ============================================================
# Data model
# ============================================================
STRUCT_CODES = ["P", "G", "D", "I", "N"]
DIR_CODES = ["Z", "X", "XZ"]


@dataclass
class AppState:
    Kx: int = 2
    Ky: int = 2
    Kz: int = 2

    Sx: float = 20.0
    Sy: float = 20.0
    Sz: float = 20.0

    res_preview: int = 30
    res_export: int = 80

    typeA: str = "G"
    typeB: str = "G"
    dir: str = "Z"

    RD: float = 0.30

    trans_center: float = 0.5
    trans_k: float = 6.0

    visualStyle: str = "Ceramic Blue"
    theme: str = "Dark"
    lang: str = "中文"


# ============================================================
# TPMS implicit fields
# ============================================================
def get_field(code: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    Z = Z.astype(np.float32, copy=False)

    if code == "P":
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    if code == "G":
        return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    if code == "D":
        return (
            np.sin(X) * np.sin(Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.cos(Z)
            + np.cos(X) * np.sin(Y) * np.cos(Z)
            + np.cos(X) * np.cos(Y) * np.sin(Z)
        )
    if code == "I":
        return (
            2.0 * (np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X))
            - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))
        )
    if code == "N":
        return 3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4.0 * np.cos(X) * np.cos(Y) * np.cos(Z)

    return np.cos(X) + np.cos(Y) + np.cos(Z)


def weld_vertices(vertices: np.ndarray, faces: np.ndarray, decimals: int = 5):
    v_round = np.round(vertices, decimals=decimals)
    uniq, inv = np.unique(v_round, axis=0, return_inverse=True)
    return uniq.astype(np.float32, copy=False), inv[faces].astype(np.int32, copy=False)


def compute_tpms_verts_faces(state: AppState, res: int, max_voxels: int = 20_000_000):
    rx = max(int(round(state.Kx * res)), 8)
    ry = max(int(round(state.Ky * res)), 8)
    rz = max(int(round(state.Kz * res)), 8)

    vox = rx * ry * rz
    if vox > max_voxels:
        est_gb = vox * 4 * 4 / (1024**3)
        raise ValueError(
            f"网格规模过大：{vox/1e6:.2f}M 体素（上限 {max_voxels/1e6:.1f}M）。\n"
            f"估算内存占用可能 > {est_gb:.2f} GB。\n"
            f"请降低 Res 或周期数 K。"
        )

    gx = np.linspace(0.0, state.Kx * 2 * math.pi, rx, dtype=np.float32)
    gy = np.linspace(0.0, state.Ky * 2 * math.pi, ry, dtype=np.float32)
    gz = np.linspace(0.0, state.Kz * 2 * math.pi, rz, dtype=np.float32)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")

    PhiA = get_field(state.typeA, X, Y, Z)

    if state.typeA == state.typeB:
        PhiFinal = PhiA
    else:
        PhiB = get_field(state.typeB, X, Y, Z)

        xn = X / float(gx.max())
        zn = Z / float(gz.max())

        if state.dir == "Z":
            dist = Z / float(gz.max())
        elif state.dir == "X":
            dist = X / float(gx.max())
        else:
            dist = (xn - zn + 1.0) / 2.0

        w = 1.0 / (1.0 + np.exp(-state.trans_k * (dist - state.trans_center) * 10.0)).astype(np.float32)
        PhiFinal = (1.0 - w) * PhiA + w * PhiB

    absPhi = np.abs(PhiFinal).ravel()
    iso = float(np.quantile(absPhi, state.RD))
    sdf = (iso - np.abs(PhiFinal)).astype(np.float32)

    sx = state.Sx / max(rx - 1, 1)
    sy = state.Sy / max(ry - 1, 1)
    sz = state.Sz / max(rz - 1, 1)

    vol = np.transpose(sdf, (2, 1, 0))

    pad = 1
    neg_const = -float(np.max(np.abs(vol)) + 1.0)
    vol_pad = np.pad(vol, pad_width=pad, mode="constant", constant_values=neg_const)

    verts, faces, _, _ = marching_cubes(vol_pad, level=0.0, spacing=(sz, sy, sx))
    verts = np.asarray(verts, dtype=np.float32)
    verts[:, 0] -= pad * sz
    verts[:, 1] -= pad * sy
    verts[:, 2] -= pad * sx
    verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    return weld_vertices(verts_xyz, faces, decimals=5)


def compute_normals(v1, v2, v3):
    n = np.cross(v2 - v1, v3 - v1)
    nm = np.linalg.norm(n, axis=1)
    return (n / (nm[:, None] + 1e-12)).astype(np.float32, copy=False)


def write_stl_binary_with_progress(filename, faces, verts, progress_cb=None, cancel_flag_cb=None, chunk_size=5000):
    faces = faces.astype(np.int32, copy=False)
    verts = verts.astype(np.float32, copy=False)
    nF = faces.shape[0]

    with open(filename, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", nF))

        for i in range(0, nF, chunk_size):
            if cancel_flag_cb and cancel_flag_cb():
                break
            end = min(i + chunk_size, nF)
            batch = faces[i:end]

            v1 = verts[batch[:, 0]]
            v2 = verts[batch[:, 1]]
            v3 = verts[batch[:, 2]]
            n = compute_normals(v1, v2, v3)

            for k in range(batch.shape[0]):
                f.write(
                    struct.pack(
                        "<12fH",
                        float(n[k, 0]), float(n[k, 1]), float(n[k, 2]),
                        float(v1[k, 0]), float(v1[k, 1]), float(v1[k, 2]),
                        float(v2[k, 0]), float(v2[k, 1]), float(v2[k, 2]),
                        float(v3[k, 0]), float(v3[k, 1]), float(v3[k, 2]),
                        0
                    )
                )

            if progress_cb:
                progress_cb(end / nF)


def child_compute_entry(state_dict: dict, res: int, out_q: mp.Queue):
    try:
        st = AppState(**state_dict)
        verts, faces = compute_tpms_verts_faces(st, res)
        out_q.put(("ok", verts, faces))
    except Exception as e:
        out_q.put(("err", f"{e}\n\n{traceback.format_exc()}"))


# ============================================================
# GUI
# ============================================================
def run_gui():
    from PySide6.QtCore import Qt, QTimer, QObject, QEvent
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
        QFileDialog, QMessageBox, QProgressDialog, QSizePolicy, QScrollArea, QFrame
    )
    import pyvista as pv
    from pyvistaqt import QtInteractor

    THEMES = {
        "Dark": dict(
            panel_bg="#0f1012", panel_fg="#eaeaea", panel_border="#2a2b2f",
            canvas_bg=(0.06, 0.07, 0.08),
            grid_minor=(0.60, 0.60, 0.70),
            grid_major=(0.85, 0.85, 0.92),
            axis_color=(1.0, 1.0, 1.0),
        ),
        "Light": dict(
            panel_bg="#f3f4f6", panel_fg="#111827", panel_border="#d1d5db",
            canvas_bg=(0.98, 0.98, 0.985),
            grid_minor=(0.52, 0.52, 0.60),
            grid_major=(0.16, 0.16, 0.18),
            axis_color=(0.16, 0.16, 0.18),
        ),
    }

    STYLE_PRESETS = {
        "Ceramic Blue": dict(color=(0.14, 0.52, 0.96), ambient=0.10, diffuse=0.95, specular=0.85, spec_power=45),
        "Aged Copper":  dict(color=(0.95, 0.55, 0.18), ambient=0.10, diffuse=0.95, specular=0.82, spec_power=42),
        "Aurora Green": dict(color=(0.10, 0.82, 0.64), ambient=0.10, diffuse=0.95, specular=0.80, spec_power=42),
    }

    STRUCT_LABELS = {
        "中文": {"P": "Primitive（P）", "G": "Gyroid（G）", "D": "Diamond（D）", "I": "I-WP（I）", "N": "Neovius（N）"},
        "English": {"P": "Primitive (P)", "G": "Gyroid (G)", "D": "Diamond (D)", "I": "I-WP (I)", "N": "Neovius (N)"},
    }
    DIR_LABELS = {
        "中文": {"Z": "Z 向梯度（下→上）", "X": "X 向梯度（左→右）", "XZ": "对角梯度（X–Z）"},
        "English": {"Z": "Z-gradient (bottom→top)", "X": "X-gradient (left→right)", "XZ": "Diagonal gradient (X–Z)"},
    }
    UI = {
        "中文": {
            "title": "TPMS生成器",
            "lang": "语言",
            "theme": "主题",
            "A": "拓扑 A（主相）",
            "B": "拓扑 B（次相）",
            "dir": "梯度方向",
            "d0": "过渡中心 d0",
            "k": "过渡陡峭度 k",
            "rd": "相对密度 RD",
            "kxyz": "周期数 K（x,y,z）",
            "size": "尺寸 (mm)",
            "pre": "预览 Res",
            "exp": "导出 Res",
            "style": "渲染风格",
            "preview": "生成预览",
            "export": "导出 STL",
            "dirty": "状态：参数已修改，请点击“生成预览”。",
            "cancel": "取消",
            "need": "请先生成预览。",
            "err": "计算错误",
            "crash": "子进程异常退出（建议降低 Res 或 K）。",
            "write": "正在写入 STL（二进制）...",
            "ok": "导出成功！\n保存至:\n",
            "shot": "截图",
            "reset": "回正",
            "top": "俯视",
            "front": "前视",
            "right": "右视",
            "zin": "放大",
            "zout": "缩小",
            "pan": "平移",
            "rot": "旋转",
            "mouse_help": "提示：左键旋转，右键平移",
        },
        "English": {
            "title": "TPMS Generator",
            "lang": "Language",
            "theme": "Theme",
            "A": "Topology A (primary)",
            "B": "Topology B (secondary)",
            "dir": "Grading direction",
            "d0": "Transition center (d0)",
            "k": "Steepness (k)",
            "rd": "Target RD",
            "kxyz": "Periods K (x,y,z)",
            "size": "Size (mm)",
            "pre": "Preview Res",
            "exp": "Export Res",
            "style": "Render style",
            "preview": "Preview",
            "export": "Export STL",
            "dirty": "Status: parameters updated, click “Preview”.",
            "cancel": "Cancel",
            "need": "Please generate a preview first.",
            "err": "Compute error",
            "crash": "Worker process crashed (reduce Res or K).",
            "write": "Writing STL (binary)...",
            "ok": "Export successful!\nSaved to:\n",
            "shot": "Screenshot",
            "reset": "Reset",
            "top": "Top",
            "front": "Front",
            "right": "Right",
            "zin": "Zoom+",
            "zout": "Zoom-",
            "pan": "Pan",
            "rot": "Rotate",
            "mouse_help": "Tip: Left drag to rotate, right drag to pan",
        }
    }

    def t(st: AppState, key: str) -> str:
        return UI[st.lang][key]

    def struct_disp(st: AppState, code: str) -> str:
        return STRUCT_LABELS[st.lang][code]

    class ValueStepper(QWidget):
        def __init__(self, spinbox: QSpinBox | QDoubleSpinBox):
            super().__init__()
            self.spin = spinbox
            self.spin.setButtonSymbols(QSpinBox.NoButtons)
            
            self.btnMinus = QPushButton("−")
            self.btnPlus = QPushButton("+")
            for b in [self.btnMinus, self.btnPlus]:
                b.setFixedSize(24, 24)
                b.setCursor(Qt.PointingHandCursor)
            
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            layout.addWidget(self.btnMinus)
            layout.addWidget(self.spin, 1)
            layout.addWidget(self.btnPlus)
            
            self.btnMinus.clicked.connect(self.spin.stepDown)
            self.btnPlus.clicked.connect(self.spin.stepUp)

    def dir_disp(st: AppState, code: str) -> str:
        return DIR_LABELS[st.lang][code]

    def faces_to_pv(faces: np.ndarray) -> np.ndarray:
        n = faces.shape[0]
        out = np.empty((n, 4), dtype=np.int32)
        out[:, 0] = 3
        out[:, 1:] = faces
        return out.ravel()

    # ============================================================
    # MATLAB-like viewport controller (stable focal point)
    # ============================================================
    class MatlabViewportController(QObject):
        """
        MATLAB-like camera interaction with stable focal point:
          - LMB drag: orbit (no focal jump)
          - Shift+LMB drag: pan
          - Ctrl+LMB drag: dolly zoom (try focus on pick if valid)
          - Wheel: dolly zoom (try focus on pick if valid)
          - Double click: reset view
        """
        def __init__(self, plotter: QtInteractor, get_pick_point, on_reset, zoom_step_cb):
            super().__init__()
            self.plotter = plotter
            self.get_pick_point = get_pick_point
            self.on_reset = on_reset
            self.zoom_step_cb = zoom_step_cb
            self.force_pan = False
            self.force_orbit = False
            self._dragging = False
            self._mode = "orbit"
            self._last = None

        def set_force_pan(self, enabled: bool):
            self.force_pan = bool(enabled)

        def set_force_orbit(self, enabled: bool):
            self.force_orbit = bool(enabled)

        def install_on(self, widget):
            widget.installEventFilter(self)

        def _camera(self):
            return self.plotter.camera

        def _try_set_focal_to_pick(self):
            p = None
            try:
                p = self.get_pick_point()
            except Exception:
                p = None
            if p is None:
                return
            try:
                cam = self._camera()
                cam.focal_point = (float(p[0]), float(p[1]), float(p[2]))
            except Exception:
                pass

        def _dolly(self, factor: float, use_pick: bool):
            cam = self._camera()
            if use_pick:
                self._try_set_focal_to_pick()
            try:
                cam.Dolly(factor)
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pos = np.array(cam.position, dtype=float)
                fp = np.array(cam.focal_point, dtype=float)
                v = pos - fp
                if np.linalg.norm(v) < 1e-9:
                    return
                cam.position = tuple((fp + v / factor).tolist())

        def _pan(self, dx: float, dy: float, w: float, h: float):
            cam = self._camera()
            pos = np.array(cam.position, dtype=float)
            fp = np.array(cam.focal_point, dtype=float)
            up = np.array(cam.up, dtype=float)

            forward = fp - pos
            fn = np.linalg.norm(forward)
            if fn < 1e-9:
                return
            forward /= fn

            right = np.cross(forward, up)
            rn = np.linalg.norm(right)
            if rn < 1e-9:
                return
            right /= rn

            upn = np.cross(right, forward)
            upn /= (np.linalg.norm(upn) + 1e-12)

            w = max(float(w), 1.0)
            h = max(float(h), 1.0)
            dist = np.linalg.norm(fp - pos)
            scale = dist * 0.0015
            shift = (-right * dx + upn * dy) * scale

            cam.position = tuple((pos + shift).tolist())
            cam.focal_point = tuple((fp + shift).tolist())

        def _orbit(self, dx: float, dy: float, w: float, h: float):
            try:
                w = max(float(w), 1.0)
                h = max(float(h), 1.0)
                speed = 140.0
                self.plotter.camera.Azimuth(-dx / w * speed)
                self.plotter.camera.Elevation(dy / h * speed)
                self.plotter.camera.OrthogonalizeViewUp()
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass

        def eventFilter(self, obj, event):
            et = event.type()

            if et == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self.on_reset()
                return True

            if et == QEvent.MouseButtonPress and event.button() in (Qt.LeftButton, Qt.RightButton):
                self._dragging = True
                self._last = event.position()
                if event.button() == Qt.RightButton:
                    self._mode = "pan"
                else:
                    self._mode = "orbit"
                return True

            if et == QEvent.MouseMove and self._dragging and self._last is not None:
                p = event.position()
                dx = float(p.x() - self._last.x())
                dy = float(p.y() - self._last.y())
                self._last = p

                if self._mode == "orbit":
                    try:
                        w = obj.width()
                        h = obj.height()
                    except Exception:
                        w = self.plotter.width()
                        h = self.plotter.height()
                    self._orbit(dx, dy, w, h)
                elif self._mode == "pan":
                    try:
                        w = obj.width()
                        h = obj.height()
                    except Exception:
                        w = self.plotter.width()
                        h = self.plotter.height()
                    self._pan(dx, dy, w, h)
                else:
                    factor = 1.0 + (-dy) * 0.01
                    factor = max(0.25, min(4.0, factor))
                    self._dolly(factor, use_pick=True)

                self.plotter.update()
                return True

            if et == QEvent.MouseButtonRelease and event.button() in (Qt.LeftButton, Qt.RightButton):
                self._dragging = False
                self._last = None
                return True

            if et == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta == 0:
                    return True
                if self.zoom_step_cb:
                    self.zoom_step_cb(delta > 0)
                else:
                    factor = 1.10 if delta > 0 else 0.90
                    self._dolly(factor, use_pick=False)
                return True

            return super().eventFilter(obj, event)

    # ============================================================
    # Main window
    # ============================================================
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.state = AppState()

            self._axis_font_base = 14

            self.cache_mesh = None
            self._proc = None
            self._q = None
            self._timer = None

            self._bounds_actor = None
            self._grid_minor_actor = None
            self._grid_major_actor = None
            self._force_pan = False
            self._force_orbit = True

            root = QWidget()
            self.setCentralWidget(root)
            layout = QHBoxLayout(root)

            self.cfg = QWidget()
            self.cfg.setFixedWidth(460)
            left = QVBoxLayout(self.cfg)
            left.setContentsMargins(10, 10, 10, 10)
            left.setSpacing(10)

            g0 = QGroupBox("UI")
            g0l = QVBoxLayout(g0)
            r = QHBoxLayout()
            self.lblLang = QLabel()
            self.cmbLang = QComboBox(); self.cmbLang.addItems(["中文", "English"]); self.cmbLang.setCurrentText(self.state.lang)
            r.addWidget(self.lblLang); r.addWidget(self.cmbLang); g0l.addLayout(r)
            r = QHBoxLayout()
            self.lblTheme = QLabel()
            self.cmbTheme = QComboBox(); self.cmbTheme.addItems(["Dark", "Light"]); self.cmbTheme.setCurrentText(self.state.theme)
            r.addWidget(self.lblTheme); r.addWidget(self.cmbTheme); g0l.addLayout(r)
            left.addWidget(g0)

            g1 = QGroupBox("Topology")
            g1l = QVBoxLayout(g1)
            r = QHBoxLayout()
            self.lblA = QLabel()
            self.cmbA = QComboBox()
            self.cmbA.addItems([struct_disp(self.state, c) for c in STRUCT_CODES])
            self.cmbA.setCurrentIndex(STRUCT_CODES.index(self.state.typeA))
            r.addWidget(self.lblA); r.addWidget(self.cmbA); g1l.addLayout(r)

            r = QHBoxLayout()
            self.lblB = QLabel()
            self.cmbB = QComboBox()
            self.cmbB.addItems([struct_disp(self.state, c) for c in STRUCT_CODES])
            self.cmbB.setCurrentIndex(STRUCT_CODES.index(self.state.typeB))
            r.addWidget(self.lblB); r.addWidget(self.cmbB); g1l.addLayout(r)
            left.addWidget(g1)

            g2 = QGroupBox("Grading")
            g2l = QVBoxLayout(g2)
            r = QHBoxLayout()
            self.lblDir = QLabel()
            self.cmbDir = QComboBox()
            self.cmbDir.addItems([dir_disp(self.state, c) for c in DIR_CODES])
            self.cmbDir.setCurrentIndex(DIR_CODES.index(self.state.dir))
            r.addWidget(self.lblDir); r.addWidget(self.cmbDir); g2l.addLayout(r)

            r = QHBoxLayout()
            self.lblD0 = QLabel()
            self.spD0 = QDoubleSpinBox(); self.spD0.setRange(0, 1); self.spD0.setDecimals(2); self.spD0.setSingleStep(0.05); self.spD0.setValue(self.state.trans_center)
            r.addWidget(self.lblD0); r.addWidget(ValueStepper(self.spD0)); g2l.addLayout(r)

            r = QHBoxLayout()
            self.lblK = QLabel()
            self.spK = QDoubleSpinBox(); self.spK.setRange(0.1, 50); self.spK.setDecimals(2); self.spK.setSingleStep(0.5); self.spK.setValue(self.state.trans_k)
            r.addWidget(self.lblK); r.addWidget(ValueStepper(self.spK)); g2l.addLayout(r)
            left.addWidget(g2)

            g3 = QGroupBox("Geometry")
            g3l = QGridLayout(g3)
            g3l.setHorizontalSpacing(10)
            g3l.setVerticalSpacing(10)
            g3l.setColumnStretch(0, 0)
            g3l.setColumnStretch(1, 1)
            g3l.setColumnStretch(2, 1)
            g3l.setColumnStretch(3, 1)

            self.lblRD = QLabel()
            self.spRD = QDoubleSpinBox(); self.spRD.setRange(0.01, 0.99); self.spRD.setDecimals(2); self.spRD.setSingleStep(0.05); self.spRD.setValue(self.state.RD)
            g3l.addWidget(self.lblRD, 0, 0)
            g3l.addWidget(ValueStepper(self.spRD), 0, 1, 1, 3)

            self.lblKXYZ = QLabel()
            self.spKx = QSpinBox(); self.spKx.setRange(1, 50); self.spKx.setValue(self.state.Kx)
            self.spKy = QSpinBox(); self.spKy.setRange(1, 50); self.spKy.setValue(self.state.Ky)
            self.spKz = QSpinBox(); self.spKz.setRange(1, 50); self.spKz.setValue(self.state.Kz)
            g3l.addWidget(self.lblKXYZ, 1, 0)
            g3l.addWidget(ValueStepper(self.spKx), 1, 1)
            g3l.addWidget(ValueStepper(self.spKy), 1, 2)
            g3l.addWidget(ValueStepper(self.spKz), 1, 3)

            self.lblSize = QLabel()
            self.spSx = QDoubleSpinBox(); self.spSx.setRange(1, 500); self.spSx.setDecimals(1); self.spSx.setValue(self.state.Sx)
            self.spSy = QDoubleSpinBox(); self.spSy.setRange(1, 500); self.spSy.setDecimals(1); self.spSy.setValue(self.state.Sy)
            self.spSz = QDoubleSpinBox(); self.spSz.setRange(1, 500); self.spSz.setDecimals(1); self.spSz.setValue(self.state.Sz)
            g3l.addWidget(self.lblSize, 2, 0)
            g3l.addWidget(ValueStepper(self.spSx), 2, 1)
            g3l.addWidget(ValueStepper(self.spSy), 2, 2)
            g3l.addWidget(ValueStepper(self.spSz), 2, 3)

            self.lblPre = QLabel()
            self.spPre = QSpinBox(); self.spPre.setRange(10, 300); self.spPre.setValue(self.state.res_preview)
            g3l.addWidget(self.lblPre, 3, 0)
            g3l.addWidget(ValueStepper(self.spPre), 3, 1, 1, 3)

            self.lblExp = QLabel()
            self.spExp = QSpinBox(); self.spExp.setRange(20, 600); self.spExp.setValue(self.state.res_export)
            g3l.addWidget(self.lblExp, 4, 0)
            g3l.addWidget(ValueStepper(self.spExp), 4, 1, 1, 3)

            for lab in [self.lblRD, self.lblKXYZ, self.lblSize, self.lblPre, self.lblExp]:
                lab.setMinimumWidth(140)

            left.addWidget(g3)

            g4 = QGroupBox("Render")
            g4l = QVBoxLayout(g4)
            r = QHBoxLayout()
            self.lblStyle = QLabel()
            self.cmbStyle = QComboBox()
            self.cmbStyle.addItems(list(STYLE_PRESETS.keys()))
            self.cmbStyle.setCurrentText(self.state.visualStyle)
            r.addWidget(self.lblStyle); r.addWidget(self.cmbStyle); g4l.addLayout(r)

            r = QHBoxLayout()
            self.btnPreview = QPushButton()
            self.btnExport = QPushButton()
            self.btnPreview.setMinimumHeight(44)
            self.btnExport.setMinimumHeight(44); self.btnExport.setEnabled(False)
            r.addWidget(self.btnPreview); r.addWidget(self.btnExport)
            g4l.addLayout(r)

            self.lblStatus = QLabel()
            self.lblStatus.setWordWrap(True)
            g4l.addWidget(self.lblStatus)
            left.addWidget(g4)

            left.addStretch(1)

            view = QWidget()
            v = QVBoxLayout(view); v.setContentsMargins(10, 10, 10, 10); v.setSpacing(6)

            bar = QWidget(); bl = QHBoxLayout(bar); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(6)
            self.btnReset = QPushButton()
            self.btnTop = QPushButton()
            self.btnFront = QPushButton()
            self.btnRight = QPushButton()
            self.btnZoomIn = QPushButton()
            self.btnZoomOut = QPushButton()
            self.btnShot = QPushButton()
            for b in [self.btnReset, self.btnTop, self.btnFront, self.btnRight, self.btnZoomIn, self.btnZoomOut, self.btnShot]:
                b.setFixedHeight(28)
            bl.addWidget(self.btnReset); bl.addWidget(self.btnTop); bl.addWidget(self.btnFront); bl.addWidget(self.btnRight)
            bl.addStretch(1)
            bl.addWidget(self.btnZoomIn); bl.addWidget(self.btnZoomOut); bl.addWidget(self.btnShot)
            v.addWidget(bar)

            self.lblMouseHelp = QLabel()
            self.lblMouseHelp.setObjectName("mouseHelp")
            self.lblMouseHelp.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            v.addWidget(self.lblMouseHelp)

            self.plotter = QtInteractor(view)
            self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            v.addWidget(self.plotter, 1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(self.cfg)
            scroll.setMinimumWidth(460)
            scroll.setMaximumWidth(520)
            layout.addWidget(scroll)
            layout.addWidget(view, 1)

            pv.global_theme.smooth_shading = True

            self._apply_language()
            self._apply_theme()

            QTimer.singleShot(0, self._setup_viewport)
            self._wire()

        def _apply_language(self):
            self.setWindowTitle(t(self.state, "title"))
            self.lblLang.setText(t(self.state, "lang"))
            self.lblTheme.setText(t(self.state, "theme"))
            self.lblA.setText(t(self.state, "A"))
            self.lblB.setText(t(self.state, "B"))
            self.lblDir.setText(t(self.state, "dir"))
            self.lblD0.setText(t(self.state, "d0"))
            self.lblK.setText(t(self.state, "k"))
            self.lblRD.setText(t(self.state, "rd"))
            self.lblKXYZ.setText(t(self.state, "kxyz"))
            self.lblSize.setText(t(self.state, "size"))
            self.lblPre.setText(t(self.state, "pre"))
            self.lblExp.setText(t(self.state, "exp"))
            self.lblStyle.setText(t(self.state, "style"))
            self.btnPreview.setText(t(self.state, "preview"))
            self.btnExport.setText(t(self.state, "export"))
            self.lblStatus.setText(t(self.state, "dirty"))
            self.btnReset.setText(t(self.state, "reset"))
            self.btnTop.setText(t(self.state, "top"))
            self.btnFront.setText(t(self.state, "front"))
            self.btnRight.setText(t(self.state, "right"))
            self.btnZoomIn.setText(t(self.state, "zin"))
            self.btnZoomOut.setText(t(self.state, "zout"))
            self.btnShot.setText(t(self.state, "shot"))
            self.lblMouseHelp.setText(t(self.state, "mouse_help"))

            a = self.cmbA.currentIndex()
            b = self.cmbB.currentIndex()
            d = self.cmbDir.currentIndex()
            self.cmbA.blockSignals(True); self.cmbB.blockSignals(True); self.cmbDir.blockSignals(True)
            self.cmbA.clear(); self.cmbB.clear(); self.cmbDir.clear()
            self.cmbA.addItems([struct_disp(self.state, c) for c in STRUCT_CODES])
            self.cmbB.addItems([struct_disp(self.state, c) for c in STRUCT_CODES])
            self.cmbDir.addItems([dir_disp(self.state, c) for c in DIR_CODES])
            self.cmbA.setCurrentIndex(a if a >= 0 else STRUCT_CODES.index(self.state.typeA))
            self.cmbB.setCurrentIndex(b if b >= 0 else STRUCT_CODES.index(self.state.typeB))
            self.cmbDir.setCurrentIndex(d if d >= 0 else DIR_CODES.index(self.state.dir))
            self.cmbA.blockSignals(False); self.cmbB.blockSignals(False); self.cmbDir.blockSignals(False)

        def _apply_theme(self):
            th = THEMES[self.state.theme]
            help_bg = "rgba(255,255,255,0.10)" if self.state.theme == "Dark" else "rgba(0,0,0,0.06)"
            self.setStyleSheet(
                f"""
                QMainWindow, QWidget {{
                    background: {th['panel_bg']};
                    color: {th['panel_fg']};
                    font-size: 12px;
                }}
                QScrollArea, QFrame {{
                    background: {th['panel_bg']};
                }}
                QGroupBox {{
                    border: 1px solid {th['panel_border']};
                    border-radius: 8px;
                    margin-top: 10px;
                    padding: 10px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 6px 0 6px;
                    font-weight: 600;
                }}
                QComboBox, QSpinBox, QDoubleSpinBox {{
                    background: rgba(255,255,255,0.08);
                    border: 1px solid {th['panel_border']};
                    border-radius: 6px;
                }}
                QComboBox {{
                    padding: 4px;
                }}
                QSpinBox, QDoubleSpinBox {{
                    padding: 3px 2px;
                }}
                QPushButton {{
                    border: 1px solid {th['panel_border']};
                    border-radius: 8px;
                    padding: 6px 10px;
                    background: rgba(255,255,255,0.05);
                }}
                QPushButton:hover {{
                    background: rgba(255,255,255,0.12);
                }}
                QPushButton:pressed {{
                    background: rgba(255,255,255,0.2);
                }}
                QPushButton:disabled {{
                    color: rgba(160,160,160,0.8);
                    background: transparent;
                }}

                QLabel#mouseHelp {{
                    font-size: 12px;
                    font-weight: 600;
                    padding: 6px 10px;
                    border-radius: 6px;
                    border: 1px solid {th['panel_border']};
                    background: {help_bg};
                }}

                ValueStepper QPushButton {{
                    background: rgba(255,255,255,0.1);
                    border: 1px solid {th['panel_border']};
                    border-radius: 4px;
                    padding: 0px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                ValueStepper QPushButton:hover {{
                    background: {th['panel_fg']}22;
                }}
                """
            )
            self.plotter.set_background(th["canvas_bg"])
            self.plotter.update()

        def _pick_under_cursor(self):
            """Return picked world point or None. Never return garbage."""
            try:
                p = self.plotter.pick_mouse_position()
                if p is None:
                    return None
                p = np.asarray(p, dtype=float)
                if p.shape != (3,) or (not np.all(np.isfinite(p))):
                    return None

                if (p[0] < 0 or p[0] > self.state.Sx or
                    p[1] < 0 or p[1] > self.state.Sy or
                    p[2] < 0 or p[2] > self.state.Sz):
                    return None
                return p
            except Exception:
                return None

        def _ensure_bounds_actor(self):
            import pyvista as pv
            if self._bounds_actor is not None:
                try:
                    self.plotter.remove_actor(self._bounds_actor)
                except Exception:
                    pass
                self._bounds_actor = None

            box = pv.Box(bounds=(0, self.state.Sx, 0, self.state.Sy, 0, self.state.Sz))
            self._bounds_actor = self.plotter.add_mesh(box, opacity=0.0, pickable=False)

        def _show_axes_box(self):
            th = THEMES[self.state.theme]
            try:
                dpr = 1.0
                try:
                    dpr = float(self.plotter.devicePixelRatioF())
                except Exception:
                    dpr = 1.0
                axis_font = max(8, int(round(self._axis_font_base / max(dpr, 0.25))))
                eps = 1e-6
                self.plotter.show_grid(
                    xtitle="X (mm)",
                    ytitle="Y (mm)",
                    ztitle="Z (mm)",
                    color=th["axis_color"],
                    grid="back",
                    location="outer",
                    ticks="both",
                    minor_ticks=True,
                    font_size=axis_font,
                    fmt="%.0f",
                    bounds=(eps, self.state.Sx, eps, self.state.Sy, eps, self.state.Sz),
                )
            except Exception:
                try:
                    self.plotter.add_axes()
                except Exception:
                    pass

            try:
                self.plotter.add_orientation_widget()
            except Exception:
                pass

        def _draw_floor_grids(self):
            import pyvista as pv
            th = THEMES[self.state.theme]

            for a in [self._grid_minor_actor, self._grid_major_actor]:
                if a is not None:
                    try:
                        self.plotter.remove_actor(a)
                    except Exception:
                        pass
            self._grid_minor_actor = None
            self._grid_major_actor = None

            def is_major(v: float) -> bool:
                return abs(v - round(v / 5.0) * 5.0) < 1e-6

            def build_grid_lines(step: float = 1.0):
                sx = float(self.state.Sx)
                sy = float(self.state.Sy)

                xs = [float(i) for i in range(0, int(math.floor(sx)) + 1)]
                ys = [float(i) for i in range(0, int(math.floor(sy)) + 1)]
                if abs(xs[-1] - sx) > 1e-6:
                    xs.append(sx)
                if abs(ys[-1] - sy) > 1e-6:
                    ys.append(sy)

                minor_pts = []
                minor_lines = []
                major_pts = []
                major_lines = []

                def add_line(p0, p1, major: bool):
                    if major:
                        idx = len(major_pts)
                        major_pts.extend([p0, p1])
                        major_lines.extend([2, idx, idx + 1])
                    else:
                        idx = len(minor_pts)
                        minor_pts.extend([p0, p1])
                        minor_lines.extend([2, idx, idx + 1])

                for x in xs:
                    add_line((x, 0.0, 0.0), (x, sy, 0.0), major=is_major(x))

                for y in ys:
                    add_line((0.0, y, 0.0), (sx, y, 0.0), major=is_major(y))

                minor_mesh = pv.PolyData(np.array(minor_pts, dtype=np.float32))
                if minor_lines:
                    minor_mesh.lines = np.array(minor_lines, dtype=np.int32)
                major_mesh = pv.PolyData(np.array(major_pts, dtype=np.float32))
                if major_lines:
                    major_mesh.lines = np.array(major_lines, dtype=np.int32)

                return minor_mesh, major_mesh

            minor, major = build_grid_lines(step=1.0)

            if minor.n_points > 0:
                self._grid_minor_actor = self.plotter.add_mesh(
                    minor,
                    color=th["grid_minor"],
                    line_width=1.05,
                    opacity=0.38 if self.state.theme == "Dark" else 0.45,
                    pickable=False
                )

            if major.n_points > 0:
                self._grid_major_actor = self.plotter.add_mesh(
                    major,
                    color=th["grid_major"],
                    line_width=2.2,
                    opacity=0.65 if self.state.theme == "Dark" else 0.72,
                    pickable=False
                )

        def _apply_lights(self):
            """Balanced, MATLAB-like lighting (no blown highlights / no crushed shadows)."""
            try:
                self.plotter.remove_all_lights()
            except Exception:
                pass

            try:
                self.plotter.enable_lightkit()
            except Exception:
                pass

            import pyvista as pv
            head = pv.Light(light_type="headlight")
            head.intensity = 0.35 if self.state.theme == "Dark" else 0.22
            self.plotter.add_light(head)

        def _enable_quality(self):
            try:
                self.plotter.enable_anti_aliasing("ssaa")
            except Exception:
                try:
                    self.plotter.enable_anti_aliasing("fxaa")
                except Exception:
                    pass

            try:
                self.plotter.enable_eye_dome_lighting()
            except Exception:
                pass

            try:
                self.plotter.disable_shadows()
            except Exception:
                pass

        def _reset_view(self):
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.plotter.update()

        def _setup_viewport(self):
            th = THEMES[self.state.theme]
            self.plotter.clear()
            self.plotter.set_background(th["canvas_bg"])

            self._ensure_bounds_actor()
            self._show_axes_box()
            self._draw_floor_grids()
            self._enable_quality()
            self._apply_lights()

            try:
                self.plotter.disable()
            except Exception:
                pass

            self._vp = MatlabViewportController(
                plotter=self.plotter,
                get_pick_point=self._pick_under_cursor,
                on_reset=self._reset_view,
                zoom_step_cb=self._zoom_step
            )
            self._vp.set_force_pan(self._force_pan)
            self._vp.set_force_orbit(self._force_orbit)
            self._vp.install_on(self.plotter)

            self._reset_view()

        def _render_mesh(self, mesh):
            th = THEMES[self.state.theme]
            self.plotter.clear()
            self.plotter.set_background(th["canvas_bg"])

            self._ensure_bounds_actor()
            self._show_axes_box()
            self._draw_floor_grids()
            self._enable_quality()
            self._apply_lights()

            preset = STYLE_PRESETS[self.state.visualStyle]
            self.plotter.add_mesh(
                mesh,
                color=preset["color"],
                smooth_shading=True,
                lighting=True,
                ambient=preset["ambient"],
                diffuse=preset["diffuse"],
                specular=preset["specular"],
                specular_power=preset["spec_power"],
            )

            self._reset_view()

        def _start_child_compute(self, res, title, msg, on_done):
            self._stop_child()

            self._q = mp.Queue()
            self._proc = mp.Process(target=child_compute_entry, args=(self.state.__dict__, res, self._q), daemon=True)
            self._proc.start()

            dlg = QProgressDialog(msg, t(self.state, "cancel"), 0, 0, self)
            dlg.setWindowTitle(title)
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.show()

            def cancel():
                self._stop_child()
                dlg.close()
                self.lblStatus.setText(t(self.state, "cancel"))

            dlg.canceled.connect(cancel)

            self._timer = QTimer(self)
            self._timer.setInterval(120)

            def poll():
                if self._q is None:
                    return
                try:
                    while not self._q.empty():
                        item = self._q.get_nowait()
                        if item[0] == "ok":
                            _, verts, faces = item
                            self._stop_child()
                            dlg.close()
                            on_done(verts, faces)
                            return
                        else:
                            _, err = item
                            self._stop_child()
                            dlg.close()
                            QMessageBox.critical(self, t(self.state, "err"), err)
                            return
                except Exception:
                    pass

                if self._proc is not None and (not self._proc.is_alive()):
                    self._stop_child()
                    dlg.close()
                    QMessageBox.critical(self, t(self.state, "err"), t(self.state, "crash"))

            self._timer.timeout.connect(poll)
            self._timer.start()

        def _stop_child(self):
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            if self._proc is not None:
                try:
                    if self._proc.is_alive():
                        self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            if self._q is not None:
                try:
                    self._q.close()
                except Exception:
                    pass
                self._q = None

        def _do_preview(self):
            st = AppState(**self.state.__dict__)

            def done(verts, faces):
                import pyvista as pv
                mesh = pv.PolyData(verts, faces_to_pv(faces)).clean(tolerance=1e-6).triangulate()
                try:
                    mesh.points = np.clip(
                        mesh.points,
                        [0.0, 0.0, 0.0],
                        [self.state.Sx, self.state.Sy, self.state.Sz]
                    )
                except Exception:
                    pass

                try:
                    if mesh.n_cells > 700_000:
                        mesh = mesh.decimate_pro(0.82)
                except Exception:
                    pass
                self.cache_mesh = mesh
                self._render_mesh(mesh)
                self.btnExport.setEnabled(True)
                self.lblStatus.setText("OK")

            self.lblStatus.setText("...")
            self._start_child_compute(st.res_preview, "Preview", "Computing mesh ...", done)

        def _do_export(self):
            if self.cache_mesh is None:
                QMessageBox.information(self, "Info", t(self.state, "need"))
                return

            filepath, _ = QFileDialog.getSaveFileName(self, "Export STL", "TPMS_Model.stl", "STL Files (*.stl)")
            if not filepath:
                return

            st = AppState(**self.state.__dict__)

            def after_compute(verts, faces):
                dlg = QProgressDialog(t(self.state, "write"), t(self.state, "cancel"), 0, 100, self)
                dlg.setWindowTitle("Export STL")
                dlg.setWindowModality(Qt.WindowModal)
                dlg.setMinimumDuration(0)
                dlg.setAutoClose(False)
                dlg.setAutoReset(False)
                dlg.setValue(0)
                dlg.show()

                cancelled = {"v": False}
                dlg.canceled.connect(lambda: cancelled.__setitem__("v", True))

                def prog(p):
                    dlg.setValue(int(max(0, min(1, p)) * 100))

                try:
                    write_stl_binary_with_progress(
                        filepath, faces, verts,
                        progress_cb=prog,
                        cancel_flag_cb=lambda: cancelled["v"]
                    )
                    dlg.close()
                    if cancelled["v"]:
                        return
                    QMessageBox.information(self, "Done", t(self.state, "ok") + filepath)
                except Exception as e:
                    dlg.close()
                    QMessageBox.critical(self, "Error", f"{e}\n\n{traceback.format_exc()}")

            self._start_child_compute(st.res_export, "High-Res Export", "Reconstructing mesh ...", after_compute)

        def _set_view(self, kind):
            if kind == "top":
                self.plotter.view_xy()
            elif kind == "front":
                self.plotter.view_xz()
            elif kind == "right":
                self.plotter.view_yz()
            self.plotter.update()

        def _zoom_step(self, zoom_in: bool):
            cam = self.plotter.camera
            factor = 1.10 if zoom_in else 0.90
            try:
                cam.Dolly(factor)
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass
            self.plotter.update()

        def _screenshot(self):
            if self.cache_mesh is None:
                QMessageBox.information(self, "Info", t(self.state, "need"))
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "TPMS_View.png", "PNG (*.png)")
            if not path:
                return
            try:
                self.plotter.screenshot(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"{e}\n\n{traceback.format_exc()}")

        def _wire(self):
            def dirty():
                self.cache_mesh = None
                self.btnExport.setEnabled(False)
                self.lblStatus.setText(t(self.state, "dirty"))

            self.cmbLang.currentTextChanged.connect(lambda v: (setattr(self.state, "lang", v), self._apply_language(), dirty()))
            self.cmbTheme.currentTextChanged.connect(lambda v: (setattr(self.state, "theme", v), self._apply_theme(), self._setup_viewport(), self._rerender_only()))

            self.cmbA.currentIndexChanged.connect(lambda i: (setattr(self.state, "typeA", STRUCT_CODES[i]), dirty()))
            self.cmbB.currentIndexChanged.connect(lambda i: (setattr(self.state, "typeB", STRUCT_CODES[i]), dirty()))
            self.cmbDir.currentIndexChanged.connect(lambda i: (setattr(self.state, "dir", DIR_CODES[i]), dirty()))

            self.spD0.valueChanged.connect(lambda v: (setattr(self.state, "trans_center", float(v)), dirty()))
            self.spK.valueChanged.connect(lambda v: (setattr(self.state, "trans_k", float(v)), dirty()))
            self.spRD.valueChanged.connect(lambda v: (setattr(self.state, "RD", float(v)), dirty()))

            self.spKx.valueChanged.connect(lambda v: (setattr(self.state, "Kx", int(v)), dirty()))
            self.spKy.valueChanged.connect(lambda v: (setattr(self.state, "Ky", int(v)), dirty()))
            self.spKz.valueChanged.connect(lambda v: (setattr(self.state, "Kz", int(v)), dirty()))

            def size_changed():
                self.state.Sx = float(self.spSx.value())
                self.state.Sy = float(self.spSy.value())
                self.state.Sz = float(self.spSz.value())
                dirty()
                self._setup_viewport()
                self._rerender_only()

            self.spSx.valueChanged.connect(lambda _: size_changed())
            self.spSy.valueChanged.connect(lambda _: size_changed())
            self.spSz.valueChanged.connect(lambda _: size_changed())

            self.spPre.valueChanged.connect(lambda v: (setattr(self.state, "res_preview", int(v)), dirty()))
            self.spExp.valueChanged.connect(lambda v: (setattr(self.state, "res_export", int(v)), dirty()))

            self.cmbStyle.currentTextChanged.connect(lambda s: (setattr(self.state, "visualStyle", s), self._rerender_only()))

            self.btnPreview.clicked.connect(self._do_preview)
            self.btnExport.clicked.connect(self._do_export)

            self.btnTop.clicked.connect(lambda: self._set_view("top"))
            self.btnFront.clicked.connect(lambda: self._set_view("front"))
            self.btnRight.clicked.connect(lambda: self._set_view("right"))
            self.btnReset.clicked.connect(self._reset_view)

            self.btnZoomIn.clicked.connect(lambda: self._zoom_step(True))
            self.btnZoomOut.clicked.connect(lambda: self._zoom_step(False))
            self.btnShot.clicked.connect(self._screenshot)

        def _rerender_only(self):
            if self.cache_mesh is not None:
                self._render_mesh(self.cache_mesh)

        def closeEvent(self, event):
            self._stop_child()
            super().closeEvent(event)

    app = QApplication([])
    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    app.exec()


def main():
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    run_gui()


if __name__ == "__main__":
    main()
