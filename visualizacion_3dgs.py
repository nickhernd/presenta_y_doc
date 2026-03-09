#!/usr/bin/env python3
"""
Visualización Interactiva – 3D Gaussian Splatting
==================================================
Demostración en tiempo real de los parámetros clave de 3DGS.

Paneles:
  1. Gaussiana 3D          – elipsoide con sliders de escala y rotación
  2. Factorización Σ=RSS^TR^T – cómo R y S construyen la covarianza
  3. Alpha Blending        – acumulación de transmitancia por rayo
  4. Spherical Harmonics   – color dependiente de la dirección de vista

Uso:
  python visualizacion_3dgs.py
  (requiere: numpy, matplotlib, scipy)
"""

import numpy as np
import matplotlib
# Probar backends disponibles en orden
for _backend in ("Qt5Agg", "Qt6Agg", "GTK3Agg", "GTK4Agg", "wxAgg"):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test
        _plt_test.figure(); _plt_test.close("all")
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────────────────────
# Paleta de colores corporativa
# ──────────────────────────────────────────────────────────────
C_BG      = "#0f0f1a"   # fondo oscuro
C_PANEL   = "#1a1a2e"
C_ACCENT  = "#e94560"
C_BLUE    = "#16213e"
C_GOLD    = "#f5a623"
C_GREEN   = "#4ecca3"
C_TEXT    = "#eaeaea"
C_SLIDER  = "#2d2d4e"

plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_PANEL,
    "axes.edgecolor":    C_TEXT,
    "axes.labelcolor":   C_TEXT,
    "xtick.color":       C_TEXT,
    "ytick.color":       C_TEXT,
    "text.color":        C_TEXT,
    "grid.color":        "#333355",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
})

# ══════════════════════════════════════════════════════════════
#  UTILIDADES MATEMÁTICAS
# ══════════════════════════════════════════════════════════════

def euler_to_R(rx_deg, ry_deg, rz_deg):
    """Matriz de rotación desde ángulos de Euler (ZYX)."""
    return Rotation.from_euler("zyx",
                               [rz_deg, ry_deg, rx_deg],
                               degrees=True).as_matrix()


def build_covariance(sx, sy, sz, rx, ry, rz):
    """Σ = R S S^T R^T  (factorización 3DGS)."""
    R = euler_to_R(rx, ry, rz)
    S = np.diag([sx, sy, sz])
    return R @ S @ S.T @ R.T


def ellipsoid_surface(sigma, n=30):
    """Superficie paramétrica del elipsoide 1-sigma dado Σ."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], axis=-1)          # (n,n,3)
    vals, vecs = np.linalg.eigh(sigma)
    radii = np.sqrt(np.maximum(vals, 1e-8))
    T = vecs * radii[np.newaxis, :]              # scale eigenvectors
    pts_t = pts @ T.T
    return pts_t[..., 0], pts_t[..., 1], pts_t[..., 2]


# ──────────────────────── Spherical Harmonics ─────────────────────────
def sh_Y00(theta, phi):  return np.full_like(theta, 0.28209)
def sh_Y1n1(t, p): return 0.48860 * np.sin(t) * np.sin(p)
def sh_Y10 (t, p): return 0.48860 * np.cos(t)
def sh_Y1p1(t, p): return 0.48860 * np.sin(t) * np.cos(p)
def sh_Y2n2(t, p): return 1.09255 * np.sin(t)**2 * np.sin(2*p)
def sh_Y2n1(t, p): return 1.09255 * np.sin(t) * np.cos(t) * np.sin(p)
def sh_Y20 (t, p): return 0.31539 * (3*np.cos(t)**2 - 1)
def sh_Y2p1(t, p): return 1.09255 * np.sin(t) * np.cos(t) * np.cos(p)
def sh_Y2p2(t, p): return 0.54628 * np.sin(t)**2 * np.cos(2*p)

SH_FUNCS = [sh_Y00, sh_Y1n1, sh_Y10, sh_Y1p1,
            sh_Y2n2, sh_Y2n1, sh_Y20, sh_Y2p1, sh_Y2p2]
SH_NAMES = ["Y₀⁰","Y₁⁻¹","Y₁⁰","Y₁⁺¹",
             "Y₂⁻²","Y₂⁻¹","Y₂⁰","Y₂⁺¹","Y₂⁺²"]


def sphere_grid(n=50):
    """Malla esférica estándar."""
    phi   = np.linspace(0, 2*np.pi, n)
    theta = np.linspace(0, np.pi,   n)
    P, T  = np.meshgrid(phi, theta)
    return T, P


def sh_color_on_sphere(coefs, T, P):
    """Calcula el color RGB sobre la esfera dado un vector de 9 coefs (L=0,1,2)."""
    val = np.zeros_like(T)
    for c, f in zip(coefs, SH_FUNCS):
        val += c * f(T, P)
    # mapear a [0,1]
    val = (val - val.min()) / (val.ptp() + 1e-8)
    r = val
    g = np.roll(val, shift=7, axis=1)  # pequeño desfase por canal
    b = np.roll(val, shift=14, axis=1)
    rgba = np.stack([r, g, b, np.ones_like(r)], axis=-1)
    rgba = np.clip(rgba, 0, 1)
    return rgba


# ══════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL
# ══════════════════════════════════════════════════════════════

class Viz3DGS:
    # ------ parámetros por defecto ------
    DEF = dict(sx=1.0, sy=0.5, sz=0.3,
               rx=20., ry=30., rz=0.,
               opacity=0.8)

    NAV_LABELS = ["① Gaussiana 3D",
                  "② Factorización Σ",
                  "③ Alpha Blending",
                  "④ Spherical Harmonics"]

    def __init__(self):
        self.panel = 0
        self.fig = plt.figure(figsize=(17, 10))
        self.nav_buttons = []   # se reconstruyen en cada panel
        self.show_panel(0)
        plt.show()

    # ──────────────────────────────────────────────────────
    #  Dispatcher de paneles  (limpia y reconstruye todo)
    # ──────────────────────────────────────────────────────
    def show_panel(self, idx):
        self.panel = idx

        # Liberar mouse grab si algún slider lo tiene capturado
        try:
            self.fig.canvas.release_mouse(self.fig.canvas.mouse_grabber)
        except Exception:
            pass

        # Limpiar figura completa (elimina ejes Y desconecta eventos internos)
        self.fig.clear()
        self.nav_buttons = []

        # Título
        self.fig.suptitle(
            "3D Gaussian Splatting – Visualización Interactiva",
            fontsize=15, fontweight="bold", color=C_ACCENT, y=0.98)

        # Botones de navegación
        x_start, w, gap = 0.05, 0.20, 0.005
        for i, lbl in enumerate(self.NAV_LABELS):
            ax_btn = self.fig.add_axes([x_start + i*(w+gap), 0.92, w, 0.04])
            btn = Button(ax_btn, lbl,
                         color=C_ACCENT if i == idx else C_SLIDER,
                         hovercolor=C_ACCENT)
            btn.label.set_color(C_TEXT)
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold" if i == idx else "normal")
            btn.on_clicked(lambda _, i=i: self.show_panel(i))
            self.nav_buttons.append(btn)

        if   idx == 0: self._panel_gaussian3d()
        elif idx == 1: self._panel_covariance()
        elif idx == 2: self._panel_alpha()
        elif idx == 3: self._panel_sh()
        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════
    #  PANEL 1 – Gaussiana 3D
    # ══════════════════════════════════════════════════════
    def _panel_gaussian3d(self):
        # ── layout ──
        gs = gridspec.GridSpec(2, 3,
                               left=0.06, right=0.98,
                               top=0.88, bottom=0.32,
                               hspace=0.4, wspace=0.35,
                               figure=self.fig)
        self.ax3d    = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_proj = [self.fig.add_subplot(gs[r, c])
                        for r, c in [(0,1),(0,2),(1,1),(1,2)]]

        self._g3d_slider_axes = []
        self._g3d_sliders     = {}
        specs = [
            ("sx",  0.1, 2.0, self.DEF["sx"],  "Escala  sₓ"),
            ("sy",  0.1, 2.0, self.DEF["sy"],  "Escala  s_y"),
            ("sz",  0.1, 2.0, self.DEF["sz"],  "Escala  s_z"),
            ("rx", -90., 90., self.DEF["rx"],  "Rotación  rx°"),
            ("ry", -90., 90., self.DEF["ry"],  "Rotación  ry°"),
            ("rz", -90., 90., self.DEF["rz"],  "Rotación  rz°"),
            ("opacity", 0.01, 1.0, self.DEF["opacity"], "Opacidad  o"),
        ]
        y0, dy = 0.28, 0.036
        for i, (key, lo, hi, val, lbl) in enumerate(specs):
            ax_s = self.fig.add_axes([0.12, y0 - i*dy, 0.78, 0.025])
            sl = Slider(ax_s, lbl, lo, hi, valinit=val,
                        color=C_ACCENT, track_color=C_SLIDER)
            sl.label.set_color(C_TEXT); sl.label.set_fontsize(8)
            sl.valtext.set_color(C_GOLD)
            sl.on_changed(lambda _: self._update_g3d())
            self._g3d_slider_axes.append(ax_s)
            self._g3d_sliders[key] = sl

        self._update_g3d()

    def _update_g3d(self):
        s = self._g3d_sliders
        sx  = s["sx"].val;  sy  = s["sy"].val;  sz  = s["sz"].val
        rx  = s["rx"].val;  ry  = s["ry"].val;  rz  = s["rz"].val
        opa = s["opacity"].val

        sigma = build_covariance(sx, sy, sz, rx, ry, rz)
        Xe, Ye, Ze = ellipsoid_surface(sigma, n=40)

        # ── Vista 3D ──────────────────────────────────────
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor(C_BG)
        surf = ax.plot_surface(Xe, Ye, Ze, alpha=opa * 0.85,
                               cmap="plasma", linewidth=0, antialiased=True)
        ax.set_xlabel("X", labelpad=2); ax.set_ylabel("Y", labelpad=2)
        ax.set_zlabel("Z", labelpad=2)
        ax.set_title(
            f"Elipsoide 3D\nΣ = R·S·Sᵀ·Rᵀ",
            color=C_GREEN, fontsize=9, pad=4)
        lim = max(sx, sy, sz) * 1.2
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.grid(True); ax.tick_params(labelsize=6)

        # ── Proyecciones 2D ───────────────────────────────
        proj_data = [
            (Xe, Ye, "XY",  "cyan"),
            (Xe, Ze, "XZ",  C_GOLD),
            (Ye, Ze, "YZ",  C_GREEN),
        ]
        titles_axes = [
            ("Proyección XY",  "X", "Y"),
            ("Proyección XZ",  "X", "Z"),
            ("Proyección YZ",  "Y", "Z"),
        ]
        for ax2, (A, B, plane, col), (ttl, xl, yl) in zip(
                self.ax_proj[:3], proj_data, titles_axes):
            ax2.cla()
            ax2.set_facecolor(C_BG)
            ax2.contourf(A, B,
                         np.exp(-0.5*(A**2 + B**2) / (1e-6 + A.std()**2)),
                         levels=12, cmap="plasma", alpha=0.6)
            ax2.plot(A[0], B[0], color=col, alpha=0.3)
            ax2.set_title(ttl, color=col, fontsize=8)
            ax2.set_xlabel(xl, fontsize=7); ax2.set_ylabel(yl, fontsize=7)
            ax2.set_aspect("equal"); ax2.grid(True); ax2.tick_params(labelsize=6)

        # 4.º panel: covarianza como heatmap pequeño
        ax4 = self.ax_proj[3]
        ax4.cla(); ax4.set_facecolor(C_BG)
        im = ax4.imshow(sigma, cmap="RdBu_r",
                        vmin=-max(sx,sy,sz)**2, vmax=max(sx,sy,sz)**2)
        for ii in range(3):
            for jj in range(3):
                ax4.text(jj, ii, f"{sigma[ii,jj]:.2f}",
                         ha="center", va="center",
                         color="white", fontsize=7, fontweight="bold")
        ax4.set_title("Matriz Σ (covarianza)", color=C_ACCENT, fontsize=8)
        ax4.set_xticks([0,1,2]); ax4.set_yticks([0,1,2])
        ax4.set_xticklabels(["x","y","z"]); ax4.set_yticklabels(["x","y","z"])
        ax4.tick_params(labelsize=7)

        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════
    #  PANEL 2 – Factorización Σ = R S Sᵀ Rᵀ
    # ══════════════════════════════════════════════════════
    def _panel_covariance(self):
        gs = gridspec.GridSpec(1, 4,
                               left=0.05, right=0.98,
                               top=0.88, bottom=0.35,
                               hspace=0.1, wspace=0.3,
                               figure=self.fig)
        self._cov_axes = [self.fig.add_subplot(gs[0, i]) for i in range(4)]

        # sliders: sx, sy, sz, rx, ry, rz
        self._cov_sliders = {}
        specs = [
            ("sx", 0.1, 2.0, 1.0,  "sₓ"),
            ("sy", 0.1, 2.0, 0.5,  "s_y"),
            ("sz", 0.1, 2.0, 0.3,  "s_z"),
            ("rx",-90., 90., 20.,  "rx°"),
            ("ry",-90., 90., 30.,  "ry°"),
            ("rz",-90., 90.,  0.,  "rz°"),
        ]
        y0, dy = 0.31, 0.045
        for i, (key, lo, hi, val, lbl) in enumerate(specs):
            ax_s = self.fig.add_axes([0.10, y0 - i*dy, 0.80, 0.028])
            sl = Slider(ax_s, lbl, lo, hi, valinit=val,
                        color=C_GREEN, track_color=C_SLIDER)
            sl.label.set_color(C_TEXT); sl.label.set_fontsize(9)
            sl.valtext.set_color(C_GOLD)
            sl.on_changed(lambda _: self._update_cov())
            self._cov_sliders[key] = sl

        self._update_cov()

    def _update_cov(self):
        s = self._cov_sliders
        sx, sy, sz = s["sx"].val, s["sy"].val, s["sz"].val
        rx, ry, rz = s["rx"].val, s["ry"].val, s["rz"].val

        R = euler_to_R(rx, ry, rz)
        S = np.diag([sx, sy, sz])
        SS = S @ S.T
        sigma = R @ SS @ R.T
        vals, vecs = np.linalg.eigh(sigma)

        mats  = [R, SS, R.T, sigma]
        names = ["R  (rotación)", "S·Sᵀ  (escala²)", "Rᵀ", "Σ = R S Sᵀ Rᵀ"]
        cmaps = ["coolwarm", "plasma", "coolwarm", "RdBu_r"]
        maxv  = max(sx, sy, sz)**2

        for ax, M, name, cmap in zip(self._cov_axes, mats, names, cmaps):
            ax.cla(); ax.set_facecolor(C_BG)
            vm = max(abs(M).max(), 1e-4)
            ax.imshow(M, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto")
            for ii in range(3):
                for jj in range(3):
                    ax.text(jj, ii, f"{M[ii,jj]:.2f}",
                            ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold")
            ax.set_title(name, color=C_GOLD, fontsize=9, pad=6)
            ax.set_xticks([]); ax.set_yticks([])

        # anotación de eigenvalores
        ax_sig = self._cov_axes[3]
        eig_txt = "  eigenvalores:\n" + \
                  "\n".join(f"  λ{i+1} = {v:.3f}  (semieje ≈ {np.sqrt(max(v,0)):.2f})"
                            for i, v in enumerate(vals))
        ax_sig.text(3.3, 1, eig_txt,
                    transform=ax_sig.transData,
                    color=C_GREEN, fontsize=7.5, va="center",
                    bbox=dict(boxstyle="round", fc=C_BG, ec=C_GREEN, alpha=0.8))

        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════
    #  PANEL 3 – Alpha Blending
    # ══════════════════════════════════════════════════════
    N_GAUSS = 8

    def _panel_alpha(self):
        gs = gridspec.GridSpec(1, 2,
                               left=0.06, right=0.98,
                               top=0.88, bottom=0.38,
                               hspace=0.1, wspace=0.35,
                               figure=self.fig)
        self.ax_alpha_bars   = self.fig.add_subplot(gs[0, 0])
        self.ax_alpha_accum  = self.fig.add_subplot(gs[0, 1])

        # colores de las gaussianas
        self._ab_colors = plt.cm.tab10(np.linspace(0, 1, self.N_GAUSS))

        # sliders de opacidad por gaussiana
        self._ab_sliders = []
        y0, dy = 0.35, 0.038
        for i in range(self.N_GAUSS):
            col = self._ab_colors[i]
            ax_s = self.fig.add_axes([0.08, y0 - i*dy, 0.84, 0.025])
            sl = Slider(ax_s, f"α_{i+1}", 0.0, 1.0,
                        valinit=np.random.uniform(0.3, 0.85),
                        color=col[:3], track_color=C_SLIDER)
            sl.label.set_color(col[:3]); sl.label.set_fontsize(8)
            sl.valtext.set_color(C_GOLD); sl.valtext.set_fontsize(8)
            sl.on_changed(lambda _: self._update_alpha())
            self._ab_sliders.append(sl)

        self._update_alpha()

    def _update_alpha(self):
        alphas = np.array([sl.val for sl in self._ab_sliders])
        cols   = self._ab_colors

        # Acumulación front-to-back
        T = 1.0
        Ts     = []
        contribs = []
        C_final = np.zeros(3)
        for i, (a, c) in enumerate(zip(alphas, cols)):
            Ts.append(T)
            contrib = a * T
            contribs.append(contrib)
            C_final += contrib * c[:3]
            T *= (1 - a)
        T_final = T

        # ── Panel barras ──────────────────────────────────
        ax = self.ax_alpha_bars
        ax.cla(); ax.set_facecolor(C_BG)
        x = np.arange(self.N_GAUSS)
        w = 0.28
        ax.bar(x - w, alphas,      w, label="αᵢ (opacidad)",   color=cols, alpha=0.9)
        ax.bar(x,     Ts,          w, label="Tᵢ (transmit.)",  color=cols, alpha=0.5, edgecolor="white", linewidth=0.5)
        ax.bar(x + w, contribs,    w, label="αᵢ·Tᵢ (contrib)", color=cols, alpha=0.7, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([f"G{i+1}" for i in range(self.N_GAUSS)], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title("Contribuciones por Gaussiana\n(front-to-back)",
                     color=C_GOLD, fontsize=9)
        ax.set_ylabel("Valor", fontsize=8)
        ax.legend(fontsize=7, loc="upper right",
                  facecolor=C_BG, edgecolor=C_TEXT)
        ax.grid(True, axis="y")

        # ── Panel acumulación ─────────────────────────────
        ax2 = self.ax_alpha_accum
        ax2.cla(); ax2.set_facecolor(C_BG)

        # Acumulación incremental de color
        C_accum = np.zeros((self.N_GAUSS + 1, 3))
        T2 = 1.0
        for i, (a, c) in enumerate(zip(alphas, cols)):
            C_accum[i+1] = C_accum[i] + a * T2 * c[:3]
            T2 *= (1 - a)

        for i in range(self.N_GAUSS):
            ax2.barh(i, 1, left=0, color=np.clip(C_accum[i+1], 0, 1),
                     height=0.7, alpha=0.85)
            ax2.text(1.02, i, f"Tᵢ={Ts[i]:.2f}", va="center",
                     color=C_TEXT, fontsize=7)

        # Color final
        rect = plt.Rectangle((0, -1.3), 1, 0.9,
                              color=np.clip(C_final, 0, 1))
        ax2.add_patch(rect)
        ax2.text(0.5, -0.9, "Color final C(u)", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
        ax2.text(0.5, -1.5,
                 f"T_final = {T_final:.3f}   (luz que pasa)",
                 ha="center", color=C_GREEN, fontsize=8)

        ax2.set_xlim(0, 1.5); ax2.set_ylim(-2, self.N_GAUSS)
        ax2.set_yticks(range(self.N_GAUSS))
        ax2.set_yticklabels([f"G{i+1}" for i in range(self.N_GAUSS)], fontsize=8)
        ax2.set_title("Acumulación de Color\nC = Σ cᵢ αᵢ Tᵢ",
                      color=C_GOLD, fontsize=9)
        ax2.set_xlabel("Intensidad RGB", fontsize=8)
        ax2.grid(True, axis="x")
        ax2.invert_yaxis()

        self.fig.canvas.draw_idle()

    # ══════════════════════════════════════════════════════
    #  PANEL 4 – Spherical Harmonics
    # ══════════════════════════════════════════════════════
    def _panel_sh(self):
        # grid izquierda: esfera coloreada; derecha: coeficientes individuales
        gs = gridspec.GridSpec(3, 4,
                               left=0.05, right=0.98,
                               top=0.88, bottom=0.38,
                               hspace=0.5, wspace=0.3,
                               figure=self.fig)
        self.ax_sh_sphere = self.fig.add_subplot(gs[:, :2], projection="3d")
        self.ax_sh_bands  = [self.fig.add_subplot(gs[r, c+2])
                             for r in range(3) for c in range(2)]

        self._sh_sliders = []
        self._sh_T, self._sh_P = sphere_grid(n=60)

        y0, dy = 0.35, 0.038
        for i in range(9):
            row, col = divmod(i, 3)
            ax_s = self.fig.add_axes([0.05, y0 - i*dy, 0.88, 0.025])
            sl = Slider(ax_s, SH_NAMES[i], -1.5, 1.5,
                        valinit=np.random.uniform(-0.5, 0.5),
                        color=C_BLUE, track_color=C_SLIDER)
            sl.label.set_color(C_GOLD); sl.label.set_fontsize(8)
            sl.valtext.set_color(C_TEXT); sl.valtext.set_fontsize(8)
            sl.on_changed(lambda _: self._update_sh())
            self._sh_sliders.append(sl)

        # Botón reset
        ax_reset = self.fig.add_axes([0.45, 0.355, 0.10, 0.028])
        btn_r = Button(ax_reset, "Reset / Random",
                       color=C_SLIDER, hovercolor=C_ACCENT)
        btn_r.label.set_color(C_TEXT); btn_r.label.set_fontsize(8)
        def _randomize(_):
            for sl in self._sh_sliders:
                sl.set_val(np.random.uniform(-1, 1))
        btn_r.on_clicked(_randomize)

        self._update_sh()

    def _update_sh(self):
        coefs = [sl.val for sl in self._sh_sliders]
        T, P  = self._sh_T, self._sh_P

        # Coordenadas cartesianas de la esfera
        X = np.sin(T) * np.cos(P)
        Y = np.sin(T) * np.sin(P)
        Z = np.cos(T)

        # ── Valor SH compuesto ────────────────────────────
        val = np.zeros_like(T)
        for c, f in zip(coefs, SH_FUNCS):
            val += c * f(T, P)
        val_norm = (val - val.min()) / (val.ptp() + 1e-8)

        # ── Esfera principal ──────────────────────────────
        ax = self.ax_sh_sphere
        ax.cla(); ax.set_facecolor(C_BG)
        ax.plot_surface(X, Y, Z,
                        facecolors=plt.cm.plasma(val_norm),
                        alpha=0.9, linewidth=0, antialiased=True)
        ax.set_title(
            "Color SH  c(d) = Σ fˡₘ Yˡₘ(d)\n"
            "(gira la esfera para ver dependencia de vista)",
            color=C_GREEN, fontsize=9, pad=4)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.grid(True); ax.tick_params(labelsize=6)

        # ── Bandas individuales ───────────────────────────
        for i, (ax_b, c, f, name) in enumerate(
                zip(self.ax_sh_bands, coefs, SH_FUNCS, SH_NAMES)):
            ax_b.cla(); ax_b.set_facecolor(C_BG)
            v = c * f(T, P)
            v_n = (v - v.min()) / (v.ptp() + 1e-8)
            ax_b.imshow(v_n, cmap="RdBu_r", origin="upper", aspect="auto")
            ax_b.set_title(f"{name}  ×{c:.2f}", color=C_GOLD, fontsize=7)
            ax_b.set_xticks([]); ax_b.set_yticks([])

        self.fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("  3D Gaussian Splatting – Visualización Interactiva")
    print("=" * 60)
    print("  Paneles (clic en los botones superiores):")
    print("  ① Gaussiana 3D    – elipsoide con sliders")
    print("  ② Factorización Σ – R·S·Sᵀ·Rᵀ en tiempo real")
    print("  ③ Alpha Blending  – acumulación de transmitancia")
    print("  ④ Spherical Harm. – color dependiente de vista")
    print()
    print("  Cierra la ventana para salir.")
    print("=" * 60)
    Viz3DGS()
