# ==============================
# SpineScan SUPER (Revopoint) — V3.4
# MODIFS DEMANDEES :
# ✅ Axe vertical configurable (AUTO / X / Y / Z) + remapping cohérent
# ✅ Référence de "verticalité" = tangente dorsale (0°) tracée en pointillés
# ✅ On NE calcule PLUS de "flèche dorsale" (fd supprimée)
# ✅ Lordose & Cyphose = angles relatifs (|theta_zone - theta_ref|)
#    où theta_ref = angle tangente dorsale (référence 0°)
# ✅ Flèche lombaire = max |z - z_ref(y)| dans la zone lombaire, avec z_ref(y) = tangente dorsale
# + UI + PDF + Fiabilité + Cobb proxy optionnel + Asymétrie optionnelle
# ==============================

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter
import tempfile, os
from plyfile import PlyData

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# ==============================
# PAGE + STYLE
# ==============================
st.set_page_config(page_title="SpineScan SUPER", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fc; }
.stButton>button { background-color: #2c3e50; color: white; width: 100%; border-radius: 10px; font-weight: 800; }
hr { margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ==============================
# IO
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

# ==============================
# PDF
# ==============================
def export_pdf_super(patient_info, results, img_front, img_sag, img_asym=None):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "rapport_spinescan_super.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)

    styles = getSampleStyleSheet()
    header_s = ParagraphStyle("Header", fontSize=16, textColor=colors.HexColor("#2c3e50"), alignment=1)
    sub_s = ParagraphStyle("Sub", fontSize=10, textColor=colors.HexColor("#2c3e50"))

    story = []
    story.append(Paragraph("<b>RAPPORT SPINESCAN SUPER (V3.4)</b>", header_s))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    data = [
        ["Indicateur", "Valeur"],
        ["Flèche lombaire (vs tangente dorsale)", f"{results['fl']:.2f} cm ({results['fl_status']})"],
        ["Déviation latérale max", f"{results['dev_f']:.2f} cm"],
        ["Lordose (angle vs ref 0°)", f"{results['lordosis_deg']:.1f}° ({results['lordosis_status']})"],
        ["Cyphose (angle vs ref 0°)", f"{results['kyphosis_deg']:.1f}° ({results['kyphosis_status']})"],
        ["Référence verticalité", f"Tangente dorsale = 0° (θref={results['theta_ref']:.1f}°)"],
        ["Axe vertical utilisé", results["up_axis_used"]],
        ["Couverture / Fiabilité", f"{results['coverage_pct']:.0f}% / {results['reliability_pct']:.0f}%"],
        ["Confiance PSIS", f"{results['psis_pct']:.0f}%"],
    ]
    if results.get("cobb_enabled", False):
        data.append(["Angle Cobb (proxy) (frontale)", f"{results['cobb_deg']:.1f}°"])

    t = Table(data, colWidths=[7.5 * cm, 7.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Graphiques", sub_s))
    story.append(Spacer(1, 0.2 * cm))

    row = [PDFImage(img_front, width=6.6*cm, height=9.2*cm),
           PDFImage(img_sag, width=6.6*cm, height=9.2*cm)]
    story.append(Table([row], colWidths=[7.5*cm, 7.5*cm]))
    story.append(Spacer(1, 0.25 * cm))

    if img_asym is not None and os.path.exists(img_asym):
        story.append(Paragraph("Carte d’asymétrie (option)", sub_s))
        story.append(Spacer(1, 0.2 * cm))
        story.append(PDFImage(img_asym, width=14.5*cm, height=6.0*cm))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(
        "Note : la référence de verticalité est définie par la <b>tangente dorsale</b> (zone haute) et fixée à <b>0°</b>. "
        "La flèche lombaire est mesurée comme le maximum de |z - z_ref(y)| sur la zone lombaire, "
        "avec z_ref(y) issue de la tangente dorsale. "
        "Les angles lordose et cyphose sont des angles <b>relatifs</b> (différence d’angles de tangentes par rapport à la référence). "
        "L’angle de Cobb est un <b>proxy de suivi</b> non radiographique.",
        styles["Normal"]
    ))

    doc.build(story)
    return path

# ==============================
# UTILITAIRES
# ==============================
def median_filter_1d(a, k):
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0:
        return a
    k = int(k)
    if k < 3:
        return a
    if k % 2 == 0:
        k += 1
    r = k // 2
    out = np.empty_like(a)
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        out[i] = np.median(a[lo:hi])
    return out

def smooth_spine(spine, window=91, strong=True, median_k=11):
    if spine.shape[0] < 7:
        return spine
    out = spine.copy()
    n = out.shape[0]

    if strong:
        mk = int(median_k)
        if mk % 2 == 0:
            mk += 1
        mk = min(mk, n if n % 2 == 1 else n - 1)
        mk = max(3, mk)
        out[:, 0] = median_filter_1d(out[:, 0], mk)
        out[:, 2] = median_filter_1d(out[:, 2], mk)

    w = int(window)
    if w % 2 == 0:
        w += 1
    max_w = n - 1
    if max_w % 2 == 0:
        max_w -= 1
    w = min(w, max_w)
    if w < 5:
        return out

    out[:, 0] = savgol_filter(out[:, 0], w, 3)
    out[:, 2] = savgol_filter(out[:, 2], w, 3)
    return out

# ==============================
# AXE VERTICAL — AUTO + REMAP
# ==============================
def infer_up_axis(pts):
    spans = [
        np.percentile(pts[:, 0], 99) - np.percentile(pts[:, 0], 1),
        np.percentile(pts[:, 1], 99) - np.percentile(pts[:, 1], 1),
        np.percentile(pts[:, 2], 99) - np.percentile(pts[:, 2], 1),
    ]
    return int(np.argmax(spans))

def remap_to_work_axes(pts, up_axis):
    """
    Repère de travail:
      X = gauche/droite
      Y = vertical (hauteur)
      Z = profondeur (sagittal)
    """
    if up_axis == 1:
        return pts.copy(), "Y"
    if up_axis == 2:
        return pts[:, [0, 2, 1]].copy(), "Z"
    return pts[:, [1, 0, 2]].copy(), "X"

# ==============================
# ROTATION CORRECTION (XZ) — repère de travail
# ==============================
def estimate_rotation_xz(pts):
    y = pts[:, 1]
    mid = (y > np.percentile(y, 30)) & (y < np.percentile(y, 70))
    pts_mid = pts[mid] if np.count_nonzero(mid) > 200 else pts
    XZ = pts_mid[:, [0, 2]]
    XZ = XZ - np.mean(XZ, axis=0)
    _, _, Vt = np.linalg.svd(XZ, full_matrices=False)
    angle = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
    c, s = np.cos(-angle), np.sin(-angle)
    return np.array([[c, -s], [s, c]], dtype=float)

def apply_rotation_xz(pts, R):
    XZ_rot = pts[:, [0, 2]] @ R.T
    return np.column_stack([XZ_rot[:, 0], pts[:, 1], XZ_rot[:, 1]])

def unrotate_spine_xz(spine, R):
    XZ_back = spine[:, [0, 2]] @ R
    out = spine.copy()
    out[:, 0] = XZ_back[:, 0]
    out[:, 2] = XZ_back[:, 1]
    return out

# ==============================
# V3 SURFACE (Rasterstéréographie)
# ==============================
def build_depth_surface(points, dx=0.5, dy=0.5):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])

    nx = int(np.ceil((xmax - xmin) / dx)) + 1
    ny = int(np.ceil((ymax - ymin) / dy)) + 1
    grid = np.full((ny, nx), np.nan, dtype=float)

    ix = ((x - xmin) / dx).astype(int)
    iy = ((y - ymin) / dy).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[valid], iy[valid], z[valid]

    for k in range(ix.size):
        gx, gy = ix[k], iy[k]
        v = z[k]
        if np.isnan(grid[gy, gx]) or v > grid[gy, gx]:
            grid[gy, gx] = v

    # comblement léger (colonne)
    for col in range(nx):
        col_vals = grid[:, col]
        m = ~np.isnan(col_vals)
        if np.count_nonzero(m) >= 4:
            grid[:, col] = np.interp(np.arange(ny), np.where(m)[0], col_vals[m])

    return grid, xmin, ymin, dx, dy

def extract_midline_symmetry_surface(grid, xmin, ymin, dx, dy, edge_q=10):
    ny, nx = grid.shape
    spine = []
    meta = []

    for j in range(ny):
        row = grid[j]
        valid = ~np.isnan(row)
        nvalid = int(np.count_nonzero(valid))
        if nvalid < 10:
            continue

        xs = np.where(valid)[0].astype(float)
        xL = float(np.percentile(xs, edge_q))
        xR = float(np.percentile(xs, 100 - edge_q))
        xM = int(np.clip(round(0.5 * (xL + xR)), 0, nx - 1))

        zM = float(row[xM]) if not np.isnan(row[xM]) else float(np.nanmedian(row[valid]))
        yM = float(ymin + j * dy)
        xM_cm = float(xmin + xM * dx)

        width_cells = float(xR - xL)
        valid_frac = float(nvalid / nx)

        spine.append([xM_cm, yM, zM])
        meta.append([valid_frac, width_cells])

    if len(spine) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=float)

    spine = np.array(spine, dtype=float)
    meta = np.array(meta, dtype=float)
    o = np.argsort(spine[:, 1])
    return spine[o], meta[o]

def detect_psis(grid, xmin, ymin, dx, dy):
    ny, nx = grid.shape
    y_low = int(ny * 0.15)
    y_high = int(ny * 0.35)
    band = grid[y_low:y_high]
    if np.isnan(band).all():
        return None, None, 0.0

    med = float(np.nanmedian(band))
    depth = med - band
    flat = depth.ravel()
    if np.all(np.isnan(flat)):
        return None, None, 0.0

    idx_flat = np.argsort(flat)[-220:]
    ys = (idx_flat // band.shape[1]) + y_low
    xs = (idx_flat % band.shape[1])

    if xs.size < 20:
        return None, None, 0.0

    x_med = float(np.median(xs))
    left_mask = xs < x_med
    right_mask = xs > x_med
    if np.count_nonzero(left_mask) < 8 or np.count_nonzero(right_mask) < 8:
        return None, None, 0.0

    lx = int(np.median(xs[left_mask]))
    rx = int(np.median(xs[right_mask]))
    ly = int(np.median(ys[left_mask]))
    ry = int(np.median(ys[right_mask]))

    psis_L = (float(xmin + lx * dx), float(ymin + ly * dy))
    psis_R = (float(xmin + rx * dx), float(ymin + ry * dy))

    sep = abs(rx - lx) / max(nx, 1)
    conf = 0.35 + 0.45 * np.clip(sep * 2.0, 0.0, 1.0) + 0.20 * np.clip(xs.size / 220.0, 0.0, 1.0)
    return psis_L, psis_R, float(np.clip(conf, 0.0, 1.0))

def quality_from_surface(spine_r, meta, psis_conf=0.0, max_jump_cm=3.0):
    if spine_r.shape[0] == 0:
        return np.array([], dtype=float)

    valid_frac = meta[:, 0]
    width_cells = meta[:, 1]

    w = width_cells
    w_p10 = float(np.percentile(w, 10))
    w_p90 = float(np.percentile(w, 90))
    w_score = np.clip((w - w_p10) / (w_p90 - w_p10 + 1e-6), 0, 1)

    x = spine_r[:, 0]
    jumps = np.abs(np.diff(x))
    j_score = np.ones_like(x)
    if jumps.size:
        j_seg = np.clip(1.0 - (jumps / max_jump_cm), 0.0, 1.0)
        j_score[1:] = j_seg

    v_p85 = float(np.percentile(valid_frac, 85))
    v_score = np.clip(valid_frac / (v_p85 + 1e-6), 0, 1)

    base = 0.50 * v_score + 0.30 * w_score + 0.20 * j_score
    base = np.clip(base, 0.0, 1.0)
    base = np.clip(base * (0.85 + 0.15 * psis_conf), 0.0, 1.0)
    return base.astype(float)

# ==============================
# RÉFÉRENCE VERTICALITÉ = TANGENTE DORSALE (0°)
# ==============================
def fit_tangent_z_of_y(spine, frac=(0.65, 0.92)):
    """
    Fit z = a*y + b sur la zone dorsale (haut du dos).
    Retour: a, b, theta_deg (= arctan(a)), y0, y1
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)

    if y.size < 20:
        a = 0.0
        b = float(np.median(z)) if z.size else 0.0
        return a, b, float(np.degrees(np.arctan(a))), float(y.min()) if y.size else 0.0, float(y.max()) if y.size else 1.0

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)
    y0 = y_min + frac[0] * span
    y1 = y_min + frac[1] * span
    m = (y >= y0) & (y <= y1)

    if np.count_nonzero(m) < 8:
        # fallback: top 25%
        y0 = y_min + 0.75 * span
        y1 = y_max
        m = (y >= y0) & (y <= y1)

    if np.count_nonzero(m) < 8:
        a = 0.0
        b = float(np.median(z))
    else:
        yy = y[m]
        zz = z[m]

        # robust: médiane par bins
        nb = 20
        edges = np.linspace(float(yy.min()), float(yy.max()), nb + 1)
        yc, zc = [], []
        for i in range(nb):
            mm = (yy >= edges[i]) & (yy < edges[i + 1])
            if np.count_nonzero(mm) < 3:
                continue
            yc.append(0.5 * (edges[i] + edges[i + 1]))
            zc.append(float(np.median(zz[mm])))

        if len(yc) >= 6:
            a, b = np.polyfit(np.array(yc), np.array(zc), 1)
        else:
            a, b = np.polyfit(yy, zz, 1)

    theta = float(np.degrees(np.arctan(float(a))))
    return float(a), float(b), theta, float(y0), float(y1)

def fit_tangent_in_zone(spine, frac=(0.12, 0.45)):
    """
    Fit z = a*y + b sur une zone (fractions en hauteur).
    Retour: a, b, theta_deg, y0, y1
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)

    if y.size < 20:
        a = 0.0
        b = float(np.median(z)) if z.size else 0.0
        return a, b, float(np.degrees(np.arctan(a))), float(y.min()) if y.size else 0.0, float(y.max()) if y.size else 1.0

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)
    y0 = y_min + frac[0] * span
    y1 = y_min + frac[1] * span
    m = (y >= y0) & (y <= y1)
    if np.count_nonzero(m) < 8:
        a = 0.0
        b = float(np.median(z))
        return a, b, float(np.degrees(np.arctan(a))), float(y0), float(y1)

    yy = y[m]
    zz = z[m]

    nb = 22
    edges = np.linspace(float(yy.min()), float(yy.max()), nb + 1)
    yc, zc = [], []
    for i in range(nb):
        mm = (yy >= edges[i]) & (yy < edges[i + 1])
        if np.count_nonzero(mm) < 3:
            continue
        yc.append(0.5 * (edges[i] + edges[i + 1]))
        zc.append(float(np.median(zz[mm])))

    if len(yc) >= 6:
        a, b = np.polyfit(np.array(yc), np.array(zc), 1)
    else:
        a, b = np.polyfit(yy, zz, 1)

    theta = float(np.degrees(np.arctan(float(a))))
    return float(a), float(b), theta, float(y0), float(y1)

def angle_relative(theta_zone, theta_ref):
    return float(abs(theta_zone - theta_ref))

def lumbar_arrow_vs_ref_tangent(spine, a_ref, b_ref, frac=(0.12, 0.45)):
    """
    Flèche lombaire = max |z - (a_ref*y + b_ref)| sur la zone lombaire.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)
    if y.size < 20:
        return 0.0

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)
    y0 = y_min + frac[0] * span
    y1 = y_min + frac[1] * span
    m = (y >= y0) & (y <= y1)
    if np.count_nonzero(m) < 8:
        return 0.0

    z_ref = a_ref * y + b_ref
    dev = z - z_ref
    return float(np.max(np.abs(dev[m])))

# ==============================
# COBB PROXY (front)
# ==============================
def estimate_cobb_proxy_front(spine, smooth_win=21):
    if spine.shape[0] < 30:
        return 0.0, None, None, None

    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    x = s[:, 0].astype(float)

    n = len(x)
    w = int(smooth_win)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    w = max(7, w)

    x_s = savgol_filter(x, w, 3)

    y10, y30, y70, y90 = np.percentile(y, [10, 30, 70, 90])
    m_bot = (y >= y10) & (y <= y30)
    m_top = (y >= y70) & (y <= y90)
    if np.count_nonzero(m_bot) < 6 or np.count_nonzero(m_top) < 6:
        return 0.0, None, None, None

    a_bot, b_bot = np.polyfit(y[m_bot], x_s[m_bot], 1)
    a_top, b_top = np.polyfit(y[m_top], x_s[m_top], 1)

    denom = 1.0 + a_bot * a_top
    ang = np.pi / 2 if abs(denom) < 1e-9 else np.arctan(abs((a_top - a_bot) / denom))
    return float(np.degrees(ang)), (float(a_bot), float(b_bot)), (float(a_top), float(b_top)), (float(y10), float(y30), float(y70), float(y90))

# ==============================
# PLOTS
# ==============================
def plot_colored_curve(ax, x, y, q, lw=2.8):
    if len(x) < 2:
        return
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    q_seg = 0.5 * (q[:-1] + q[1:])
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(q_seg)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale_view()

def save_fig(fig, name):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, name)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    return path

def make_asymmetry_heatmap(grid):
    ny, nx = grid.shape
    asym = np.full_like(grid, np.nan, dtype=float)

    for j in range(ny):
        row = grid[j]
        valid = ~np.isnan(row)
        xs = np.where(valid)[0]
        if xs.size < 10:
            continue
        mid = int(np.median(xs))
        for i in xs:
            sym = 2 * mid - i
            if 0 <= sym < nx and not np.isnan(row[sym]):
                asym[j, i] = row[i] - row[sym]

    vals = asym[~np.isnan(asym)]
    if vals.size < 50:
        return None
    lim = float(np.percentile(np.abs(vals), 95))
    lim = max(lim, 0.3)

    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    im = ax.imshow(asym, aspect="auto", origin="lower", vmin=-lim, vmax=lim)
    ax.set_title("Carte d’asymétrie gauche/droite (surface)", fontsize=10)
    ax.set_xlabel("X (cellules)")
    ax.set_ylabel("Y (cellules)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return fig

# ==============================
# MESURES / UI
# ==============================
def classify_range(val, lo, hi):
    if val < lo:
        return "Trop faible"
    if val > hi:
        return "Trop élevée"
    return "Normale"

# ==============================
# UI — SIDEBAR
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")

    st.divider()
    st.subheader("🧭 Axes (Revopoint)")
    up_choice = st.selectbox("Axe vertical (hauteur)", ["AUTO", "X", "Y", "Z"], index=0)
    st.caption("👉 Si les graphes sont couchés : choisis Z (très fréquent).")

    st.divider()
    st.subheader("📏 Zones tangentes (fractions hauteur)")
    st.caption("Référence verticalité (dorsale): 0° = tangente zone haute")
    ref_lo = st.slider("Réf dorsale — bas (%)", 40, 80, 65, step=1) / 100.0
    ref_hi = st.slider("Réf dorsale — haut (%)", 70, 98, 92, step=1) / 100.0

    st.caption("Zone Lordose (lombaire)")
    lo_lo = st.slider("Lordose — bas (%)", 0, 40, 12, step=1) / 100.0
    lo_hi = st.slider("Lordose — haut (%)", 20, 70, 45, step=1) / 100.0

    st.caption("Zone Cyphose (dorsale)")
    ky_lo = st.slider("Cyphose — bas (%)", 30, 80, 50, step=1) / 100.0
    ky_hi = st.slider("Cyphose — haut (%)", 50, 98, 78, step=1) / 100.0

    st.divider()
    st.subheader("🧩 Raster (surface)")
    dx = st.slider("Résolution X (cm)", 0.2, 1.2, 0.5, step=0.1)
    dy = st.slider("Résolution Y (cm)", 0.2, 1.2, 0.5, step=0.1)
    edge_q = st.slider("Bords (quantile X)", 5, 20, 10, step=1)

    st.divider()
    st.subheader("🧽 Lissage courbe")
    do_smooth = st.toggle("Activer", True)
    strong_smooth = st.toggle("Lissage fort (anti-pics)", True)
    smooth_window = st.slider("Fenêtre lissage", 5, 151, 91, step=2)
    median_k = st.slider("Anti-pics (médian)", 3, 31, 11, step=2)

    st.divider()
    st.subheader("📐 Cobb (proxy) — optionnel")
    cobb_enabled = st.toggle("Afficher angle de Cobb (proxy)", False)
    cobb_smooth = st.slider("Lissage Cobb (fenêtre)", 7, 41, 21, step=2)

    st.divider()
    st.subheader("🗺️ Asymétrie — optionnel")
    show_asym = st.toggle("Afficher heatmap asymétrie", False)

    st.divider()
    st.subheader("📏 Normes")
    show_norms = st.toggle("Afficher normes", True)
    fl_lo, fl_hi = 2.5, 4.5
    lord_lo, lord_hi = 40.0, 60.0
    kyph_lo, kyph_hi = 27.0, 47.0
    st.caption(f"Flèche lombaire: {fl_lo:.1f}–{fl_hi:.1f} cm")
    st.caption(f"Lordose: {lord_lo:.0f}–{lord_hi:.0f}°")
    st.caption(f"Cyphose: {kyph_lo:.0f}–{kyph_hi:.0f}°")

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

# ==============================
# MAIN
# ==============================
st.title("🦴 SpineScan SUPER — V3.4 (Réf verticalité = tangente dorsale 0°)")

if not ply_file:
    st.info("Veuillez importer un fichier .PLY (Revopoint) pour lancer l’analyse.")
    st.stop()

if st.button("⚙️ LANCER L'ANALYSE"):
    # ---- Load + convert to cm ----
    pts0 = load_ply_numpy(ply_file) * 0.1  # mm -> cm

    # Nettoyage léger (retirer extrêmes sur l'axe Y source, avant remap: on prend l'axe 1 par défaut)
    m = (pts0[:, 1] > np.percentile(pts0[:, 1], 1)) & (pts0[:, 1] < np.percentile(pts0[:, 1], 99))
    pts0 = pts0[m]

    # Choix axe vertical
    if up_choice == "AUTO":
        up_axis = infer_up_axis(pts0)
    else:
        up_axis = {"X": 0, "Y": 1, "Z": 2}[up_choice]

    pts, up_axis_label = remap_to_work_axes(pts0, up_axis=up_axis)
    up_axis_used_label = ("AUTO→" if up_choice == "AUTO" else "") + up_axis_label

    # Centrage X (affichage / extraction)
    pts[:, 0] -= np.median(pts[:, 0])

    # Rotation XZ pour stabiliser le plan/symétrie
    R = estimate_rotation_xz(pts)
    pts_r = apply_rotation_xz(pts, R)

    # ---- Raster surface + midline ----
    grid, xmin, ymin, dx_used, dy_used = build_depth_surface(pts_r, dx=float(dx), dy=float(dy))
    psis_L, psis_R, psis_conf = detect_psis(grid, xmin, ymin, dx_used, dy_used)
    spine_r, meta = extract_midline_symmetry_surface(grid, xmin, ymin, dx_used, dy_used, edge_q=int(edge_q))

    if spine_r.shape[0] < 25:
        st.error("Surface insuffisante pour extraire une ligne médiane stable.")
        st.stop()

    # Fiabilité
    q = quality_from_surface(spine_r, meta, psis_conf=psis_conf, max_jump_cm=3.0)

    # Retour repère travail (dé-rotation)
    spine = unrotate_spine_xz(spine_r, R)

    # Lissage final
    if do_smooth:
        spine = smooth_spine(spine, window=smooth_window, strong=strong_smooth, median_k=median_k)

    # --- Référence verticalité = tangente dorsale ---
    ref_frac = (float(ref_lo), float(ref_hi))
    a_ref, b_ref, theta_ref, y_ref0, y_ref1 = fit_tangent_z_of_y(spine, frac=ref_frac)

    # --- Tangentes zones lordose / cyphose ---
    a_lo, b_lo, theta_lo, y_lo0, y_lo1 = fit_tangent_in_zone(spine, frac=(float(lo_lo), float(lo_hi)))
    a_ky, b_ky, theta_ky, y_ky0, y_ky1 = fit_tangent_in_zone(spine, frac=(float(ky_lo), float(ky_hi)))

    lord_deg = angle_relative(theta_lo, theta_ref)
    kyph_deg = angle_relative(theta_ky, theta_ref)

    lord_status = classify_range(lord_deg, lord_lo, lord_hi)
    kyph_status = classify_range(kyph_deg, kyph_lo, kyph_hi)

    # --- Flèche lombaire vs référence dorsale ---
    fl = lumbar_arrow_vs_ref_tangent(spine, a_ref, b_ref, frac=(float(lo_lo), float(lo_hi)))
    fl_status = classify_range(fl, fl_lo, fl_hi)

    # Déviation latérale max
    dev_f = float(np.max(np.abs(spine[:, 0]))) if spine.size else 0.0

    # Cobb proxy
    cobb_deg, fit_bot, fit_top, y_ranges = (None, None, None, None)
    if cobb_enabled:
        cobb_deg, fit_bot, fit_top, y_ranges = estimate_cobb_proxy_front(spine, smooth_win=int(cobb_smooth))

    # Couverture / Fiabilité
    y_span_pts = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
    y_span_sp = float(np.max(spine[:, 1]) - np.min(spine[:, 1]))
    coverage_pct = 100.0 * (y_span_sp / y_span_pts) if y_span_pts > 1e-6 else 0.0
    coverage_pct = float(np.clip(coverage_pct, 0.0, 100.0))
    reliability_pct = 100.0 * float(np.mean(q >= 0.65)) if q.size else 0.0
    psis_pct = 100.0 * float(psis_conf)

    # =========================
    # GRAPHIQUES
    # =========================
    st.write("### 📈 Analyse visuelle")
    c1, c2 = st.columns(2)

    # Frontale X vs Y + Cobb
    fig_f, ax_f = plt.subplots(figsize=(3.0, 5.2))
    ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.07, color="gray")
    plot_colored_curve(ax_f, spine[:, 0], spine[:, 1], q, lw=3.0)

    if cobb_enabled and fit_bot and fit_top and y_ranges:
        a_bot, b_bot = fit_bot
        a_top, b_top = fit_top
        y10, y30, y70, y90 = y_ranges
        yy_bot = np.array([y10, y30])
        yy_top = np.array([y70, y90])
        xx_bot = a_bot * yy_bot + b_bot
        xx_top = a_top * yy_top + b_top
        ax_f.plot(xx_bot, yy_bot, linewidth=2.2)
        ax_f.plot(xx_top, yy_top, linewidth=2.2)
        ax_f.text(0.02, 0.98, f"Cobb proxy: {cobb_deg:.1f}°", transform=ax_f.transAxes,
                  va="top", ha="left", fontsize=9)

    ax_f.set_title("Frontale (couleur = fiabilité)", fontsize=10)
    ax_f.axis("off")
    img_front_path = save_fig(fig_f, "front_super_v34.png")
    c1.pyplot(fig_f)

    # Sagittale Z vs Y + tangentes (réf dorsale en pointillés)
    spine_s = spine[np.argsort(spine[:, 1])]
    y_sorted = spine_s[:, 1]
    z_sorted = spine_s[:, 2]

    fig_s, ax_s = plt.subplots(figsize=(3.0, 5.2))
    ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.07, color="gray")
    plot_colored_curve(ax_s, z_sorted, y_sorted, q, lw=3.0)

    # Tangente référence (pointillés) — 0°
    yy = np.array([y_ref0, y_ref1])
    zz = a_ref * yy + b_ref
    ax_s.plot(zz, yy, "k--", linewidth=1.6, alpha=0.85)

    # Tangente lordose
    yy_lo = np.array([y_lo0, y_lo1])
    zz_lo = a_lo * yy_lo + b_lo
    ax_s.plot(zz_lo, yy_lo, linewidth=1.8, alpha=0.9)

    # Tangente cyphose
    yy_ky = np.array([y_ky0, y_ky1])
    zz_ky = a_ky * yy_ky + b_ky
    ax_s.plot(zz_ky, yy_ky, linewidth=1.8, alpha=0.9)

    ax_s.text(0.02, 0.98, f"Réf dorsale: θref={theta_ref:.1f}° (fixée à 0°)", transform=ax_s.transAxes,
              va="top", ha="left", fontsize=9)

    ax_s.set_title("Sagittale (réf = tangente dorsale pointillée)", fontsize=10)
    ax_s.axis("off")
    img_sag_path = save_fig(fig_s, "sag_super_v34.png")
    c2.pyplot(fig_s)

    # Légende fiabilité
    st.caption("Fiabilité: Rouge = faible, Jaune = moyen, Vert = fiable (score 0→1).")
    fig_leg, ax_leg = plt.subplots(figsize=(5.8, 0.35))
    ax_leg.set_axis_off()
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(gradient, aspect="auto", cmap="RdYlGn")
    st.pyplot(fig_leg)

    if psis_L is not None:
        st.caption(f"PSIS détectées (confiance {psis_pct:.0f}%) — gauche: {psis_L[0]:.1f},{psis_L[1]:.1f} cm | droite: {psis_R[0]:.1f},{psis_R[1]:.1f} cm")
    else:
        st.caption(f"PSIS non détectées (confiance {psis_pct:.0f}%)")

    # Heatmap asymétrie optionnelle
    img_asym_path = None
    if show_asym:
        fig_a = make_asymmetry_heatmap(grid)
        if fig_a is not None:
            st.write("### 🗺️ Asymétrie gauche/droite (option)")
            st.pyplot(fig_a)
            img_asym_path = save_fig(fig_a, "asym_super_v34.png")
        else:
            st.info("Asymétrie: données insuffisantes pour une heatmap stable.")

    # =========================
    # SYNTHESE (HTML)
    # =========================
    st.write("### 🧾 Synthèse des résultats")

    def badge(ok):
        if ok:
            return '<span style="margin-left:8px; padding:2px 8px; border-radius:999px; background:#e8f7ee; color:#156f3b; font-weight:800; font-size:0.85rem;">Normale</span>'
        return '<span style="margin-left:8px; padding:2px 8px; border-radius:999px; background:#fdecec; color:#9b1c1c; font-weight:800; font-size:0.85rem;">Hors norme</span>'

    fl_ok = (fl_status == "Normale")
    lord_ok = (lord_status == "Normale")
    kyph_ok = (kyph_status == "Normale")

    cobb_block = ""
    if cobb_enabled and cobb_deg is not None:
        cobb_block = f"""
        <p><b>📐 Angle de Cobb (proxy) :</b> <span style="font-weight:900;">{cobb_deg:.1f}°</span>
        <br><span style="color:#666; font-size:0.9rem;">Proxy de suivi (frontale), pas un Cobb radiographique.</span></p>
        """

    norms_fl = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {fl_lo:.1f} à {fl_hi:.1f} cm</span>" if show_norms else ""
    norms_lord = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {lord_lo:.0f}° à {lord_hi:.0f}°</span>" if show_norms else ""
    norms_kyph = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {kyph_lo:.0f}° à {kyph_hi:.0f}°</span>" if show_norms else ""

    html_card = f"""
    <div style="
        background:#fff; padding:16px; border-radius:12px; border:1px solid #e0e0e0;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        color:#2c3e50;
    ">
      <div style="font-size:1.05rem; line-height:1.55;">
        <p><b>📏 Flèche lombaire (vs tangente dorsale) :</b> <span style="font-weight:900;">{fl:.2f} cm</span>
          {badge(fl_ok) if show_norms else ""}{norms_fl}</p>

        <p><b>↔️ Déviation latérale max :</b> <span style="font-weight:900;">{dev_f:.2f} cm</span></p>

        <p><b>📐 Lordose (angle vs ref 0°) :</b> <span style="font-weight:900;">{lord_deg:.1f}°</span>
          {badge(lord_ok) if show_norms else ""}{norms_lord}</p>

        <p><b>📐 Cyphose (angle vs ref 0°) :</b> <span style="font-weight:900;">{kyph_deg:.1f}°</span>
          {badge(kyph_ok) if show_norms else ""}{norms_kyph}</p>

        <p><b>🧭 Référence verticalité :</b> <span style="font-weight:900;">Tangente dorsale = 0°</span>
          <br><span style="color:#666; font-size:0.9rem;">θref mesuré = {theta_ref:.1f}° (mais affiché comme 0° pour la référence)</span></p>

        {cobb_block}

        <p><b>✅ Couverture :</b> <span style="font-weight:900;">{coverage_pct:.0f}%</span>
           &nbsp; <b>Fiabilité :</b> <span style="font-weight:900;">{reliability_pct:.0f}%</span>
           <br><span style="color:#666; font-size:0.9rem;">Fiabilité = % des points avec score ≥ 0.65 | Confiance PSIS = {psis_pct:.0f}%</span></p>

        <p><b>🧭 Axe vertical :</b> <span style="font-weight:900;">{up_axis_used_label}</span></p>
      </div>

      <div style="
          margin-top:10px; font-size:0.82rem; color:#555; font-style:italic;
          border-left:3px solid #ccc; padding-left:10px;
      ">
        Référence sagittale : tangente dorsale (pointillés) fixée à 0°.<br/>
        Flèche lombaire : max |z - z_ref(y)| sur zone lombaire.<br/>
        Angles : |θ(zone) - θref|.
      </div>
    </div>
    """
    components.html(html_card, height=560 if (cobb_enabled and cobb_deg is not None) else 520, scrolling=False)

    # =========================
    # PDF
    # =========================
    results = {
        "fl": float(fl),
        "fl_status": fl_status,
        "dev_f": float(dev_f),
        "lordosis_deg": float(lord_deg),
        "lordosis_status": lord_status,
        "kyphosis_deg": float(kyph_deg),
        "kyphosis_status": kyph_status,
        "theta_ref": float(theta_ref),
        "coverage_pct": float(coverage_pct),
        "reliability_pct": float(reliability_pct),
        "psis_pct": float(psis_pct),
        "cobb_enabled": bool(cobb_enabled),
        "cobb_deg": float(cobb_deg) if (cobb_enabled and cobb_deg is not None) else 0.0,
        "up_axis_used": str(up_axis_used_label),
    }

    pdf_path = export_pdf_super({"nom": nom, "prenom": prenom}, results, img_front_path, img_sag_path, img_asym_path)

    st.divider()
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Télécharger le rapport PDF", f, f"Rapport_SpineScan_SUPER_{nom}.pdf")
