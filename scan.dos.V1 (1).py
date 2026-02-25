import streamlit as st
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
# CONFIG & DESIGN
# ==============================
st.set_page_config(page_title="SpineScan Pro V3 (Raster)", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fc; }
.result-box { background-color:#fff; padding:14px; border-radius:10px; border:1px solid #e0e0e0; margin-bottom:10px; }
.value-text { font-size: 1.1rem; font-weight: bold; color: #2c3e50; }
.stButton>button { background-color: #2c3e50; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
.disclaimer { font-size: 0.82rem; color: #555; font-style: italic; margin-top: 10px; border-left: 3px solid #ccc; padding-left: 10px;}
.badge-ok {display:inline-block; padding:2px 8px; border-radius:999px; background:#e8f7ee; color:#156f3b; font-weight:700; font-size:0.85rem;}
.badge-no {display:inline-block; padding:2px 8px; border-radius:999px; background:#fdecec; color:#9b1c1c; font-weight:700; font-size:0.85rem;}
.small {color:#666;font-size:0.9rem;}
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
def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spinescan_v3.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    header_s = ParagraphStyle("Header", fontSize=16, textColor=colors.HexColor("#2c3e50"), alignment=1)

    story = []
    story.append(Paragraph("<b>BILAN DE SANTÉ RACHIDIENNE 3D — V3</b>", header_s))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    data = [
        ["Indicateur", "Valeur Mesurée"],
        ["Flèche Dorsale", f"{results['fd']:.2f} cm"],
        ["Flèche Lombaire", f"{results['fl']:.2f} cm ({results['fl_status']})"],
        ["Déviation Latérale Max", f"{results['dev_f']:.2f} cm"],
        ["Angle Lordose Lombaire (est.)", f"{results['lordosis_deg']:.1f}° ({results['lordosis_status']})"],
        ["Angle Cyphose Dorsale (est.)", f"{results['kyphosis_deg']:.1f}° ({results['kyphosis_status']})"],
        ["Jonction Thoraco-Lombaire (est.)", f"{results['y_junction']:.1f} cm" if results['y_junction'] is not None else "n/a"],
        ["Couverture / Fiabilité", f"{results['coverage_pct']:.0f}% / {results['reliability_pct']:.0f}%"],
        ["Confiance PSIS", f"{results['psis_conf']:.0f}%"],
    ]

    t = Table(data, colWidths=[7 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    img_t = Table([[PDFImage(img_f, width=6.2 * cm, height=9.0 * cm),
                    PDFImage(img_s, width=6.2 * cm, height=9.0 * cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# CLASSIFICATIONS
# ==============================
def classify_fl(fl_cm, lo=2.5, hi=4.5):
    if fl_cm < lo:
        return "Trop faible"
    if fl_cm > hi:
        return "Trop élevée"
    return "Normale"

def classify_angle(val_deg, lo, hi):
    if val_deg < lo:
        return "Trop faible"
    if val_deg > hi:
        return "Trop élevée"
    return "Normale"

# ==============================
# METRICS (sagittal)
# ==============================
def compute_sagittal_arrow_lombaire_v2(spine_cm):
    """
    Gardé comme ton code : fl = |z_min - z_max|.
    """
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]
    if len(z) == 0:
        return 0.0, 0.0, np.array([])
    idx_dorsal = int(np.argmax(z))
    z_dorsal = float(z[idx_dorsal])
    vertical_z = np.full_like(y, z_dorsal)
    idx_lombaire = int(np.argmin(z))
    z_lombaire = float(z[idx_lombaire])
    fd = 0.0
    fl = float(abs(z_lombaire - z_dorsal))
    return fd, fl, vertical_z

# ==============================
# LISSAGE
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
# ROTATION CORRECTION (XZ)
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
# V3 RASTER SURFACE
# ==============================
def build_depth_surface(points, dx=0.5, dy=0.5):
    """
    Surface Z(x,y) : max Z par cellule => "surface dos" robuste densité.
    """
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

    # max z per cell
    for k in range(ix.size):
        gx, gy = ix[k], iy[k]
        v = z[k]
        if np.isnan(grid[gy, gx]) or v > grid[gy, gx]:
            grid[gy, gx] = v

    # fill holes column-wise (simple but effective)
    for col in range(nx):
        col_vals = grid[:, col]
        m = ~np.isnan(col_vals)
        if np.count_nonzero(m) >= 4:
            grid[:, col] = np.interp(np.arange(ny), np.where(m)[0], col_vals[m])

    return grid, xmin, ymin, dx, dy

def extract_midline_symmetry_surface(grid, xmin, ymin, dx, dy, edge_q=10):
    """
    Midline = milieu entre bords gauche/droit sur chaque ligne Y de la surface.
    Retourne spine (x,y,z) + meta par point pour fiabilité.
    """
    ny, nx = grid.shape
    spine = []
    meta = []  # (valid_frac_row, width_cells)

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

    # sort by y
    o = np.argsort(spine[:, 1])
    return spine[o], meta[o]

def detect_psis(grid, xmin, ymin, dx, dy):
    """
    Détection PSIS (heuristique) : 2 dépressions (minima) dans bande bas du dos.
    Renvoie (psis_L, psis_R, conf 0..1)
    """
    ny, nx = grid.shape
    y_low = int(ny * 0.15)
    y_high = int(ny * 0.35)
    band = grid[y_low:y_high]

    if np.isnan(band).all():
        return None, None, 0.0

    # depth = median - z (dépression => depth haut)
    med = float(np.nanmedian(band))
    depth = med - band
    flat = depth.ravel()
    if np.all(np.isnan(flat)):
        return None, None, 0.0

    # top candidates
    idx_flat = np.argsort(flat)[-200:]
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

    # confidence: séparation + quantité + symétrie grossière
    sep = abs(rx - lx) / max(nx, 1)
    conf = 0.35 + 0.45 * np.clip(sep * 2.0, 0.0, 1.0) + 0.20 * np.clip(xs.size / 200.0, 0.0, 1.0)
    conf = float(np.clip(conf, 0.0, 1.0))
    return psis_L, psis_R, conf

def quality_from_surface(spine, meta, psis_conf=0.0, max_jump_cm=3.0):
    """
    Score fiabilité 0..1 par point:
    - valid_frac_row (plus il y a de surface, mieux c'est)
    - width plausible (évite les lignes où on n'a que l'épaule/bras)
    - continuité (jump)
    - psis_conf (boost global, utile pour suivi)
    """
    if spine.shape[0] == 0:
        return np.array([], dtype=float)

    valid_frac = meta[:, 0]
    width_cells = meta[:, 1]
    # width score (en cellules): trop petit => mauvais
    # on convertit en cm approximatif (cells*dx), mais dx connu plus haut ; ici on normalise juste.
    w = width_cells
    w_score = np.clip((w - np.percentile(w, 10)) / (np.percentile(w, 90) - np.percentile(w, 10) + 1e-6), 0, 1)

    # continuity score
    x = spine[:, 0]
    jumps = np.abs(np.diff(x))
    j_score = np.ones_like(x)
    if jumps.size:
        j_seg = np.clip(1.0 - (jumps / max_jump_cm), 0.0, 1.0)
        j_score[1:] = j_seg

    base = 0.45 * np.clip(valid_frac / (np.percentile(valid_frac, 85) + 1e-6), 0, 1) + 0.35 * w_score + 0.20 * j_score
    base = np.clip(base, 0.0, 1.0)

    # PSIS conf boost (global)
    base = np.clip(base * (0.85 + 0.15 * psis_conf), 0.0, 1.0)
    return base.astype(float)

# ==============================
# ANGLES V2 (concavité/convexité + tangentes)
# ==============================
def estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=21):
    if spine.shape[0] < 25:
        return 0.0, 0.0, None

    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)

    n = len(z)
    w = int(smooth_win)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    w = max(7, w)

    z_s = savgol_filter(z, w, 3)

    dz = np.gradient(z_s, y)
    d2z = np.gradient(dz, y)

    # auto-orient: bas concave (+) et haut convexe (-)
    y20 = np.percentile(y, 20)
    y80 = np.percentile(y, 80)
    low = d2z[y <= y20]
    high = d2z[y >= y80]
    if low.size < 3 or high.size < 3:
        return 0.0, 0.0, None

    sign_low = np.sign(np.median(low))
    sign_high = np.sign(np.median(high))
    if (sign_low < 0 and sign_high > 0):
        d2z = -d2z
        dz = -dz

    # jonction = proche inflexion au centre
    y_mid_lo = np.percentile(y, 35)
    y_mid_hi = np.percentile(y, 65)
    mid_mask = (y >= y_mid_lo) & (y <= y_mid_hi)
    idx_mid = np.where(mid_mask)[0]
    if idx_mid.size == 0:
        return 0.0, 0.0, None

    j = idx_mid[np.argmin(np.abs(d2z[idx_mid]))]
    y_j = float(y[j])

    y_bot = float(np.percentile(y, 8))
    y_top = float(np.percentile(y, 92))
    i_bot = int(np.argmin(np.abs(y - y_bot)))
    i_top = int(np.argmin(np.abs(y - y_top)))
    i_j = int(j)

    theta = np.degrees(np.arctan(dz))
    lordosis = float(abs(theta[i_j] - theta[i_bot]))
    kyphosis = float(abs(theta[i_top] - theta[i_j]))

    # fallback si jonction trop proche d'un bord
    if (y_j - y_bot) < 0.15 * (y_top - y_bot) or (y_top - y_j) < 0.15 * (y_top - y_bot):
        y_j_fb = float(np.percentile(y, 50))
        i_j_fb = int(np.argmin(np.abs(y - y_j_fb)))
        lordosis = float(abs(theta[i_j_fb] - theta[i_bot]))
        kyphosis = float(abs(theta[i_top] - theta[i_j_fb]))
        y_j = y_j_fb

    return lordosis, kyphosis, y_j

# ==============================
# PLOT COLORED CURVE
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

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")

    st.divider()
    st.subheader("🧩 V3 Raster (Revopoint)")
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
    st.subheader("📐 Angles")
    angle_smooth = st.slider("Lissage angles (fenêtre)", 7, 41, 21, step=2)

    st.divider()
    st.subheader("📏 Normes")
    show_norms = st.toggle("Afficher normes", True)
    fl_lo, fl_hi = 2.5, 4.5
    lord_lo, lord_hi = 40.0, 60.0
    kyph_lo, kyph_hi = 27.0, 47.0
    st.write(f"- Flèche lombaire: {fl_lo:.1f}–{fl_hi:.1f} cm")
    st.write(f"- Lordose: {lord_lo:.0f}–{lord_hi:.0f}°")
    st.write(f"- Cyphose: {kyph_lo:.0f}–{kyph_hi:.0f}°")

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro — V3 Rasterstéréographie (Revopoint)")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # ---- load + convert ----
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm

        # nettoyage Y léger (garder tout le dos)
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 1)) & (pts[:, 1] < np.percentile(pts[:, 1], 99))
        pts = pts[mask]

        # centrage X (affichage)
        pts[:, 0] -= np.median(pts[:, 0])

        # ---- rotation XZ (stabilité orientation) ----
        R = estimate_rotation_xz(pts)
        pts_r = apply_rotation_xz(pts, R)

        # ---- build surface + midline ----
        grid, xmin, ymin, dx_used, dy_used = build_depth_surface(pts_r, dx=float(dx), dy=float(dy))
        psis_L, psis_R, psis_conf = detect_psis(grid, xmin, ymin, dx_used, dy_used)

        spine_r, meta = extract_midline_symmetry_surface(grid, xmin, ymin, dx_used, dy_used, edge_q=int(edge_q))

        if spine_r.shape[0] < 20:
            st.error("Impossible d'extraire une courbe (surface insuffisante).")
            st.stop()

        # unrotate spine back to original coordinates
        spine = unrotate_spine_xz(spine_r, R)

        # ---- quality per point ----
        q = quality_from_surface(spine_r, meta, psis_conf=psis_conf, max_jump_cm=3.0)

        # ---- lissage ----
        if do_smooth:
            spine = smooth_spine(spine, window=smooth_window, strong=strong_smooth, median_k=median_k)

        # ---- metrics ----
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        fl_status = classify_fl(fl, fl_lo, fl_hi)

        dev_f = float(np.max(np.abs(spine[:, 0]))) if spine.size else 0.0

        # angles V2 (concavité/convexité)
        lordosis_deg, kyphosis_deg, y_junction = estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=int(angle_smooth))
        lordosis_status = classify_angle(lordosis_deg, lord_lo, lord_hi)
        kyphosis_status = classify_angle(kyphosis_deg, kyph_lo, kyph_hi)

        # coverage / reliability
        y_span_pts = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
        y_span_sp = float(np.max(spine[:, 1]) - np.min(spine[:, 1]))
        coverage_pct = 100.0 * (y_span_sp / y_span_pts) if y_span_pts > 1e-6 else 0.0
        reliability_pct = 100.0 * float(np.mean(q >= 0.6)) if q.size else 0.0
        psis_pct = 100.0 * float(psis_conf)

        # ---- figures ----
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "front_v3.png"), os.path.join(tmp, "sag_v3.png")

        # Frontale: X vs Y
        fig_f, ax_f = plt.subplots(figsize=(2.4, 4.2))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        plot_colored_curve(ax_f, spine[:, 0], spine[:, 1], q, lw=2.8)
        ax_f.set_title("Frontale (couleur = fiabilité)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=170)

        # Sagittale: Z vs Y
        fig_s, ax_s = plt.subplots(figsize=(2.4, 4.2))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        plot_colored_curve(ax_s, spine[:, 2], spine[:, 1], q, lw=2.8)
        if vertical_z.size:
            ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        if y_junction is not None:
            ax_s.axhline(y_junction, linestyle="--", linewidth=1, alpha=0.6)

        # PSIS markers (projected on sagittal not meaningful; show only as text; optional overlay in frontal)
        ax_s.set_title("Sagittale (couleur = fiabilité)", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=170)

        # ---- UI ----
        st.write("### 📈 Analyse Visuelle")
        _, c1, c2, _ = st.columns([1, 1, 1, 1])
        c1.pyplot(fig_f)
        c2.pyplot(fig_s)

        st.write("### 🎨 Légende fiabilité")
        st.caption("Rouge = faible | Jaune = moyen | Vert = fiable (score 0→1)")
        fig_leg, ax_leg = plt.subplots(figsize=(5.0, 0.35))
        ax_leg.set_axis_off()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax_leg.imshow(gradient, aspect="auto", cmap="RdYlGn")
        st.pyplot(fig_leg)

        if psis_L is not None:
            st.caption(f"PSIS détectées (confiance {psis_pct:.0f}%) — gauche: {psis_L[0]:.1f},{psis_L[1]:.1f} cm | droite: {psis_R[0]:.1f},{psis_R[1]:.1f} cm")
        else:
            st.caption(f"PSIS non détectées (confiance {psis_pct:.0f}%)")

        badge_fl = '<span class="badge-ok">Normale</span>' if fl_status == "Normale" else '<span class="badge-no">Hors norme</span>'
        badge_lord = '<span class="badge-ok">Normale</span>' if lordosis_status == "Normale" else '<span class="badge-no">Hors norme</span>'
        badge_kyph = '<span class="badge-ok">Normale</span>' if kyphosis_status == "Normale" else '<span class="badge-no">Hors norme</span>'

        st.write("### 📋 Synthèse des résultats")
        st.markdown(f"""
        <div class="result-box">
            <p><b>📏 Flèche Dorsale :</b> <span class="value-text">{fd:.2f} cm</span></p>

            <p><b>📏 Flèche Lombaire :</b> <span class="value-text">{fl:.2f} cm</span>
               {"&nbsp; " + badge_fl if show_norms else ""}
               {"<br><span class='small'>Référence: 2.5 à 4.5 cm</span>" if show_norms else ""}</p>

            <p><b>↔️ Déviation Latérale Max :</b> <span class="value-text">{dev_f:.2f} cm</span></p>

            <p><b>📐 Angle lordose lombaire (est.) :</b> <span class="value-text">{lordosis_deg:.1f}°</span>
               {"&nbsp; " + badge_lord if show_norms else ""}
               {"<br><span class='small'>Référence: 40° à 60°</span>" if show_norms else ""}</p>

            <p><b>📐 Angle cyphose dorsale (est.) :</b> <span class="value-text">{kyphosis_deg:.1f}°</span>
               {"&nbsp; " + badge_kyph if show_norms else ""}
               {"<br><span class='small'>Référence: 27° à 47°</span>" if show_norms else ""}</p>

            <p><b>🔁 Jonction thoraco-lombaire (est.) :</b> <span class="value-text">{(f"{y_junction:.1f} cm" if y_junction is not None else "n/a")}</span></p>

            <p><b>✅ Couverture hauteur :</b> <span class="value-text">{coverage_pct:.0f}%</span>
               &nbsp; <b>Fiabilité :</b> <span class="value-text">{reliability_pct:.0f}%</span>
               <br><span class="small">Fiabilité = % des points avec score ≥ 0.60 | Confiance PSIS = {psis_pct:.0f}%</span></p>

            <div class="disclaimer">
                V3 “Rasterstéréographie”: surface Z(x,y) = max Z par cellule (anti-biais densité) + midline par symétrie.
                Angles V2 = plan sagittal via concavité/convexité (z''(y)) + différence de tangentes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- PDF ----
        res = {
            "fd": float(fd),
            "fl": float(fl),
            "fl_status": fl_status,
            "dev_f": float(dev_f),
            "lordosis_deg": float(lordosis_deg),
            "kyphosis_deg": float(kyphosis_deg),
            "lordosis_status": lordosis_status,
            "kyphosis_status": kyphosis_status,
            "y_junction": None if y_junction is None else float(y_junction),
            "coverage_pct": float(coverage_pct),
            "reliability_pct": float(reliability_pct),
            "psis_conf": float(psis_pct),
        }
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_V3_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
