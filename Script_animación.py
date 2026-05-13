import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import glob, os
from IPython.display import HTML

CARPETA  = "frames"
PML      = 20
INTERVAL = 80

MATERIALES = [
    {"label": "Vidrio",    "x0": 180, "y0": 280, "w": 80,  "h": 150, "color": "#4CAF50"},
    {"label": "Conductor", "x0": 140, "y0": 100, "w": 20,  "h": 300, "color": "#888888"},
    {"label": "Ferrita",   "x0": 360, "y0": 320, "w": 60,  "h": 100, "color": "#9C27B0"},
]

archivos = sorted(
    glob.glob(os.path.join(CARPETA, "hz_t*.dat")),
    key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or "0")
)
if not archivos:
    raise FileNotFoundError(f"No hay archivos hz_t*.dat en '{CARPETA}/'")

print(f"Cargando {len(archivos)} frames...", end=" ", flush=True)
frames = [np.loadtxt(f) for f in archivos]
print("listo.")

vmax = max(np.percentile(np.abs(d), 99.5) for d in frames)
vmax = vmax if vmax > 1e-10 else 1e-6

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(frames[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               origin="upper", interpolation="bilinear")

for m in MATERIALES:
    ax.add_patch(patches.Rectangle(
        (m["y0"], m["x0"]), m["w"], m["h"],
        lw=1.5, edgecolor=m["color"], facecolor=m["color"], alpha=0.2, label=m["label"]
    ))

ax.add_patch(patches.Rectangle(
    (PML, PML), frames[0].shape[1]-2*PML, frames[0].shape[0]-2*PML,
    lw=1, edgecolor="#FF9800", facecolor="none", ls="--", alpha=0.5, label="PML"
))

for xi, yi in [(frames[0].shape[1]//2, frames[0].shape[0]//2),
               (frames[0].shape[1]//4, frames[0].shape[0]//4)]:
    ax.plot(xi, yi, "w+", ms=8, mew=1.5, zorder=5)

fig.colorbar(im, ax=ax, fraction=0.046).set_label("Hz")
titulo = ax.set_title(os.path.basename(archivos[0]), fontsize=11)
ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
ax.set_xlabel("j"); ax.set_ylabel("i")
plt.tight_layout()

def update(fi):
    im.set_data(frames[fi])
    titulo.set_text(os.path.basename(archivos[fi]))
    return im, titulo

anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                interval=INTERVAL, blit=True)
plt.close()          # evita figura estática duplicada
HTML(anim.to_jshtml())
