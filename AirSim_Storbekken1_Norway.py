#!/usr/bin/env python3
"""
Simulate magnetic anomaly decay with two customizable cross-profiles,
each with its own angle, offset, and corridor width.

"""

import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# --- USER SETTINGS ---
src_path            = 'Mag.tif'
sensor_heights      = [0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5, 6, 10, 15]
anomaly_depth       = 0
field_window        = 100
noise_density       = 0.004
sample_rate         = 10
noise_std           = noise_density * np.sqrt(sample_rate/2)

# Profile 1 parameters
profile1_angle_deg      = 50.0    # ° clockwise from East
profile1_offset_m       = -3.5    # m lateral shift
corridor1_half_width    = 2.0     # m

# Profile 2 parameters
profile2_angle_deg      = 140.0   # ° clockwise from East
profile2_offset_m       = -20     # m lateral shift
corridor2_half_width    = 2.0     # m

# --- Effective power-law model ---
def eff_model(h, A0, n):
    return A0 * (h/h0)**(-n)

# --- 1) Load & mask raster ---
with rasterio.open(src_path) as src:
    arr, trans, nodv = src.read(1).astype(float), src.transform, src.nodata
    W, H = src.width, src.height

data = np.ma.masked_equal(arr, nodv) if nodv is not None else np.ma.masked_invalid(arr)
ulx, uly = trans * (0, 0)
lrx, lry = trans * (W, H)
dx, inv_tr = trans.a, ~trans

# --- Helper: build a profile line ---
def build_profile(angle_deg, offset_m):
    θ = math.radians(angle_deg)
    ux, uy = math.cos(θ), -math.sin(θ)
    px, py = uy, -ux
    L = math.hypot(W*dx, H*dx)
    N = int(L/dx)*2
    ts = np.linspace(-L, L, N)
    left = np.abs(data)[:, :W//2]
    r0, c0 = np.unravel_index(np.argmax(left), left.shape)
    cx, cy = trans * (c0, r0)
    xs = cx + offset_m*px + ts*ux
    ys = cy + offset_m*py + ts*uy
    cols, rows = inv_tr * (xs, ys)
    ri = np.round(rows).astype(int)
    ci = np.round(cols).astype(int)
    mask = (ci>=0)&(ci<W)&(ri>=0)&(ri<H)
    return xs[mask], ys[mask], ri[mask], ci[mask], ux, uy, px, py

# Build Profile 1 & 2
xs1, ys1, r1, c1, u1x, u1y, p1x, p1y = build_profile(profile1_angle_deg, profile1_offset_m)
xs2, ys2, r2, c2, u2x, u2y, p2x, p2y = build_profile(profile2_angle_deg, profile2_offset_m)

# --- Corridor builder ---
def make_corridor(xs, ys, px, py, half_width):
    n_off = int(np.ceil(half_width / dx))
    offs = np.linspace(-half_width, half_width, 2*n_off+1)
    rows, cols = [], []
    for x0, y0 in zip(xs, ys):
        rl, cl = [], []
        for d in offs:
            xo, yo = x0 + d*px, y0 + d*py
            co, ro = inv_tr * (xo, yo)
            ri, ci = int(round(ro)), int(round(co))
            if 0 <= ri < H and 0 <= ci < W:
                rl.append(ri); cl.append(ci)
        rows.append(rl); cols.append(cl)
    # discrete ticks
    M = 20
    idxs = np.round(np.linspace(0, len(xs)-1, M)).astype(int)
    xu = xs[idxs] + half_width*px; yu = ys[idxs] + half_width*py
    xl = xs[idxs] - half_width*px; yl = ys[idxs] - half_width*py
    return rows, cols, xu, yu, xl, yl

c1r, c1c, x1u, y1u, x1l, y1l = make_corridor(xs1, ys1, p1x, p1y, corridor1_half_width)
c2r, c2c, x2u, y2u, x2l, y2l = make_corridor(xs2, ys2, p2x, p2y, corridor2_half_width)

# --- Upward continuation via FFT ---
def upward_continue(grid, dx, h):
    ny, nx = grid.shape
    kx = np.fft.fftfreq(nx, dx)*2*np.pi
    ky = np.fft.fftfreq(ny, dx)*2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    H = np.exp(-np.sqrt(KX**2 + KY**2)*h)
    return np.real(ifft2(fft2(grid.filled(0))*H))

# --- Compute maps & sample corridors ---
maps = []
mins1, maxs1 = [], []
mins2, maxs2 = [], []

for h in sensor_heights:
    cont = upward_continue(data, dx, anomaly_depth + h)
    cont = np.ma.array(cont, mask=data.mask)
    cont += gaussian_filter(np.random.normal(0, noise_std, cont.shape), sigma=5)
    maps.append(cont)

    v1 = []
    for rl, cl in zip(c1r, c1c):
        v1.extend(cont[rl, cl].compressed())
    mins1.append(np.nanmin(v1)); maxs1.append(np.nanmax(v1))

    v2 = []
    for rl, cl in zip(c2r, c2c):
        v2.extend(cont[rl, cl].compressed())
    mins2.append(np.nanmin(v2)); maxs2.append(np.nanmax(v2))

# --- Fit effective power-laws ---
h0 = sensor_heights[0]
h_arr = np.array(sensor_heights)
min1_a, max1_a = np.array(mins1), np.array(maxs1)
min2_a, max2_a = np.array(mins2), np.array(maxs2)

p1min, _ = curve_fit(eff_model, h_arr, min1_a, p0=[min1_a[0],0.7])
p1max, _ = curve_fit(eff_model, h_arr, max1_a, p0=[max1_a[0],0.7])
p2min, _ = curve_fit(eff_model, h_arr, min2_a, p0=[min2_a[0],0.7])
p2max, _ = curve_fit(eff_model, h_arr, max2_a, p0=[max2_a[0],0.7])

A1m, n1m = p1min;   A1M, n1M = p1max
A2m, n2m = p2min;   A2M, n2M = p2max

print(f"\nProfile1 fit (min): A0={A1m:.3f}, n_eff={n1m:.3f}")
print(f"Profile1 fit (max): A0={A1M:.3f}, n_eff={n1M:.3f}")
print(f"Profile2 fit (min): A0={A2m:.3f}, n_eff={n2m:.3f}")
print(f"Profile2 fit (max): A0={A2M:.3f}, n_eff={n2M:.3f}\n")

# --- Print tables ---
def print_table(label, mn, mx):
    print(label)
    print("Height |   Min (nT) | %Min |   Max (nT) | %Max")
    print("------------------------------------------------")
    for h, mi, ma in zip(sensor_heights, mn, mx):
        pmi = mi/mn[0]*100; pma = ma/mx[0]*100
        print(f"{h:4.1f}   | {mi:8.3f} | {pmi:6.1f}% | {ma:8.3f} | {pma:6.1f}%")
    print()

print_table("Profile1:", min1_a, max1_a)
print_table("Profile2:", min2_a, max2_a)

r1M_obs = max1_a/max1_a[0]; r1M_fit = eff_model(h_arr, A1M, n1M)/A1M
r2M_obs = max2_a/max2_a[0]; r2M_fit = eff_model(h_arr, A2M, n2M)/A2M


# --- 2) Plot maps with profiles ---
cols, rows = 4, math.ceil(len(sensor_heights) / 4)

# create figure with constrained layout
fig, axs = plt.subplots(
    rows, cols,
    figsize=(cols * 3, rows * 3),
    constrained_layout=True,
    squeeze=False
)

cmap = plt.cm.gray_r
cmap.set_bad('white')

for idx, (h, cont) in enumerate(zip(sensor_heights, maps)):
    r, c = divmod(idx, cols)
    ax = axs[r][c]

    # show the map
    im = ax.imshow(
        cont,
        origin='upper',
        cmap=cmap,
        vmin=-field_window,
        vmax=field_window,
        extent=[ulx, lrx, lry, uly]
    )

    # draw both profiles
    ax.plot(xs1, ys1, 'r--', lw=1)
    ax.plot(xs2, ys2, 'g--', lw=1)

    # corridor ticks for profile 1 (red) and 2 (green)
    for xu, yu, xl, yl in zip(x1u, y1u, x1l, y1l):
        ax.plot([xl, xu], [yl, yu], '-', c='r', lw=0.5)
    for xu, yu, xl, yl in zip(x2u, y2u, x2l, y2l):
        ax.plot([xl, xu], [yl, yu], '-', c='g', lw=0.5)

    # compute max-decay percentages
    pct1_obs = (1 - r1M_obs[idx]) * 100
    pct1_fit = (1 - r1M_fit[idx]) * 100
    pct2_obs = (1 - r2M_obs[idx]) * 100
    pct2_fit = (1 - r2M_fit[idx]) * 100

    # three-line MathText title with a bit of pad
    title = (
        rf"$\mathbf{{{h:.1f}\ \mathrm{{m}}}}$" "\n"
        rf"$\mathit{{P1\ max:\ obs\ {pct1_obs:.1f}\%,\ fit\ {pct1_fit:.1f}\%}}$" "\n"
        rf"$\mathit{{P2\ max:\ obs\ {pct2_obs:.1f}\%,\ fit\ {pct2_fit:.1f}\%}}$"
    )
    ax.set_title(title, fontsize=10, pad=6)

    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

# delete any empty subplots
for j in range(len(sensor_heights), rows * cols):
    fig.delaxes(axs[j // cols][j % cols])

# shared colorbar, tightly packed
fig.colorbar(
    im, ax=axs,
    fraction=0.02,
    pad=0.02,
    label='Vertical Gradient (nT)'
)

# save with minimal margins for journal submission
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()

# --- Absolute decay & fits (both profiles) with customizable font sizes ---
plt.figure(figsize=(6,4))

# --- Font size settings (tweak these as needed) ---
label_fs  = 16   # x/y axis label font size
legend_fs = 18   # legend font size
tick_fs   = 14   # tick label font size

# observational uncertainties (1σ) – replace these with your actual values
sigma_min1 = 0.1
sigma_max1 = 0.1
sigma_min2 = 0.1
sigma_max2 = 0.1

err_min1 = np.full_like(min1_a, sigma_min1)
err_max1 = np.full_like(max1_a, sigma_max1)
err_min2 = np.full_like(min2_a, sigma_min2)
err_max2 = np.full_like(max2_a, sigma_max2)

# P1 – min
plt.errorbar(
    h_arr, min1_a,
    yerr=3 * err_min1,
    fmt='o', markersize=7, capsize=4,
    label='P1 min obs',
    color='tab:blue'
)
plt.plot(
    h_arr, eff_model(h_arr, A1m, n1m),
    '--', lw=1.2,
    color='tab:blue',
    label=f'P1 min fit (n={n1m:.2f})'
)

# P1 – max
plt.errorbar(
    h_arr, max1_a,
    yerr=3 * err_max1,
    fmt='s', markersize=7, capsize=4,
    label='P1 max obs',
    color='tab:orange'
)
plt.plot(
    h_arr, eff_model(h_arr, A1M, n1M),
    '-.', lw=1.2,
    color='tab:orange',
    label=f'P1 max fit (n={n1M:.2f})'
)

# P2 – min
plt.errorbar(
    h_arr, min2_a,
    yerr=3 * err_min2,
    fmt='x', markersize=7, capsize=4,
    label='P2 min obs',
    color='tab:green'
)
plt.plot(
    h_arr, eff_model(h_arr, A2m, n2m),
    ':', lw=1.2,
    color='tab:green',
    label=f'P2 min fit (n={n2m:.2f})'
)

# P2 – max
plt.errorbar(
    h_arr, max2_a,
    yerr=3 * err_max2,
    fmt='d', markersize=7, capsize=4,
    label='P2 max obs',
    color='tab:red'
)
plt.plot(
    h_arr, eff_model(h_arr, A2M, n2M),
    '-', lw=1.2,
    color='tab:red',
    label=f'P2 max fit (n={n2M:.2f})'
)

# Axis labels with adjustable font size
plt.xlabel('Sensor Height (m)', fontsize=label_fs)
plt.ylabel('Bz Anomaly (nT)',     fontsize=label_fs)

# Adjust tick label size
plt.xticks(fontsize=tick_fs)
plt.yticks(fontsize=tick_fs)

# Legend with adjustable font size
plt.legend(ncol=2, fontsize=legend_fs)

plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))

# --- Font size settings ---
label_fs  = 16   # x/y axis label font size
legend_fs = 18   # legend font size
tick_fs   = 14   # tick label font size

# Define a color map so fits match their scatters
colors = {
    'P1 min': 'tab:blue',
    'P1 max': 'tab:orange',
    'P2 min': 'tab:green',
    'P2 max': 'tab:red'
}

for obs, fit, label in [
    (min1_a/min1_a[0], eff_model(h_arr, A1m, n1m)/A1m, 'P1 min'),
    (max1_a/max1_a[0], eff_model(h_arr, A1M, n1M)/A1M, 'P1 max'),
    (min2_a/min2_a[0], eff_model(h_arr, A2m, n2m)/A2m, 'P2 min'),
    (max2_a/max2_a[0], eff_model(h_arr, A2M, n2M)/A2M, 'P2 max'),
]:
    c = colors[label]
    # plot only markers for observations
    plt.plot(
        h_arr, obs,
        'o', markersize=7,
        label=f'{label} obs',
        color=c
    )
    # dashed fit line, same color
    plt.plot(
        h_arr, fit,
        '--', linewidth=1.2,
        label=f'{label} fit',
        color=c
    )

# Axis labels with adjustable font size
plt.xlabel('Sensor Height (m)', fontsize=label_fs)
plt.ylabel('Relative to 0.2 m',    fontsize=label_fs)

# Adjust tick label size
plt.xticks(fontsize=tick_fs)
plt.yticks(fontsize=tick_fs)

# Legend with adjustable font size
plt.legend(ncol=2, fontsize=legend_fs)

plt.grid(True)
plt.tight_layout()
plt.show()


