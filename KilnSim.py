#!/usr/bin/env python3
"""
Compute vertical‐component Bᶻ anomaly from a buried “kiln” (uniformly magnetized circular cylinder)
plus soil trend, heterogeneity, and noise at ground level. Then upward‐continue via FFT to simulate
sensor measurements at varying heights, extract a horizontal profile at a configurable y‐position,
fit an effective power‐law to the global peak decay, and plot results.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import math
from mpl_toolkits.mplot3d import Axes3D

# Apply a clean style with white background
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
})



# =============================================================================
# 1) USER SETTINGS
# =============================================================================

# GRID
area_size          = 10.0     # [m] side‐length of square
spatial_resolution = 0.10     # [m] grid spacing

# “KILN” SOURCE
kiln_diameter      = 2.0      # [m]
kiln_thickness     = 1.0      # [m]
kiln_magnetization = 1.5      # [A/m]
kiln_depth         = 0.30     # [m] top below ground

# EARTH FIELD ORIENTATION
incl               = np.deg2rad(70.0)
decl               = np.deg2rad(3.0)

# SOIL + HETEROGENEITY
soil_slope    = 1.0           # [nT/m]
hetero_sigma  = 0.5           # [nT]
hetero_smooth = 10            # [pixels]

# SENSOR + NOISE
noise_density = 0.004         # [nT/√Hz]
sample_rate   = 10            # [Hz]
noise_std     = noise_density * np.sqrt(sample_rate/2)

# ELEVATIONS TO SIMULATE
sensor_heights = [0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5, 6, 10, 15]  # [m]

# PLOTTING RANGE
field_window = 100.0  # [nT]

# PROFILE SETTINGS
profile_y_position = 0  # [m] y‐coordinate of horizontal profile

# =============================================================================
# 2) BUILD CENTERED GRID & “KILN” MAP AT GROUND
# =============================================================================

# grid from -area_size/2 to +area_size/2
x = np.arange(-area_size/2, area_size/2 + spatial_resolution, spatial_resolution)
y = np.arange(-area_size/2, area_size/2 + spatial_resolution, spatial_resolution)
X, Y = np.meshgrid(x, y)
rows, cols = X.shape
dx = spatial_resolution

# sub‐dipoles inside circle centered at (0,0)
radius = kiln_diameter / 2.0
mask_circle = (X**2 + Y**2) <= radius**2
x0_list = X[mask_circle].ravel()
y0_list = Y[mask_circle].ravel()

# sub‐dipole moment components
sub_volume = dx*dx*kiln_thickness
m_scalar   = kiln_magnetization * sub_volume
mx = m_scalar * np.cos(incl)*np.cos(decl)
my = m_scalar * np.cos(incl)*np.sin(decl)
mz = m_scalar * np.sin(incl)

# =============================================================================
# 3) POINT‐DIPOLE Bᶻ
# =============================================================================
def point_dipole_Bz(Xr, Yr, z_s, depth):
    rx, ry, rz = Xr, Yr, z_s + depth
    r2 = rx*rx + ry*ry + rz*rz + 1e-12
    r3 = r2**1.5
    r5 = r2**2.5
    dot = mx*rx + my*ry + mz*rz
    coeff = 1e-7
    return coeff*(3*dot*rz/r5 - mz/r3)

# ground‐level Bᶻ (nT)
Bz_ground = np.zeros_like(X)
for x0, y0 in zip(x0_list, y0_list):
    Bz_ground += point_dipole_Bz(X - x0, Y - y0,
                                 z_s=0.0,
                                 depth=kiln_depth + kiln_thickness/2) * 1e9

# add trend, heterogeneity, noise
Bz_ground += soil_slope * X
Bz_ground += gaussian_filter(
    np.random.normal(0, hetero_sigma, Bz_ground.shape),
    sigma=hetero_smooth
)
Bz_ground += np.random.normal(0, noise_std, Bz_ground.shape)

Bz0 = np.ma.array(Bz_ground, mask=np.zeros_like(Bz_ground, bool))

# =============================================================================
# 4) UPWARD CONTINUATION VIA FFT
# =============================================================================
def upward_continue(grid, dx, h):
    data_f = grid.filled(0.0)
    ny, nx = data_f.shape
    kx = np.fft.fftfreq(nx, dx)*2*np.pi
    ky = np.fft.fftfreq(ny, dx)*2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    G = fft2(data_f)
    H = np.exp(-np.sqrt(KX**2 + KY**2)*h)
    cont = np.real(ifft2(G * H))
    return np.ma.array(cont, mask=grid.mask)

# =============================================================================
# 5) GENERATE MAPS, EXTRACT PROFILE & GLOBAL PEAK
# =============================================================================
maps_Bz    = []
profile_Bz = []
global_peaks = []

# find row index for profile
row_idx = np.argmin(np.abs(y - profile_y_position))

for h in sensor_heights:
    cont = upward_continue(Bz0, dx, h)
    cont += gaussian_filter(np.random.normal(0, noise_std, cont.shape), sigma=5)
    maps_Bz.append(cont)
    profile_Bz.append(cont[row_idx, :])
    global_peaks.append(np.max(np.abs(cont)))

h_arr  = np.array(sensor_heights)
peak_arr = np.array(global_peaks)

# =============================================================================
# 6) FIT EFFECTIVE POWER‐LAW TO GLOBAL PEAKS ONLY
# =============================================================================
def eff_model(h, A0, n):
    return A0 * (h/h_arr[0])**(-n)

p_e, _ = curve_fit(eff_model, h_arr, peak_arr, p0=[peak_arr[0], 0.7])

print(f"\nPower‐law fit: A0={p_e[0]:.3f}, n_eff={p_e[1]:.3f}")

# =============================================================================
# 7) PRINT PEAK-DECAY TABLE
# =============================================================================
print("\nHeight | Peak (nT) | % of baseline")
print("------------------------------------")
base = peak_arr[0]
for h, pk in zip(h_arr, peak_arr):
    print(f"{h:6.2f} | {pk:9.3f} | {pk/base*100:12.1f}%")

# =============================================================================
# 8) PLOTTING
# =============================================================================

# 8a) Maps with profile and grid (–5…+5 m)
# 1. Upward-continued Bz Maps (Figure 1)
cols = 6
rows = math.ceil(len(sensor_heights) / cols)
fig = plt.figure(figsize=(cols * 4, rows * 3), constrained_layout=True, dpi=100)
axs = [fig.add_subplot(rows, cols, i + 1) for i in range(rows * cols)]
baseline_peak = peak_arr[0]
for idx, (h, cont) in enumerate(zip(sensor_heights, maps_Bz)):
    ax = axs[idx]
    ax.set_facecolor('white')
    # only left/bottom spines crossing at -5
    ax.spines['left'].set_color('black'); ax.spines['left'].set_position(('data', -area_size/2))
    ax.spines['bottom'].set_color('black'); ax.spines['bottom'].set_position(('data', -area_size/2))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    im = ax.imshow(cont,
                   extent=[-area_size/2, area_size/2, -area_size/2, area_size/2],
                   origin='lower', cmap='seismic', vmin=-field_window, vmax=field_window,
                   interpolation='bicubic')
    pct = global_peaks[idx] / baseline_peak * 100
    ax.set_title(f"{h:.1f} m ({pct:.1f}% of baseline)")
    ax.set_xlim(-area_size/2, area_size/2); ax.set_ylim(-area_size/2, area_size/2)
    ax.set_xticks([-area_size/2, area_size/2]); ax.set_yticks([-area_size/2, area_size/2])
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.grid(False)
    ax.axhline(profile_y_position, color='red', linestyle='--', linewidth=1)
# remove slots
idx3d = len(sensor_heights)
fig.delaxes(axs[idx3d]); fig.delaxes(axs[idx3d+1])
# kiln 3D (as before)
r3, c3 = divmod(idx3d, cols)
ax3d = plt.subplot2grid((rows, cols), (r3, c3), colspan=2, projection='3d', fig=fig)
# geometry
top_r = kiln_diameter/2
flare = 0.5; thick = kiln_thickness
theta = np.linspace(0,2*np.pi,60); z = np.linspace(0,thick,40)
Z,Th = np.meshgrid(z,theta)
R = top_r*(1+flare*(Z/thick)); Xc = R*np.cos(Th); Yc = R*np.sin(Th)
# surfaces
ax3d.plot_surface(Xc,Yc,Z,color='dimgray',edgecolor='none',alpha=0.9)
ax3d.plot_surface(Xc*0.9,Yc*0.9,Z,color='darkred',edgecolor='none',alpha=0.7)
# soil plane
xp = np.linspace(-area_size/2,area_size/2,20)
Yp = xp; Xp,Yp = np.meshgrid(xp,Yp); Zp = np.zeros_like(Xp)
ax3d.plot_surface(Xp,Yp,Zp,color='sienna',edgecolor='none',alpha=0.3)
# axes
ax3d.set_xlim(-area_size/2,area_size/2); ax3d.set_ylim(-area_size/2,area_size/2); ax3d.set_zlim(0,thick)
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_xlabel(''); ax3d.set_ylabel('')
ax3d.set_zlabel('Depth (m)',labelpad=10)
"""ax3d.set_title('Kiln (Flared Base, Burned-stone & Slag)')"""
ax3d.view_init(elev=30,azim=45)
# colorbar
cbar = fig.colorbar(im, ax=axs[:len(sensor_heights)], orientation='vertical', pad=0.02)
cbar.set_label('$B_z$ Anomaly (nT)')
"""fig.suptitle(f"Figure 1. Upward-Continued $B_z$ Maps with Profile at y = {profile_y_position:.1f} m")"""
plt.show()

# 2. Profile vs X (Figure 2)

# Set up font sizes
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 18
title_fontsize = 16
fig2 = plt.figure(figsize=(8, 5), dpi=100)
# Plot each profile
for h, prof in zip(sensor_heights, profile_Bz):
    plt.plot(x, prof, label=f"{h:.1f} m", linewidth=2)
# Labels with custom font size
plt.xlabel('X (m)', fontsize=label_fontsize)
plt.ylabel('$B_z$ Anomaly (nT)', fontsize=label_fontsize)
# Optional title (commented out)
# plt.title(f"Figure 2. Profile at y={profile_y_position:.1f} m", fontsize=title_fontsize)
# Legend and grid
plt.legend(title='Height', fontsize=legend_fontsize, title_fontsize=legend_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Global Peak Decay vs Height (Figure 3)
# Set up font sizes
label_fontsize = 16
tick_fontsize = 12
legend_fontsize = 18
title_fontsize = 16
fig3 = plt.figure(figsize=(8, 5), dpi=100)
# Scatter points
plt.scatter(h_arr, peak_arr, s=50, label='Observed')
# Fit line with custom color
plt.plot(h_arr, eff_model(h_arr, *p_e), linewidth=2, color='crimson', label=f"Fit n={p_e[1]:.2f}")
# Labels and styling
plt.xlabel('Height (m)', fontsize=label_fontsize)
plt.ylabel('Peak |$B_z$| (nT)', fontsize=label_fontsize)
# plt.title('Figure 3. Peak Decay & Fit', fontsize=title_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. Heatmap of Profile vs Height (Figure 4)
fig4 = plt.figure(figsize=(8,6), dpi=150)
A = np.vstack(profile_Bz)
plt.imshow(A,aspect='auto',extent=[-area_size/2,area_size/2,sensor_heights[-1],sensor_heights[0]],origin='lower',cmap='seismic',vmin=-max(peak_arr),vmax=max(peak_arr),interpolation='bicubic')
plt.colorbar(label='$B_z$ Anomaly (nT)')
plt.xlabel('X (m)'); plt.ylabel('Height (m)');
"""plt.title('Figure 4. Profile Heatmap'); """
plt.tight_layout(); plt.show()

"""
# 5. 3 Perspectives of 3D Kiln with Volumetric Anomaly (Figure 5)
# 5. 3 Perspectives of 3D Kiln with Volumetric Anomaly (Figure 5)
fig5 = plt.figure(figsize=(12, 4), dpi=100, constrained_layout=True)
titles = ['Tilt Perspective', 'Profile Perspective', 'Top-down View']
views = [(30, 45), (30, 0), (90, 0)]
# prepare anomaly volume over full grid
x_vol = X.flatten()
y_vol = Y.flatten()
b_vol = Bz_ground.flatten()
# normalize anomaly for colormap
vmin, vmax = -field_window, field_window
norm_b = (b_vol - vmin) / (vmax - vmin)
# choose Z levels for volume sampling
nz = 15
z_levels = np.linspace(0, thick, nz)
for i, (title, (elev, azim)) in enumerate(zip(titles, views)):
    axp = fig5.add_subplot(1, 3, i+1, projection='3d')
    # plot anomaly volume as semi-transparent scatter layers
    for z_val in z_levels:
        axp.scatter(
            x_vol, y_vol, z_val,
            c=norm_b, cmap='seismic', alpha=0.05, s=2
        )
    # plot kiln surfaces
    axp.plot_surface(Xc, Yc, Z, color='dimgray', edgecolor='none', alpha=0.9)
    axp.plot_surface(Xc*0.9, Yc*0.9, Z, color='darkred', edgecolor='none', alpha=0.7)
    # invert z-axis so 0 at top
    axp.set_zlim(thick, 0)
    # axes limits
    axp.set_xlim(-area_size/2, area_size/2)
    axp.set_ylim(-area_size/2, area_size/2)
    # set ticks and labels for X and Y
    axp.set_xticks([-area_size/2, area_size/2])
    axp.set_yticks([-area_size/2, area_size/2])
    axp.set_xticklabels([f"{-area_size/2:.0f}", f"{area_size/2:.0f}"])
    axp.set_yticklabels([f"{-area_size/2:.0f}", f"{area_size/2:.0f}"])
    if title == 'Top-down View':
        # no depth label or ticks
        axp.set_zticks([])
        axp.set_xlabel('X (m)', labelpad=10)
        axp.set_ylabel('Y (m)', labelpad=10)
        axp.set_zlabel('')
    else:
        # show depth ticks and label
        axp.set_zticks([0, thick])
        axp.set_zticklabels(['0', f"{thick:.0f}"])
        axp.set_xlabel('X (m)', labelpad=10)
        axp.set_ylabel('Y (m)', labelpad=10)
        axp.set_zlabel('Depth (m)', labelpad=5)
    axp.view_init(elev=elev, azim=azim)
    axp.set_title(title)
plt.show()
"""


