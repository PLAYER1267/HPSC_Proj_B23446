"""
plot_results.py — Publication-quality figures for thermal stress casting solver
Usage: python3 plot_results.py [results_dir]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'legend.fontsize':11,
    'figure.dpi':     150,
    'lines.linewidth':1.8,
    'axes.grid':      True,
    'grid.alpha':     0.3,
})

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

def save(fig, name):
    p = FIGS / (name + ".pdf")
    fig.tight_layout()
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {p}")

def load(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None

def load_field(path):
    """Load a temperature/strain field CSV (rows=i, cols=j, first col is i-index)."""
    p = Path(path)
    if not p.exists(): return None
    df = pd.read_csv(p, index_col=0)
    return df.values.astype(float)

# ============================================================
# 1. Steady-state: temperature field + centre-row profile
# ============================================================
def plot_steady_state():
    od = ROOT / "steady_state"
    # Final temperature field
    T = load_field(od / "T_step000375.csv")
    if T is None:
        # Try last available
        files = sorted((od).glob("T_step*.csv"))
        if files: T = load_field(files[-1])
    if T is None: print("  [skip] steady_state field"); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 2D heatmap
    im = axes[0].imshow(T.T, origin='lower', aspect='equal',
                        cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=axes[0], label='Temperature (K)')
    axes[0].set_title('Steady-State Temperature Field')
    axes[0].set_xlabel('i (x-direction)')
    axes[0].set_ylabel('j (y-direction)')

    # Centre-row profile vs analytical
    df = load(od / "centre_row_profile.csv")
    if df is not None:
        axes[1].plot(df['x'], df['T'],        lw=2,   label='Numerical')
        axes[1].plot(df['x'], df['T_linear'], lw=2, ls='--', label='Analytical (linear)')
        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].set_title('Centre-Row Profile (Steady State)')
        axes[1].legend()

    # Error from linear
    if df is not None:
        err = np.abs(df['T'] - df['T_linear'])
        axes[2].semilogy(df['x'], err + 1e-12, lw=2, color='firebrick')
        axes[2].set_xlabel('x (m)')
        axes[2].set_ylabel('|T - T_linear| (K)')
        axes[2].set_title('Deviation from Linear Gradient')

    save(fig, "fig_steady_state")

# ============================================================
# 2. Cooling curve: energy decay + temperature snapshots
# ============================================================
def plot_cooling_curve():
    od = ROOT / "cooling_curve"
    df = load(od / "diagnostics.csv")
    if df is None: print("  [skip] cooling_curve diagnostics"); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Total energy vs time
    axes[0].plot(df['time'], df['total_energy'], lw=2, color='steelblue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Σ T  (proxy for energy)')
    axes[0].set_title('Total Energy vs Time')

    # Mean + max temperature
    axes[1].plot(df['time'], df['T_mean'], lw=2, label=r'$\bar{T}$')
    axes[1].plot(df['time'], df['T_max'],  lw=2, ls='--', label='$T_{max}$')
    axes[1].plot(df['time'], df['T_min'],  lw=2, ls=':',  label='$T_{min}$')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Temperature Statistics vs Time')
    axes[1].legend()

    # Strain RMS vs time
    axes[2].plot(df['time'], df['strain_rms'], lw=2, color='darkorange')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel(r'$\epsilon_{rms}$ (dimensionless)')
    axes[2].set_title('RMS Thermal Strain vs Time')

    save(fig, "fig_cooling_curve")

    # Also plot final temperature field
    files = sorted((od).glob("T_step*.csv"))
    if len(files) >= 2:
        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, fpath, label in zip(axes2, [files[0], files[-1]], ['t=0', 'Final']):
            T = load_field(fpath)
            if T is None: continue
            im = ax.imshow(T.T, origin='lower', aspect='equal',
                           cmap='hot', vmin=300, vmax=1500, interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Temperature (K)')
            ax.set_title(f'Temperature Field ({label})')
            ax.set_xlabel('i'); ax.set_ylabel('j')
        save(fig2, "fig_cooling_snapshots")

# ============================================================
# 3. Geometry experiment results
# ============================================================
def plot_geometry():
    df = load(ROOT / "geometry" / "geometry_results.csv")
    if df is None: print("  [skip] geometry_results.csv"); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#1f77b4', '#ff7f0e']

    for ax, col, ylabel, key in zip(
            axes,
            ['#e15759','#4e79a7','#f28e2b'],
            ['Final Mean T (K)', 'Max Strain $|\\epsilon|$', 'RMS Strain'],
            ['final_Tmean', 'max_strain', 'strain_rms']):

        for feeder_val, lbl, col in [(0,'No feeder','steelblue'),(1,'With feeder','darkorange')]:
            sub = df[df['use_feeder'] == feeder_val]
            ax.plot(sub['insulator_thickness'], sub[key], 'o-',
                    color=col, label=lbl, ms=7)
        ax.set_xlabel('Insulator thickness (cells)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('(')[0].strip())
        ax.legend()

    save(fig, "fig_geometry_experiment")

    # Show temperature fields for thinnest and thickest insulator (no feeder)
    cases = [("geom_t1_nofeed","Thin insulator (t=1)"),
             ("geom_t6_nofeed","Thick insulator (t=6)"),
             ("geom_t3_feeder","Medium + feeder")]
    files_exist = [ROOT / "geometry" / c / sorted(
        list((ROOT/"geometry"/c).glob("T_step*.csv")))[-1]
        for c,_ in cases
        if (ROOT/"geometry"/c).exists() and
           list((ROOT/"geometry"/c).glob("T_step*.csv"))]

    valid_cases = [(c,l) for c,l in cases
                   if (ROOT/"geometry"/c).exists() and
                   list((ROOT/"geometry"/c).glob("T_step*.csv"))]
    if valid_cases:
        fig2, axes2 = plt.subplots(1, len(valid_cases),
                                    figsize=(5*len(valid_cases), 4.5))
        if len(valid_cases) == 1: axes2 = [axes2]
        for ax, (cname, clabel) in zip(axes2, valid_cases):
            flist = sorted((ROOT/"geometry"/cname).glob("T_step*.csv"))
            T = load_field(flist[-1])
            if T is None: continue
            im = ax.imshow(T.T, origin='lower', aspect='equal',
                           cmap='RdYlBu_r', interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='T (K)')
            ax.set_title(clabel)
            ax.set_xlabel('i'); ax.set_ylabel('j')
        save(fig2, "fig_geometry_fields")

    # Strain fields
    valid_strain = [(c,l) for c,l in cases
                    if (ROOT/"geometry"/c).exists() and
                    list((ROOT/"geometry"/c).glob("strain_step*.csv"))]
    if valid_strain:
        fig3, axes3 = plt.subplots(1, len(valid_strain),
                                    figsize=(5*len(valid_strain), 4.5))
        if len(valid_strain) == 1: axes3 = [axes3]
        for ax, (cname, clabel) in zip(axes3, valid_strain):
            flist = sorted((ROOT/"geometry"/cname).glob("strain_step*.csv"))
            E = load_field(flist[-1])
            if E is None: continue
            vmax = np.max(np.abs(E)) if np.max(np.abs(E)) > 0 else 1e-6
            im = ax.imshow(E.T, origin='lower', aspect='equal',
                           cmap='seismic', vmin=-vmax, vmax=vmax,
                           interpolation='bilinear')
            plt.colorbar(im, ax=ax, label=r'$\epsilon_{thermal}$')
            ax.set_title(f'Strain — {clabel}')
            ax.set_xlabel('i'); ax.set_ylabel('j')
        save(fig3, "fig_geometry_strain")

# ============================================================
# 4. Profiling pie chart
# ============================================================
def plot_profiling():
    df = load(ROOT / "steady_state" / "profile.csv")
    if df is None: print("  [skip] profile.csv"); return

    df = df[df['time_s'] > 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#e15759','#4e79a7','#f28e2b','#76b7b2']
    explode = [0.06 if p == df['percent'].max() else 0 for p in df['percent']]
    wedges, texts, autotexts = ax.pie(
        df['percent'], labels=df['stage'],
        autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
    for at in autotexts: at.set_fontsize(10)
    ax.set_title('Runtime Distribution (80×80 grid, serial)')
    save(fig, "fig_profiling")

# ============================================================
# 5. Scaling: strong scaling + grid size
# ============================================================
def plot_scaling():
    # Strong scaling
    df_s = load(ROOT / "scaling" / "strong_scaling.csv")
    # Grid size
    df_g = load(ROOT / "scaling" / "grid_scaling.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    if df_s is not None:
        axes[0].plot(df_s['threads'], df_s['speedup'],    'o-', color='steelblue', ms=7)
        t_max = df_s['threads'].max()
        axes[0].plot([1, t_max], [1, t_max], 'k:', lw=1.2, label='Ideal')
        axes[0].set_xlabel('Threads')
        axes[0].set_ylabel('Speedup $S(p)$')
        axes[0].set_title('Strong Scaling: Speedup (200×200)')
        axes[0].legend()

        axes[1].plot(df_s['threads'], df_s['efficiency'], 's--', color='darkorange', ms=7)
        axes[1].axhline(1.0, color='k', ls=':', lw=1.2)
        axes[1].set_xlabel('Threads')
        axes[1].set_ylabel('Efficiency $E(p)$')
        axes[1].set_title('Strong Scaling: Efficiency')
        axes[1].set_ylim(0, 1.15)

    if df_g is not None:
        N2 = df_g['Nx'] * df_g['Ny']
        axes[2].loglog(N2, df_g['wall_s'], 'o-', color='seagreen', ms=7)
        # O(N^2) reference
        ref = N2**1.0 / N2.iloc[0]**1.0 * df_g['wall_s'].iloc[0]
        axes[2].loglog(N2, ref, 'k:', lw=1.2, label='$O(N^2)$')
        axes[2].set_xlabel('Total grid nodes $N_x \\times N_y$')
        axes[2].set_ylabel('Runtime (s)')
        axes[2].set_title('Grid-Size Scaling (Serial)')
        axes[2].legend()

    save(fig, "fig_scaling")

# ============================================================
# 6. Combined temperature + strain heatmap (cooling curve final)
# ============================================================
def plot_final_state():
    od = ROOT / "cooling_curve"
    T_files = sorted(od.glob("T_step*.csv"))
    E_files = sorted(od.glob("strain_step*.csv"))
    if not T_files or not E_files:
        print("  [skip] final state fields"); return

    T = load_field(T_files[-1])
    E = load_field(E_files[-1])
    if T is None or E is None: return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im1 = axes[0].imshow(T.T, origin='lower', aspect='equal',
                          cmap='hot', interpolation='bilinear')
    plt.colorbar(im1, ax=axes[0], label='Temperature (K)')
    axes[0].set_title('Final Temperature Distribution')
    axes[0].set_xlabel('i (x)'); axes[0].set_ylabel('j (y)')

    vmax = np.max(np.abs(E))
    im2 = axes[1].imshow(E.T, origin='lower', aspect='equal',
                          cmap='seismic', vmin=-vmax, vmax=vmax,
                          interpolation='bilinear')
    plt.colorbar(im2, ax=axes[1], label=r'$\epsilon_{thermal}$')
    axes[1].set_title('Final Thermal Strain Field')
    axes[1].set_xlabel('i (x)'); axes[1].set_ylabel('j (y)')

    save(fig, "fig_final_state")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating figures...")
    plot_steady_state()
    plot_cooling_curve()
    plot_geometry()
    plot_profiling()
    plot_scaling()
    plot_final_state()
    print(f"\nAll figures saved to {FIGS}/")
