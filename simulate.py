"""
Pro Wrestling Biomechanics - first-order physics model of 23 move types.

Four mechanisms:
  bump   - receiver falls onto a surface (free-fall + surface compliance + technique)
  power  - lift/slam: lumbar load on the lifter + landing load on the receiver
  aerial - attacker flight (projectile + rotation) ending in body-to-body contact
  strike - collision model for kicks/arm strikes, with a "worked" (pulled) fraction

All outputs are model ESTIMATES from stated assumptions, not measured data.
Run:  python src/simulate.py   (from project root)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

G = 9.81
HALF_SINE = np.pi / 2          # peak/average force for a half-sine pulse
NIOSH_AL, NIOSH_MPL = 3400.0, 6400.0   # N, lumbar compression (Waters et al. 1993)
BOXER_HEAD_G = 71.0            # mean peak head accel, Olympic boxer punches (Walilko et al. 2005)
M_HEAD = 6.0                   # kg, effective head + neck mass for linear accel estimate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, FIG = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# surface: (contact time s, fraction of impact energy absorbed by surface failure)
SURF = {"ring": (0.060, 0.0), "floor": (0.040, 0.0), "table": (0.090, 0.30)}

COL = {"Bump": "#c0392b", "Power": "#8e44ad", "Aerial": "#2980b9", "Strike": "#d35400"}

# ----------------------------------------------------------------------------------
# MOVE LIBRARY  (parameters are modelling assumptions - see README "Assumptions")
# ----------------------------------------------------------------------------------
MOVES = [
 # ---- BUMPS: h = CoM drop (m), tech = contact-time multiplier from flat-body/slap technique
 dict(name="Flat-back bump (standing)", cat="Bump", kind="bump", h=0.85, surf="ring", tech=1.4, slap=0.20, m_rec=100),
 dict(name="Apron-to-floor bump",       cat="Bump", kind="bump", h=1.70, surf="floor", tech=1.2, slap=0.15, m_rec=100),
 dict(name="Through a table (standing)",cat="Bump", kind="bump", h=1.10, surf="table", tech=1.2, slap=0.10, m_rec=100),
 dict(name="Fall from ladder (top)",    cat="Bump", kind="bump", h=3.00, surf="ring", tech=1.5, slap=0.20, m_rec=100),
 # ---- POWER: lift dh (m), t_lift (s), d_load = horizontal load distance from L5/S1 (m)
 dict(name="Vertical suplex",     cat="Power", kind="power", h=1.50, surf="ring", tech=1.4, slap=0.15, m_rec=100, m_att=105, dh=0.90, t_lift=1.2, d_load=0.35),
 dict(name="Powerbomb",           cat="Power", kind="power", h=1.95, surf="ring", tech=1.4, slap=0.15, m_rec=100, m_att=105, dh=1.10, t_lift=1.0, d_load=0.30),
 dict(name="Chokeslam",           cat="Power", kind="power", h=1.70, surf="ring", tech=1.4, slap=0.15, m_rec=100, m_att=105, dh=1.00, t_lift=0.9, d_load=0.55),
 dict(name="Fireman's carry slam (F-5 type)", cat="Power", kind="power", h=1.30, surf="ring", tech=1.4, slap=0.15, m_rec=100, m_att=105, dh=0.60, t_lift=1.0, d_load=0.25, spin=1.3),
 dict(name="Top-rope superplex",  cat="Power", kind="power", h=2.40, surf="ring", tech=1.3, slap=0.10, m_rec=100, m_att=105, dh=0.80, t_lift=1.3, d_load=0.35, pile=0.5),
 dict(name="Powerbomb through table", cat="Power", kind="power", h=1.95, surf="table", tech=1.2, slap=0.10, m_rec=100, m_att=105, dh=1.10, t_lift=1.0, d_load=0.30),
 # ---- AERIALS: h0 launch CoM height, vy0 vertical launch speed, vx horizontal speed, n_rot rotations, hc contact CoM, dtc contact time
 dict(name="Frog splash",        cat="Aerial", kind="aerial", m_att=100, m_rec=100, h0=2.70, vy0=2.0, vx=0.8, n_rot=0.00, hc=0.70, dtc=0.070),
 dict(name="Swanton bomb",       cat="Aerial", kind="aerial", m_att=100, m_rec=100, h0=2.70, vy0=2.8, vx=0.8, n_rot=0.50, hc=0.70, dtc=0.070),
 dict(name="Moonsault (top rope)",cat="Aerial", kind="aerial", m_att=100, m_rec=100, h0=2.70, vy0=3.2, vx=0.8, n_rot=1.00, hc=0.70, dtc=0.070),
 dict(name="Shooting star press",cat="Aerial", kind="aerial", m_att=100, m_rec=100, h0=2.70, vy0=3.6, vx=0.8, n_rot=1.00, hc=0.70, dtc=0.070),
 dict(name="450 splash",         cat="Aerial", kind="aerial", m_att=100, m_rec=100, h0=2.70, vy0=3.8, vx=0.8, n_rot=1.25, hc=0.70, dtc=0.070),
 dict(name="Asai moonsault (to floor)", cat="Aerial", kind="aerial", m_att=90, m_rec=100, h0=2.00, vy0=3.2, vx=2.5, n_rot=1.00, hc=1.20, dtc=0.090),
 dict(name="Tope suicida",       cat="Aerial", kind="aerial", m_att=95, m_rec=100, h0=1.70, vy0=1.5, vx=4.0, n_rot=0.00, hc=1.00, dtc=0.090),
 # ---- STRIKES: v = striking-segment speed (m/s), m_eff = effective striking mass (kg), dt = contact (s), e = restitution, work = fraction of full-force actually delivered
 dict(name="Superkick",   cat="Strike", kind="strike", v=11.0, m_eff=3.0, dt=0.015, e=0.2, work=0.20, head=True,  m_rec=100),
 dict(name="Enzuigiri",   cat="Strike", kind="strike", v=10.0, m_eff=3.0, dt=0.015, e=0.2, work=0.20, head=True,  m_rec=100),
 dict(name="Lariat",      cat="Strike", kind="strike", v=8.0,  m_eff=6.0, dt=0.020, e=0.1, work=0.30, head=True,  m_rec=100),
 dict(name="Big boot",    cat="Strike", kind="strike", v=7.5,  m_eff=6.0, dt=0.020, e=0.1, work=0.25, head=True,  m_rec=100),
 dict(name="Spear",       cat="Strike", kind="strike", v=5.5,  m_eff=63.0, dt=0.100, e=0.1, work=0.50, head=False, m_rec=100),
 dict(name="Open-hand chop", cat="Strike", kind="strike", v=6.5, m_eff=1.5, dt=0.020, e=0.1, work=0.70, head=False, m_rec=100),
]

def compute(mv, rng=None):
    """Return dict of biomechanical metrics. rng -> Monte Carlo perturbation."""
    p = dict(mv)
    if rng is not None:
        p["m_rec"] = float(np.clip(rng.normal(p["m_rec"], 15), 70, 150))
        if "m_att" in p: p["m_att"] = float(np.clip(rng.normal(p["m_att"], 12), 70, 150))
        if p["kind"] in ("bump", "power"):
            p["h"] *= rng.normal(1, 0.08); p["tech"] *= rng.lognormal(0, 0.20)
        if p["kind"] == "aerial":
            p["vy0"] *= rng.normal(1, 0.08); p["dtc"] *= rng.lognormal(0, 0.20)
        if p["kind"] == "strike":
            p["v"] *= rng.normal(1, 0.10); p["dt"] *= rng.lognormal(0, 0.20)
            p["work"] = float(np.clip(rng.normal(p["work"], 0.06), 0.05, 1.0))
    out = dict(lumbar_N=0.0, rot_rate=0.0, land_err=0.0, head_g=0.0, flight_t=0.0)
    kind, mr = p["kind"], p["m_rec"]
    if kind in ("bump", "power"):
        dts, absorb = SURF[p["surf"]]
        v = np.sqrt(2 * G * p["h"])
        ke = mr * G * p["h"]
        fE = (1 - absorb) * (1 - p["slap"])
        v_eff = v * np.sqrt(fE)
        dt = dts * p["tech"]
        f = HALF_SINE * mr * v_eff / dt
        if kind == "power":
            f *= 1 + p.get("pile", 0.0) * p["m_att"] / mr
            a = 2 * p["dh"] / p["t_lift"] ** 2
            mt = 0.5 * p["m_att"]
            moment = mr * G * (1 + a / G) * p["d_load"] * p.get("spin", 1.0) + mt * G * 0.10
            out["lumbar_N"] = moment / 0.06 + mr * G * (1 + a / G)
        out.update(v_imp=v, energy=ke, contact=dt, peak_F=f, peak_G=f / (mr * G))
    elif kind == "aerial":
        ma = p["m_att"]
        tf = (p["vy0"] + np.sqrt(p["vy0"] ** 2 + 2 * G * (p["h0"] - p["hc"]))) / G
        vy = p["vy0"] - G * tf
        v = np.hypot(p["vx"], vy)
        f = HALF_SINE * ma * v / p["dtc"]
        w = 2 * np.pi * p["n_rot"] / tf
        out.update(v_imp=v, energy=0.5 * ma * v ** 2, contact=p["dtc"], peak_F=f,
                   peak_G=max(f / (mr * G), f / (ma * G)), flight_t=tf, rot_rate=w,
                   land_err=np.degrees(w * 0.05))
    else:  # strike
        j = p["m_eff"] * p["v"] * (1 + p["e"])
        f_full = HALF_SINE * j / p["dt"]
        f = f_full * p["work"]
        out.update(v_imp=p["v"], energy=0.5 * p["m_eff"] * p["v"] ** 2 * p["work"],
                   contact=p["dt"], peak_F=f, peak_G=f / (mr * G), f_full=f_full)
        if p.get("head"):
            out["head_g"] = f / M_HEAD / G
            out["head_g_full"] = f_full / M_HEAD / G
    return out

# ----------------------------------------------------------------------------------
# BASE RUN + RELATIVE LOAD INDEX (RLI)
# ----------------------------------------------------------------------------------
W = np.array([0.40, 0.20, 0.15, 0.15, 0.10])   # peak G, energy, lumbar, rotation, head accel
COMP = ["c_peakG", "c_energy", "c_lumbar", "c_rot", "c_head"]

def build_base():
    rows = []
    for mv in MOVES:
        m = compute(mv)
        rows.append(dict(move=mv["name"], category=mv["cat"], mechanism=mv["kind"],
                         impact_speed_ms=m["v_imp"], impact_energy_J=m["energy"], contact_ms=m["contact"] * 1000,
                         peak_force_kN=m["peak_F"] / 1000, peak_G=m["peak_G"],
                         lumbar_kN=m["lumbar_N"] / 1000, lumbar_vs_NIOSH_MPL=m["lumbar_N"] / NIOSH_MPL,
                         flight_time_s=m["flight_t"], rot_rate_rads=m["rot_rate"], landing_err_deg_at_50ms=m["land_err"],
                         head_accel_g=m["head_g"], head_accel_full_g=m.get("head_g_full", 0.0),
                         strike_force_full_kN=m.get("f_full", 0) / 1000))
    df = pd.DataFrame(rows)
    return df

def components(df):
    mx = dict(peak_G=df.peak_G.max(), energy=df.impact_energy_J.max(), lumbar=df.lumbar_kN.max(),
              rot=df.rot_rate_rads.max(), head=df.head_accel_g.max())
    return mx

def rli_from(vals, mx, w=W):
    c = np.array([vals["peak_G"] / mx["peak_G"], vals["energy"] / mx["energy"],
                  vals["lumbar_N"] / 1000 / mx["lumbar"], vals["rot_rate"] / mx["rot"],
                  vals["head_g"] / mx["head"]])
    return 100 * float(c @ w), c

def run():
    df = build_base(); mx = components(df)
    cs, ris = [], []
    for mv in MOVES:
        m = compute(mv)
        r, c = rli_from(m, mx); ris.append(r); cs.append(c)
    df["RLI"] = ris
    for i, k in enumerate(COMP): df[k] = [c[i] for c in cs]
    df["rank"] = df.RLI.rank(ascending=False).astype(int)

    # Monte Carlo (5,000 draws per move)
    rng = np.random.default_rng(2026); N = 5000
    mc_rows, mc_ri = [], {}
    for mv in MOVES:
        gs, rs = [], []
        for _ in range(N):
            m = compute(mv, rng); gs.append(m["peak_G"]); rs.append(rli_from(m, mx)[0])
        gs, rs = np.array(gs), np.array(rs); mc_ri[mv["name"]] = rs
        mc_rows.append(dict(move=mv["name"], category=mv["cat"],
                            G_p05=np.percentile(gs, 5), G_p50=np.percentile(gs, 50), G_p95=np.percentile(gs, 95),
                            RLI_p05=np.percentile(rs, 5), RLI_p50=np.percentile(rs, 50), RLI_p95=np.percentile(rs, 95)))
    mc = pd.DataFrame(mc_rows)
    df = df.merge(mc[["move", "G_p05", "G_p95", "RLI_p05", "RLI_p95"]], on="move")

    # weight sensitivity: 2,000 random Dirichlet weight vectors
    rng2 = np.random.default_rng(7)
    C = df[COMP].values
    ranks = np.zeros((2000, len(df)))
    for i in range(2000):
        w = rng2.dirichlet(np.ones(5)); s = C @ w
        ranks[i] = (-s).argsort().argsort() + 1
    df["rank_p05"] = np.percentile(ranks, 5, axis=0); df["rank_p95"] = np.percentile(ranks, 95, axis=0)
    df["rank_median"] = np.median(ranks, axis=0)

    df.round(4).to_csv(os.path.join(DATA, "moves_biomechanics.csv"), index=False)
    mc.round(4).to_csv(os.path.join(DATA, "monte_carlo_summary.csv"), index=False)
    pd.DataFrame(MOVES).to_csv(os.path.join(DATA, "move_parameters_assumptions.csv"), index=False)
    return df, mc, ranks

# ----------------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def legend_cats(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=c, label=k) for k, c in COL.items()], loc=loc, frameon=False)

def fig_ranking(df):
    d = df.sort_values("RLI")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(d.move, d.RLI, color=[COL[c] for c in d.category])
    ax.errorbar(d.RLI, d.move, xerr=[d.RLI - d.RLI_p05, d.RLI_p95 - d.RLI], fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.set_xlabel("Relative Load Index (0-100), bars = Monte Carlo 5-95%")
    ax.set_title("Relative Load Index by move"); legend_cats(ax)
    fig.savefig(os.path.join(FIG, "fig1_rli_ranking.png")); plt.close(fig)

def fig_force_energy(df):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c, g in df.groupby("category"):
        ax.scatter(g.impact_energy_J, g.peak_force_kN, s=45, color=COL[c], label=c)
        for _, r in g.iterrows():
            ax.annotate(r.move.split(" (")[0], (r.impact_energy_J, r.peak_force_kN), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Impact energy (J)"); ax.set_ylabel("Peak contact force (kN)")
    ax.set_title("Energy vs peak force: why some moves hurt more per joule"); ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig2_force_vs_energy.png")); plt.close(fig)

def fig_drop(df):
    h = np.linspace(0.3, 3.5, 100)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(h, np.sqrt(2 * G * h), color="k"); ax[0].set_xlabel("CoM drop height (m)"); ax[0].set_ylabel("Impact speed (m/s)")
    ax[0].set_title("Free-fall impact speed")
    for s, (dts, ab) in SURF.items():
        v = np.sqrt(2 * G * h * (1 - ab) * 0.85)
        gg = HALF_SINE * v / (dts * 1.3) / G
        ax[1].plot(h, gg, label=f"{s} (dt={dts*1000:.0f} ms)")
    for _, r in df[df.mechanism.isin(["bump", "power"])].iterrows():
        mv = next(m for m in MOVES if m["name"] == r.move)
        ax[1].scatter(mv["h"], r.peak_G, color=COL[r.category], s=22, zorder=3)
    ax[1].set_xlabel("CoM drop height (m)"); ax[1].set_ylabel("Peak body acceleration (g)")
    ax[1].set_title("Peak g vs height by surface (dots = modelled moves)"); ax[1].legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig3_drop_height_surface.png")); plt.close(fig)

def fig_aerial(df):
    a = df[df.mechanism == "aerial"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].barh(a.move, a.rot_rate_rads, color=COL["Aerial"]); ax[0].set_xlabel("Required rotation rate (rad/s)")
    ax[0].axvline(6, ls="--", color="grey"); ax[0].text(6.1, 0, "tuck needed", fontsize=7, color="grey")
    ax[0].set_title("Rotation demand over flight")
    ax[1].barh(a.move, a.landing_err_deg_at_50ms, color="#7fb3d5"); ax[1].set_yticklabels([])
    ax[1].set_xlabel("Landing-angle error (deg) for 50 ms timing error"); ax[1].set_title("Margin for error")
    fig.savefig(os.path.join(FIG, "fig4_aerial_rotation.png")); plt.close(fig)

def fig_lumbar(df):
    p = df[df.mechanism == "power"].sort_values("lumbar_kN")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(p.move, p.lumbar_kN, color=COL["Power"])
    ax.axvline(NIOSH_AL / 1000, ls="--", color="green"); ax.axvline(NIOSH_MPL / 1000, ls="--", color="red")
    ax.text(NIOSH_AL / 1000 + .1, -0.45, "NIOSH AL 3.4 kN", color="green", fontsize=7)
    ax.text(NIOSH_MPL / 1000 + .1, -0.45, "NIOSH MPL 6.4 kN", color="red", fontsize=7)
    ax.set_xlabel("Estimated L5/S1 compression on the lifter (kN)")
    ax.set_title("Lifter spine load vs occupational reference limits")
    fig.savefig(os.path.join(FIG, "fig5_lifter_lumbar_load.png")); plt.close(fig)

def fig_strikes(df):
    s = df[(df.mechanism == "strike") & (df.head_accel_g > 0)]
    fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(s))
    ax.bar(x - .2, s.head_accel_full_g, .4, color="#e59866", label="Full contact (model)")
    ax.bar(x + .2, s.head_accel_g, .4, color=COL["Strike"], label="Worked (pulled)")
    ax.axhline(BOXER_HEAD_G, ls="--", color="k"); ax.text(len(s) - .5, BOXER_HEAD_G + 3, "Olympic boxer punch ~71 g", ha="right", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(s.move, rotation=20); ax.set_ylabel("Linear head acceleration (g)")
    ax.set_title("What 'working' a strike buys: head acceleration, full vs pulled"); ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig6_strikes_worked_vs_full.png")); plt.close(fig)

def fig_sens(df, ranks):
    order = df.sort_values("RLI", ascending=False).index
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.boxplot([ranks[:, i] for i in order], orientation="horizontal", tick_labels=df.loc[order, "move"], showfliers=False)
    ax.invert_yaxis(); ax.set_xlabel("Rank under 2,000 random weight sets (1 = highest load)")
    ax.set_title("Rank stability to index weights")
    fig.savefig(os.path.join(FIG, "fig7_weight_sensitivity.png")); plt.close(fig)

def fig_model_map():
    fig, ax = plt.subplots(figsize=(9, 4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    cols = [("Inputs", ["mass, height", "drop/launch geometry", "surface compliance", "technique, 'work' factor"], "#d5d8dc", 0.1),
            ("Mechanism models", ["Bump: free fall + contact", "Power: lift + slam", "Aerial: projectile + spin", "Strike: collision"], "#aed6f1", 3.4),
            ("Outputs", ["peak force / g", "energy, contact time", "lumbar load, rotation", "head acceleration"], "#d7bde2", 6.7)]
    for t, items, c, x in cols:
        ax.add_patch(FancyBboxPatch((x, 0.3), 3.0, 3.3, boxstyle="round,pad=0.05", fc=c, ec="k", lw=.8))
        ax.text(x + 1.5, 3.3, t, ha="center", weight="bold")
        for i, it in enumerate(items): ax.text(x + 1.5, 2.6 - i * 0.65, it, ha="center", fontsize=8)
    for x in (3.15, 6.45): ax.annotate("", xy=(x + .25, 2), xytext=(x - .05, 2), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5, 0.05, "Monte Carlo (5,000 draws/move)  ->  Relative Load Index  ->  weight sensitivity", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig8_model_map.png")); plt.close(fig)

if __name__ == "__main__":
    df, mc, ranks = run()
    fig_ranking(df); fig_force_energy(df); fig_drop(df); fig_aerial(df)
    fig_lumbar(df); fig_strikes(df); fig_sens(df, ranks); fig_model_map()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df[["move", "category", "peak_G", "peak_force_kN", "impact_energy_J", "lumbar_kN", "rot_rate_rads", "head_accel_g", "head_accel_full_g", "RLI", "rank"]].round(2).sort_values("rank").to_string(index=False))
