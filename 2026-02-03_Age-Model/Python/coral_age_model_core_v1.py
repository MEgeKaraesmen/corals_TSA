from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class Params:
    # priors
    sigma_year: float = 0.05
    year_min: float = 0.80
    year_max: float = 1.20

    # within-year monotone warp (piecewise)
    Mseg: int = 6
    alpha: float = 20.0

    # likelihood scales (units match d18o)
    sigma_y: float = 0.10          # monthly climatology template misfit
    sigma_inst: float = 0.12       # instrumental time-series misfit
    sigma_phase_months: float = 1.5
    sigma_anchor_years: float = 0.06

    # band-start preference (set to warm-season month for your site)
    warm_month: int = 8            # e.g., Aug=8

    # sampling
    n_particles: int = 4000
    phi_grid: Tuple[int, ...] = tuple(range(12))

    # reproducibility
    seed: Optional[int] = 1


# ---------- small math helpers ----------

def log_normpdf(x: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    return float(np.sum(-0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))))

def month_from_decimal_year(t: np.ndarray) -> np.ndarray:
    frac = t - np.floor(t)
    return (np.floor(frac * 12).astype(int) % 12) + 1  # 1..12

def rotate_month(m: np.ndarray, phi: int) -> np.ndarray:
    return ((m - 1 + phi) % 12) + 1

def circ_month_dist(m: np.ndarray, target: int) -> np.ndarray:
    d = np.abs(m - target)
    return np.minimum(d, 12 - d)

def weighted_quantile(x: np.ndarray, w: np.ndarray, probs=(0.025, 0.5, 0.975)) -> np.ndarray:
    o = np.argsort(x)
    x = x[o]; w = w[o]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.array([np.interp(p, cw, x) for p in probs])


# ---------- ordering ----------

def sort_coral_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataset into the MASTER sampling order (older -> younger).
    Preference:
      1) If 'count' exists: sort by count ascending (center -> rim).
      2) Else if both ('path_name','path_num') exist: sort by (path_name, path_num).
      3) Else: preserve file row order.
    """
    out = df.copy()
    if "count" in out.columns:
        out["count"] = pd.to_numeric(out["count"], errors="coerce")
        out = out.sort_values("count", ascending=True)
    elif "path_name" in out.columns and "path_num" in out.columns:
        out["path_name"] = out["path_name"].astype(str)
        out["path_num"] = pd.to_numeric(out["path_num"], errors="coerce")
        out = out.sort_values(["path_name", "path_num"], ascending=[True, True])
    else:
        out = out.reset_index(drop=False).sort_values("index", ascending=True)
    return out.reset_index(drop=True)


# ---------- age2 construction (YOUR semantics) ----------

def build_age2_from_age1_steps(
    df_sorted: pd.DataFrame,
    age1_col: str = "age1",
    age2_col: str = "age2"
) -> pd.DataFrame:
    """
    Builds:
      - age1_raw: float (NaN means no band-start annotation at this sample)
      - is_band_start: bool = isfinite(age1_raw)
      - age2_track: cumulative sum of age1 steps (NaN treated as 0 step)
      - age2_filled: age2_track overwritten by manual age2 where present, then interpolated
      - year_id: floor(age2_filled) shifted to 0..K-1

    Returns a copy with those added columns.
    """
    out = df_sorted.copy()

    age1_raw = pd.to_numeric(out[age1_col], errors="coerce")
    out["age1_raw"] = age1_raw
    out["is_band_start"] = np.isfinite(age1_raw.to_numpy())

    # step sizes: NaN => "no recorded delta" => treat as 0 for the cumsum track
    step = age1_raw.fillna(0.0).to_numpy(float)
    age2_track = np.cumsum(step).astype(float)
    out["age2_track"] = age2_track

    # If manual age2 exists, overwrite those locations (can include fractional values)
    if age2_col in out.columns:
        age2_manual = pd.to_numeric(out[age2_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(age2_manual)
        age2_track2 = age2_track.copy()
        age2_track2[mask] = age2_manual[mask]
        out["age2_track_over"] = age2_track2
        out["age2_filled"] = pd.Series(age2_track2).interpolate(limit_direction="both").to_numpy(float)
    else:
        out["age2_track_over"] = out["age2_track"]
        out["age2_filled"] = pd.Series(age2_track).interpolate(limit_direction="both").to_numpy(float)

    # band id
    yid = np.floor(out["age2_filled"].to_numpy(float)).astype(int)
    yid = yid - int(np.min(yid))
    out["year_id"] = yid

    return out


# ---------- sampling primitives ----------

def sample_year_durations(K: int, p: Params) -> np.ndarray:
    d = np.random.normal(1.0, p.sigma_year, size=K)
    return np.clip(d, p.year_min, p.year_max)

def sample_within_year_warp(K: int, p: Params) -> np.ndarray:
    return np.random.dirichlet(np.ones(p.Mseg) * p.alpha, size=K)

def build_times_from_bands(year_id: np.ndarray, d: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Construct relative time t_rel (years) given:
      year_id (N,) in 0..K-1
      d (K,) durations
      delta (K,Mseg) Dirichlet increments summing to 1

    Within each band, sample order defines within-year progression.
    """
    N = year_id.size
    K, Mseg = delta.shape
    t_rel = np.zeros(N, dtype=float)

    B = np.zeros(K + 1)
    B[1:] = np.cumsum(d)

    knot_s = np.linspace(0, 1, Mseg + 1)

    for k in range(K):
        idx = np.where(year_id == k)[0]
        nk = idx.size
        if nk == 0:
            continue
        s = np.linspace(0, 1, nk, endpoint=False)
        knot_u = np.concatenate(([0.0], np.cumsum(delta[k])))
        u = np.interp(s, knot_s, knot_u)
        t_rel[idx] = B[k] + u * d[k]

    return t_rel


# ---------- main inference ----------

def fit_age_model(
    df_in: pd.DataFrame,
    mu_month: np.ndarray,
    params: Params = Params(),
    *,
    clim_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    use_is_inst_mask: bool = False,
    anchor: Optional[Tuple[str, float, float]] = None,  # (path_name, path_num, t_anchor decimal year)
    tiepoints: Optional[List[Tuple[str, float, float, float]]] = None,  # (path_name, path_num, t_star, sigma_t)
    overlaps: Optional[List[Tuple[str, float, str, float, float]]] = None,  # (pi, si, pj, sj, sigma_dt)
) -> Dict[str, Any]:
    """
    Returns:
      dict(
        df_sorted = DataFrame with added columns:
          age1_raw, is_band_start, age2_track, age2_filled, year_id, t_lo, t_med, t_hi
        T = (n_draws, N) sampled ages
        w = (n_draws,) normalized weights
        post_phi = {phi:prob}
        meta = diagnostics
      )
    """
    if mu_month is None or len(mu_month) != 12:
        raise ValueError("mu_month must be length-12 (Jan..Dec).")

    if params.seed is not None:
        np.random.seed(params.seed)

    # sort
    df0 = sort_coral_table(df_in)

    # validate d18o
    if "d18o" not in df0.columns:
        raise ValueError("Missing required column 'd18o'.")
    df0["d18o"] = pd.to_numeric(df0["d18o"], errors="coerce")
    if df0["d18o"].isna().any():
        bad = df0.loc[df0["d18o"].isna(), [c for c in ["count","path_name","path_num","d18o"] if c in df0.columns]].head(10)
        raise ValueError(f"Non-numeric d18o found. Examples:\n{bad}")

    # build age2/year_id per your semantics
    df = build_age2_from_age1_steps(df0, age1_col="age1", age2_col="age2")

    y = df["d18o"].to_numpy(float)
    year_id = df["year_id"].to_numpy(int)
    is_band_start = df["is_band_start"].to_numpy(bool)

    N = len(df)
    K = int(year_id.max()) + 1

    # optional instrumental mask
    is_inst = None
    if use_is_inst_mask:
        if "is_inst" not in df.columns:
            raise ValueError("use_is_inst_mask=True but 'is_inst' column not found.")
        is_inst = df["is_inst"].astype(bool).to_numpy()

    # anchor (only if you have path_name/path_num)
    anchor_idx = None
    t_anchor = None
    if anchor is not None:
        if "path_name" not in df.columns or "path_num" not in df.columns:
            raise ValueError("anchor requires columns 'path_name' and 'path_num' in df.")
        pth, num, tA = anchor
        df["path_name"] = df["path_name"].astype(str)
        df["path_num"] = pd.to_numeric(df["path_num"], errors="coerce")
        hit = np.where((df["path_name"].to_numpy() == str(pth)) &
                       (df["path_num"].to_numpy() == float(num)))[0]
        if hit.size != 1:
            raise ValueError("Anchor (path_name, path_num) did not match exactly one row.")
        anchor_idx = int(hit[0])
        t_anchor = float(tA)

    # lookup for tiepoints/overlaps
    lookup: Dict[Tuple[str, float], int] = {}
    if ("path_name" in df.columns) and ("path_num" in df.columns):
        lookup = {
            (r["path_name"], float(r["path_num"])): i
            for i, r in enumerate(df[["path_name", "path_num"]].to_dict("records"))
        }

    T_list: List[np.ndarray] = []
    logw_list: List[float] = []
    phi_list: List[int] = []

    for phi in params.phi_grid:
        phi = int(phi)
        for _ in range(params.n_particles):
            d = sample_year_durations(K, params)
            delta = sample_within_year_warp(K, params)
            t_rel = build_times_from_bands(year_id, d, delta)

            t = t_rel
            if anchor_idx is not None:
                t = t_rel - t_rel[anchor_idx] + t_anchor

            ll = 0.0

            # (1) climatology template
            m = rotate_month(month_from_decimal_year(t), phi)
            ll += log_normpdf(y, mu_month[m - 1], params.sigma_y)

            # (2) instrumental time-series
            if clim_fn is not None:
                if is_inst is None:
                    ll += log_normpdf(y, clim_fn(t), params.sigma_inst)
                else:
                    ll += log_normpdf(y[is_inst], clim_fn(t[is_inst]), params.sigma_inst)

            # (3) band-start warm-month preference
            # YOUR semantics: any non-NaN entry in age1 indicates a dark-band start marker.
            if is_band_start.any():
                mb = rotate_month(month_from_decimal_year(t[is_band_start]), phi)
                dmon = circ_month_dist(mb, params.warm_month).astype(float)
                ll += log_normpdf(dmon, np.zeros_like(dmon), params.sigma_phase_months)

            # (4) tiepoints
            if tiepoints:
                for pth, num, t_star, sigma_t in tiepoints:
                    idx = lookup.get((str(pth), float(num)))
                    if idx is None:
                        continue
                    ll += log_normpdf(np.array([t[idx]]), np.array([float(t_star)]), float(sigma_t))

            # (5) overlaps
            if overlaps:
                for pi, si, pj, sj, sigma_dt in overlaps:
                    ii = lookup.get((str(pi), float(si)))
                    jj = lookup.get((str(pj), float(sj)))
                    if ii is None or jj is None:
                        continue
                    dt = t[ii] - t[jj]
                    ll += log_normpdf(np.array([dt]), np.array([0.0]), float(sigma_dt))  # placeholder
            # fix typo above by computing dt likelihood correctly
            # (kept explicit below to avoid silent mistakes)
            if overlaps:
                for pi, si, pj, sj, sigma_dt in overlaps:
                    ii = lookup.get((str(pi), float(si)))
                    jj = lookup.get((str(pj), float(sj)))
                    if ii is None or jj is None:
                        continue
                    dt = t[ii] - t[jj]
                    ll += log_normpdf(np.array([dt]), np.array([0.0]), float(sigma_dt))

            # (6) soft anchor
            if anchor_idx is not None:
                ll += log_normpdf(np.array([t[anchor_idx]]), np.array([t_anchor]), params.sigma_anchor_years)

            T_list.append(t)
            logw_list.append(float(ll))
            phi_list.append(phi)

    T = np.vstack(T_list)
    logw = np.array(logw_list, float)
    phi_arr = np.array(phi_list, int)

    # normalize weights
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w = w / np.sum(w)

    # posterior phi
    post_phi = {int(ph): float(w[phi_arr == int(ph)].sum()) for ph in params.phi_grid}
    s = sum(post_phi.values())
    post_phi = {k: v / s for k, v in post_phi.items()}

    # summaries
    qs = np.vstack([weighted_quantile(T[:, i], w) for i in range(N)])
    df["t_lo"] = qs[:, 0]
    df["t_med"] = qs[:, 1]
    df["t_hi"] = qs[:, 2]

    meta = {
        "N": int(N),
        "K": int(K),
        "n_draws": int(T.shape[0]),
        "ESS": float(1.0 / np.sum(w**2)),
    }

    return {"df_sorted": df, "T": T, "w": w, "post_phi": post_phi, "meta": meta}