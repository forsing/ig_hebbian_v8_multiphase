from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Geometrija prostora kombinacija — v8 više cirkadijalnih faza

Ista Hebbian+RLM masa (mean_sK kao v6/v7), jedna faza φ₀=0:
  skor = mean_sK · L · circ(φ₀)
next: lokalna pretraga po skoru + kazna zbijenosti (bez sum/pos/odd → ne klon v2;
      bez sirovog top7 → ne Perez-blok oko zenita).

CSV: loto7_4652_k57.csv, seed=39.
Ime: ig_hebbian_v8_multiphase.py
"""

import csv
from itertools import combinations
from math import cos, exp, pi
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35
K_RLM = 5
EPS_EXC = 0.08
DELTA_C = 1e-3
EPS_OT = 0.08
SINKHORN_ITERS = 80
FOLLOW_THR = 0.25
ALPHA_C = 2.0
BETA_S = 0.5
WARMUP = 500
STEP = 50
ZENITH = 20.0
PEREZ_A = 4.0
PEREZ_B = 0.6
PEREZ_C = 1.2
PEREZ_D = 2.5
CIRC_PERIOD = 39
CIRC_KAPPA = 0.25
N_PHASES = 2
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws, lam=LAMBDA_TEMP):
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def hebbian_add_draw(W, prev, cur, lam=LAMBDA_TEMP):
    idx = [int(x) - 1 for x in cur.tolist()]
    for a, b in combinations(idx, 2):
        W[a, b] += 1.0
        W[b, a] += 1.0
    a_idx = [int(x) - 1 for x in prev.tolist()]
    for a in a_idx:
        for b in idx:
            if a == b:
                continue
            W[a, b] += lam
            W[b, a] += lam
    np.fill_diagonal(W, 0.0)


def energy_distribution(W):
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    return D / row


def hebbian_mass(D, last):
    idx = [int(x) - 1 for x in last.tolist()]
    return D[idx].mean(axis=0)


def learned_cost(W):
    return 1.0 / (W + DELTA_C)


def sinkhorn(a, b, C):
    K = np.exp(-C / EPS_OT)
    u = np.ones(FRONT_N)
    v = np.ones(FRONT_N)
    for _ in range(SINKHORN_ITERS):
        u = a / np.clip(K @ v, 1e-18, None)
        v = b / np.clip(K.T @ u, 1e-18, None)
    return (u[:, None] * K) * v[None, :]


def net_flow(pi):
    return pi.sum(axis=0) - pi.sum(axis=1)


def excite(s, rng):
    noise = rng.random(FRONT_N) * np.sqrt(np.clip(s, 1e-12, None))
    e = s + noise
    return e / e.sum()


def rlm_walk(D, last, seed=SEED):
    rng = np.random.default_rng(seed)
    s = np.zeros(FRONT_N)
    for x in last.tolist():
        s[int(x) - 1] = 1.0 / FRONT_SELECT
    for _ in range(K_RLM):
        s = s @ D
        s = np.clip(s, 0.0, None)
        s = s / s.sum() if s.sum() > 0 else np.ones(FRONT_N) / FRONT_N
        s = (1.0 - EPS_EXC) * s + EPS_EXC * excite(s, rng)
        s = s / s.sum()
    return s


def follow_score(flow, delta_s):
    a = flow - flow.mean()
    b = delta_s - delta_s.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def draw_mass(draw):
    m = np.zeros(FRONT_N)
    for x in draw.tolist():
        m[int(x) - 1] = 1.0 / FRONT_SELECT
    return m


def upgrade_path(C, s_prev, s_k, flow):
    residual = (s_k - s_prev) - flow
    boost = np.maximum(residual, 0.0)
    C2 = C / (1.0 + ALPHA_C * boost[None, :])
    s2 = np.clip(s_k + BETA_S * residual, 0.0, None)
    s2 = s2 / s2.sum() if s2.sum() > 0 else s_k.copy()
    flow2 = net_flow(sinkhorn(s_prev, s2, C2))
    return s2, flow2, follow_score(flow2, s2 - s_prev)


def path_at(W, last, seed):
    D = energy_distribution(W)
    s_prev = draw_mass(last)
    s_k = rlm_walk(D, last, seed=seed)
    C = learned_cost(W)
    flow = net_flow(sinkhorn(s_prev, s_k, C))
    follow = follow_score(flow, s_k - s_prev)
    if follow < FOLLOW_THR:
        s_k, flow, follow = upgrade_path(C, s_prev, s_k, flow)
    return s_k, flow, follow


def perez_luminance(sun):
    L = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        n = i + 1
        gamma = abs(n - sun) / float(FRONT_N)
        theta = abs(n - ZENITH) / float(FRONT_N)
        L[i] = (1.0 + PEREZ_C * exp(-PEREZ_A * gamma * gamma)) * (
            1.0 + PEREZ_B * exp(-PEREZ_D * theta * theta)
        )
    return L / L.sum()


def circadian_field(t_index, phi0):
    phi = 2.0 * pi * (t_index % CIRC_PERIOD) / float(CIRC_PERIOD) + phi0
    circ = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        circ[i] = 1.0 + CIRC_KAPPA * cos(phi + 2.0 * pi * i / float(FRONT_N))
    return circ


def mean_sk_walk(draws):
    T = len(draws)
    W = hebbian_weights(draws[:WARMUP])
    acc = np.zeros(FRONT_N)
    t = WARMUP
    while t < T - 1:
        if (t - WARMUP) % STEP == 0:
            s_k, _, _ = path_at(W, draws[t - 1], seed=SEED + t)
            acc += s_k
        hebbian_add_draw(W, draws[t - 1], draws[t])
        t += 1
    if acc.sum() <= 0:
        return np.ones(FRONT_N) / FRONT_N
    return acc / acc.sum()


def number_scores(mean_sk, L, circ, ban):
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(mean_sk[i] * L[i] * circ[i])
    return out


def _combo_fit(combo, score, ban):
    """Energija skora; kazna zbijenih rupa (anti Perez-blok). Bez istorijskog sum/pos/odd."""
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    gaps = [nums[i + 1] - nums[i] for i in range(FRONT_SELECT - 1)]
    s -= 0.25 * sum(1.0 / g for g in gaps)
    s += 0.01 * (nums[-1] - nums[0])
    return s


def predict_next(score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_v8(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    mean_sk = mean_sk_walk(draws)
    sun = float(np.mean(last))
    L = perez_luminance(sun)
    t_now = len(draws) - 1

    phases = [2.0 * pi * k / max(N_PHASES, 1) for k in range(N_PHASES)]
    combos = []
    for k, phi0 in enumerate(phases):
        circ = circadian_field(t_now, phi0)
        score = number_scores(mean_sk, L, circ, ban)
        combo = predict_next(score, ban)
        combos.append({"phase": k, "next": combo})

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | N_PHASES={N_PHASES} | ig_hebbian_v8")
    print(f"last: {last.tolist()}")
    print()
    print("=== next po fazi ===")
    for c in combos:
        print(f"  φ{c['phase']:02d}: {c['next']}")
    print()
    print("=== next ===")
    for c in combos:
        print("next:", c["next"])


if __name__ == "__main__":
    run_v8()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | N_PHASES=2 | ig_hebbian_v8
last: [7, 8, 14, 15, 17, 23, 32]

=== next po fazi ===
  φ00: [2, 13, 16, 19, 31, 33, 35]
  φ01: [3, 5, 6, 20, 22, 24, 39]

=== next ===
next: [2, 13, 16, 19, 31, 33, 35]
next: [3, 5, 6, 20, 22, 24, 39]
"""



"""
1 cirkadijalna faza → 1 next

mean_sK · L · circ + fit sa kaznom zbijenosti
skor + lokalna pretraga, kazna zbijenosti (bez sum/pos/odd).
"""



"""
Sledeći mehanizam (v8): više cirkadijalnih faza (φ₀ mreža) × ista geometrija — više putanja / više kombinacija, ne novi „šum model“.
"""



"""
Hebbian linija (geometrija → Perez → RLM → OT → follow → walk → kalibracija) je zatvorena. 
"""
