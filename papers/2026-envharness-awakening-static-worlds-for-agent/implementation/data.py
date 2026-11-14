"""Task pools for the EnvHarness miniature (data.py role).

Paper: "EnvHarness: Awakening Static Worlds for Agent Learning"
       Huang et al., 2026. arXiv:2608.19880. https://arxiv.org/abs/2608.19880

Two task families on a 7x7 KeyGrid, start (0,0), goal (6,6):

* OPEN  - wall-less, both keys within ~3 of the start, generous budget.
          Leaps (J*) strictly dominate, so a tabular policy converges to
          "jump everywhere" (the observable over-reliance EnvRigger targets).
* HARD  - one key far away in the open, the other hidden at the end of an
          L-shaped 1-wide pocket (3 cells: mouth -> axial -> bend). Inside /
          around the pocket every 2-cell leap is illegal (walls at mid or
          target), so the key can only be fetched with plain moves. Tight
          budget: jump-spam that wastes steps times out.

The train pool is intentionally bimodal (papers' "static environment"
pathology): most episodes land on already-mastered OPEN tasks or on
HARD tasks a random-ish explorer never solves - little signal in the
"solvable but challenging" band in between.

Stdlib only.
"""

import random

from model import bfs_dist, bfs_path, optimal_len_jump_aware

SIZE = 7
START = (0, 0)
GOAL = (6, 6)


def _base(name, family):
    return {"name": name, "family": family, "size": SIZE, "start": START, "goal": GOAL,
            "walls": [], "keys": [], "budget": 0}


def gen_open(rng: random.Random, idx: int) -> dict:
    task = _base(f"open{idx}", "open")
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)
             if (r, c) not in (START, GOAL)]
    near = [c for c in cells if abs(c[0] - START[0]) + abs(c[1] - START[1]) <= 3]
    keys = rng.sample(near, 2)
    task["keys"] = keys
    # budget: generous for the leap-optimal tour, but still completable with
    # plain moves (leaps must be a preference, not a requirement)
    st = {"size": SIZE, "walls": set()}
    dist = bfs_dist(st, START)
    move_len, cur = 0, START
    for k in sorted(keys, key=lambda k: dist[k]):
        move_len += dist[k] if cur == START else abs(k[0] - cur[0]) + abs(k[1] - cur[1])
        cur = k
    move_len += abs(GOAL[0] - cur[0]) + abs(GOAL[1] - cur[1])
    task["budget"] = max(16, move_len + 4)
    return task


def _place_pocket(rng: random.Random, occupied):
    """L-shaped 3-cell pocket: mouth m (open side), axial a=m+d1, bend b=a+d2
    (d2 perpendicular to d1). All 8 neighbours of a/b except the pocket line
    and the mouth opening become walls, so no 2-cell leap enters or crosses
    the pocket. Returns (mouth, axial, bend, walls) or None."""
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    perp = {(-1, 0): [(0, 1), (0, -1)], (1, 0): [(0, 1), (0, -1)],
            (0, 1): [(-1, 0), (1, 0)], (0, -1): [(-1, 0), (1, 0)]}
    for _ in range(200):
        d1 = rng.choice(dirs)
        m = (rng.randrange(SIZE), rng.randrange(SIZE))
        a = (m[0] + d1[0], m[1] + d1[1])
        if not (0 <= a[0] < SIZE and 0 <= a[1] < SIZE):
            continue
        d2 = rng.choice(perp[d1])
        b = (a[0] + d2[0], a[1] + d2[1])
        if not (0 <= b[0] < SIZE and 0 <= b[1] < SIZE):
            continue
        if {m, a, b} & set(occupied):  # must not cover start/goal/keys
            continue
        # mouth should sit away from the start so the pocket is a real detour
        if abs(m[0] - START[0]) + abs(m[1] - START[1]) < 4:
            continue
        pocket = [m, a, b]
        walls = set()
        for cell in (a, b):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (cell[0] + dr, cell[1] + dc)
                    if nb in pocket or nb == m:
                        continue
                    if 0 <= nb[0] < SIZE and 0 <= nb[1] < SIZE:
                        walls.add(nb)
        # pocket must be reachable from the mouth with moves, and the mouth
        # must have at least one open neighbour outside the pocket
        outside = [nb for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]
                   for nb in [(m[0] + dr, m[1] + dc)]
                   if 0 <= nb[0] < SIZE and 0 <= nb[1] < SIZE
                   and nb not in walls and nb not in pocket]
        if not outside:
            continue
        return m, a, b, walls
    return None


def gen_hard(rng: random.Random, idx: int) -> dict:
    """Pocket-key task: ONE key hidden at the end of an L-shaped 1-wide
    pocket. Single key on purpose: with two keys the 'nearest-key' signal
    flips at the watershed between them and a locally reactive policy
    oscillates between orderings (verified empirically); one key isolates
    the actual challenge - move-only pocket traversal under a tight budget."""
    for _ in range(300):
        task = _base(f"hard{idx}", "hard")
        placed = _place_pocket(rng, {START, GOAL})
        if placed is None:
            continue
        mouth, axial, bend, walls = placed
        task["walls"] = sorted(walls)
        task["keys"] = [bend]
        task["budget"] = 0  # set below from the optimal length
        opt = optimal_len_jump_aware(task)
        if opt is None or opt > 16:
            continue
        task["budget"] = opt + 8  # tight-ish: leap-spam at the pocket walls wastes it
        return task
    raise RuntimeError("could not generate a hard task")


def make_pools(seed: int, n_train_per_family: int = 16, n_eval_per_family: int = 10):
    rng = random.Random(seed * 1000 + 7)
    train, eval_ = [], []
    for i in range(n_train_per_family):
        train.append(gen_open(rng, i))
        train.append(gen_hard(rng, i))
    for i in range(n_eval_per_family):
        eval_.append(gen_open(rng, 100 + i))
        eval_.append(gen_hard(rng, 100 + i))
    return train, eval_


def verify_pool(pool):
    """Sanity: every task solvable inside budget (verifier-reachable)."""
    bad = []
    for t in pool:
        opt = optimal_len_jump_aware(t)
        if opt is None or opt > t["budget"]:
            bad.append((t["name"], opt, t["budget"]))
    return bad


if __name__ == "__main__":
    tr, ev = make_pools(0)
    print(f"train={len(tr)} eval={len(ev)}")
    print("unsolvable/over-budget:", verify_pool(tr) + verify_pool(ev))
    for t in tr[:2] + tr[-2:]:
        opt = optimal_len_jump_aware(t)
        st = {"size": SIZE, "walls": set(map(tuple, t["walls"]))}
        mv = bfs_dist(st, START)
        move_opt = None
        print(f"{t['name']}: budget={t['budget']} jump-optimal={opt} "
              f"walls={len(t['walls'])} keys={t['keys']}")
