"""
Compare src/routing/baseline_iaco.py with the numbers in Cheng (2023).

    python scripts/verify_baseline_against_paper.py

Uses the paper's own network (Figure 3, Tables 5-7) and prints each result next 
to the paper's value.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.routing.baseline_iaco import IacoConfig, ahp_weights, iaco_search  # noqa: E402

# Load the paper's network from the test file (by path, since `tests` clashes
# with an installed package name).
_spec = importlib.util.spec_from_file_location("paper_fixture", REPO_ROOT / "tests/test_baseline_iaco.py")
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
PAPER_EDGES, PAPER_NAMES, paper_network = _fixture.PAPER_EDGES, _fixture.PAPER_NAMES, _fixture.paper_network


def row(label: str, ours: str, paper: str) -> None:
    print(f"  {label:<44} ours: {ours:<24} paper: {paper}")


def main() -> None:
    print("1. AHP (sec. 3.2, Table 3)")
    ahp = ahp_weights()
    row("weights w1 / w2 / w3", " / ".join(f"{w:.3f}" for w in ahp.weights), "0.637 / 0.258 / 0.105")
    row("lambda_max", f"{ahp.lambda_max:.4f}", "3.0385")
    row("CI", f"{ahp.ci:.4f}", "0.0193")
    row("CR (must be < 0.1)", f"{ahp.cr:.4f}", "0.0333")

    g, idx, s = paper_network()

    def totals(route):
        edges = g.edges_of([idx[x] for x in route])
        return g.distance[edges].sum(), s[edges].sum()

    print("\n2. Eq. 6 on the paper's network (Table 8)")
    for name, route, paper_len, paper_s in [
        ("Improved ACO path", "O-3-4-5-7-12-13-D", "3.015", "2.885"),
        ("Basic ACO path", "O-1-4-5-7-8-13-D", "2.902", "2.911"),
    ]:
        length, combined = totals(route.split("-"))
        row(f"{name} {route}: length", f"{length:.3f}", paper_len)
        row(f"{name} {route}: combined s", f"{combined:.3f}", paper_s)

    print("\n3. Every O -> D path in Figure 3, ranked by the paper's own cost s")
    adjacency = {}
    for u, v in PAPER_EDGES:
        adjacency.setdefault(u, []).append(v)

    def paths(u):
        if u == "D":
            yield ["D"]
            return
        for v in adjacency.get(u, []):
            for rest in paths(v):
                yield [u] + rest

    ranked = sorted(paths("O"), key=lambda p: totals(p)[1])
    for rank, p in enumerate(ranked, 1):
        length, combined = totals(p)
        note = "  <- paper's reported optimum" if "-".join(p) == "O-3-4-5-7-12-13-D" else ""
        print(f"  {rank:>2}. {'-'.join(p):<22} s = {combined:.3f}   length = {length:.3f}{note}")

    print("\n4. The implemented Improved ACO on that network (m = 23 ants, as in sec. 5.1.2)")
    for q0 in (0.7, 0.5, 0.2):
        found = {}
        for seed in range(10):
            r = iaco_search(g, idx["O"], idx["D"], s, np.ones(g.n_edges),
                            IacoConfig(n_ants=23, n_iterations=100, q0=q0, seed=seed), np.random.default_rng(seed))
            key = "-".join(PAPER_NAMES[i] for i in r.path)
            found.setdefault(key, []).append(r.cost)
        summary = ", ".join(f"{k} (s={v[0]:.3f}) x{len(v)}" for k, v in sorted(found.items(), key=lambda kv: -len(kv[1])))
        print(f"  q0 = {q0}: over 10 seeds -> {summary}")


if __name__ == "__main__":
    main()
