"""
The text prior: a year of traffic posts per road and hour, placed on the road graph by landmark
or road name, built only from posts before the first camera session.
"""

import re
from datetime import date, datetime

import numpy as np
import scipy.sparse as sp

from src.data import text_prior as tp
from src.data.graph_data import GraphData, edge_index_from_adjacency, normalize_adjacency
from src.data.node_context import FEATURES, FEATURES_V1, GROUPS


def placer():
    road_nodes = {"kalayaan": np.array([10, 11, 12]), "edsa": np.array([20, 21])}
    names = sorted(road_nodes, key=len, reverse=True)
    pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(names) + r")(?![a-z0-9])", re.I)
    return tp.RoadPlacer(road_nodes, pattern, {"EDSA-Ortigas": np.array([21])}, {"edsa ortigas": "EDSA-Ortigas"})


def test_a_landmark_places_a_post_more_precisely_than_its_road():
    p = placer()
    assert p.place("Stalled bus at EDSA Ortigas SB").tolist() == [21]                   # the intersection only
    assert p.place("TRAFFIC UPDATE: along Kalayaan Ave. from Makati Ave.").tolist() == [10, 11, 12]
    assert p.place("Heavy traffic on EDSA and Kalayaan").tolist() == [10, 11, 12, 20, 21]
    assert p.place("We are #7 with 54,713 votes").size == 0                              # no place: not used


def test_hour_buckets_separate_weekdays_from_weekends():
    assert tp.bucket(datetime(2026, 5, 25, 17, 30)) == 17          # Monday
    assert tp.bucket(datetime(2026, 5, 23, 17, 30)) == 41          # Saturday


def tiny_graph(node_ids):
    n = len(node_ids)
    adjacency = sp.csr_matrix((np.ones(n - 1), (range(n - 1), range(1, n))), shape=(n, n))
    return GraphData(node_ids=np.array(node_ids), adjacency=adjacency, a_hat=normalize_adjacency(adjacency),
                     edge_index=edge_index_from_adjacency(adjacency), camera_nodes={})


def test_the_prior_counts_reports_per_node_and_hour_and_uses_only_posts_before_the_cutoff(monkeypatch):
    before = date(2026, 5, 1)
    corpus = tp.PlacedCorpus(
        created=[datetime(2026, 4, 6, 17, 10), datetime(2026, 4, 13, 17, 40), datetime(2026, 4, 7, 8, 0)],
        nodes=[np.array([10, 11]), np.array([10]), np.array([12])],
        incident=np.array([True, False, False]),
        embeddings=np.random.default_rng(0).normal(size=(3, 16)).astype(np.float32),
        n_posts=5, before=before,
    )
    monkeypatch.setattr(tp, "placed_corpus", lambda *a, **k: corpus)
    prior = tp.build_text_prior(tiny_graph([10, 11, 12, 99]), before=before, components=0)
    reports = prior.values[:, :, 0]
    assert reports[0, 17] == 1.0                    # node 10 at 17:00 on a weekday: the most reports
    assert 0 < reports[1, 17] < 1.0                 # node 11: one report
    assert reports[2, 8] > 0 and reports[2, 17] == 0
    assert (prior.values[3] == 0).all()             # a node no post concerns
    assert prior.values[0, 17, 1] > 0 and prior.values[2, 8, 1] == 0      # incidents counted apart
    assert prior.n_posts_used == 3 and prior.describe()["share_used"] == 0.6


def test_text_prior_columns_are_appended_so_older_checkpoints_read_a_prefix():
    assert FEATURES[:len(FEATURES_V1)] == FEATURES_V1 and len(FEATURES_V1) == 17
    assert GROUPS["text_prior"] == tp.FEATURES and FEATURES[-len(tp.FEATURES):] == tp.FEATURES
