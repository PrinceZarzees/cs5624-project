from hda.quality import (normalized_quality, parse_nodes, validate_clique,
                         validate_independent_set, validate_vertex_cover)

# triangle 0-1-2 plus edge 2-3
N = 4
E = [(0, 1), (1, 2), (0, 2), (2, 3)]


def test_parse_nodes():
    assert parse_nodes("[1, 2, 3]") == [1, 2, 3]
    assert parse_nodes("1 2 3\n") == [1, 2, 3]
    assert parse_nodes("") is None


def test_clique():
    assert validate_clique([0, 1, 2], N, E)
    assert not validate_clique([0, 1, 3], N, E)
    assert not validate_clique([0, 0, 1], N, E)
    assert not validate_clique([9], N, E)


def test_independent_set():
    assert validate_independent_set([3], N, E)
    assert validate_independent_set([0, 3], N, E)
    assert not validate_independent_set([0, 1], N, E)


def test_vertex_cover():
    assert validate_vertex_cover([2, 1], N, E)
    assert validate_vertex_cover([0, 2], N, E)
    assert not validate_vertex_cover([0], N, E)


def test_quality():
    assert normalized_quality("max_clique", 3, 3) == 1.0
    assert normalized_quality("max_clique", 2, 4) == 0.5
    assert normalized_quality("min_vertex_cover", 2, 2) == 1.0
    assert normalized_quality("min_vertex_cover", 4, 2) == 0.5
    assert normalized_quality("max_clique", 0, 0) == 0.0
