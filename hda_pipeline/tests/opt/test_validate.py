from hda.opt.validate import validate_candidate


def test_empty_ok():
    assert validate_candidate({})["ok"]


def test_generic_technique_ok():
    c = {"strategy": "Prefer iterative DFS over recursion. Use a priority queue."}
    assert validate_candidate(c)["ok"]


def test_task_name_rejected():
    assert not validate_candidate({"strategy": "Use Dijkstra for the distance."})["ok"]


def test_number_rejected():
    assert not validate_candidate({"iteration_note": "Try 1000 nodes."})["ok"]


def test_code_block_rejected():
    c = {"strategy": "```cpp\n" + "\n".join("int x;" for _ in range(20)) + "\n```"}
    assert not validate_candidate(c)["ok"]


def test_limit_rejected():
    assert not validate_candidate({"iteration_note": "x" * 801})["ok"]
