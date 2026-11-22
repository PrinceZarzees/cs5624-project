from hda.opt.run import better, extract_json, extract_tag


def test_extract_json_and_tag():
    assert extract_json('x {"a": 1}')["a"] == 1
    assert extract_tag("<candidate>{}</candidate>", "candidate") == "{}"


def test_better_rule():
    assert better(0.55, 0.50, "a", "bb")
    assert not better(0.51, 0.50, "a"*10, "b")
    assert better(0.51, 0.50, "a", "bbbb")
