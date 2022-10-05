from soundscape import composition


def test_Composable_simple():
    @composition.Composable
    def f(x):
        return x + 1

    @composition.Composable
    def g(x):
        return x * 2

    fn = f | g

    assert fn(1) == 4
    assert fn(2) == 6
    assert fn(0.5) == 3


def test_Composable_dict():
    @composition.Composable
    def f(values):
        return {**values, "y": values["x"] + 1}

    @composition.Composable
    def g(values):
        return {**values, "z": values["y"] * 2}

    def h(values):
        return values["z"] - 1

    h = composition.Composable(h, key="h")

    fn = f | g | h

    assert fn({"x": 1}) == {"x": 1, "y": 2, "z": 4, "h": 3}
    assert fn({"x": 2}) == {"x": 2, "y": 3, "z": 6, "h": 5}
    assert fn({"x": 0.5}) == {"x": 0.5, "y": 1.5, "z": 3, "h": 2}


def test_Composable_idempotent():
    @composition.Composable
    def f(x):
        return x + 1

    f2 = composition.Composable(f)

    assert f(1) == f2(1)
    assert f(2) == f2(2)
    assert f(0.5) == f2(0.5)


def test_hashable_dict():
    d1 = {"a": 1, "b": 2}
    d1_copy = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 3}

    assert hash(composition.hashable_dict(d1)) == hash(
        composition.hashable_dict(d1_copy)
    )
    assert hash(composition.hashable_dict(d1)) != hash(composition.hashable_dict(d2))


def test_split_dict():
    d = {"a": 1, "b": 2, "c": 3, "d": 4}
    d1, d2 = composition.split_dict(d, ["a", "c"])
    assert d1 == {"b": 2, "d": 4}
    assert d2 == {"a": 1, "c": 3}


def test_jit():
    @composition.Composable
    def f(values):
        st = values["static"]
        x = values["x"]
        return {**values, "y": x + st}

    jitted_f = composition.jit(f, static_keys=["static"], ignored_keys=["ignored"])

    values = {"x": 1, "static": 2, "ignored": 3, "_underlined": 4}

    assert (
        f(values)
        == jitted_f(values)
        == {"x": 1, "y": 3, "static": 2, "ignored": 3, "_underlined": 4}
    )


def test_grad():
    @composition.Composable
    def f(values):
        par = values["params"]
        x = values["x"]
        return {**values, "y": par * x}

    grad_f = composition.grad(f, input_key="params", output_key="y")

    values = {"params": 2.0, "x": 3.0}
    assert grad_f(values) == {"params": 2.0, "x": 3.0, "y": 6.0, "grad": 3.0}


def test_identity():
    @composition.Composable
    def f(values):
        return {**values, "y": values["x"] + 1}

    fn = f | composition.identity

    values = {"x": 1}
    assert fn(values) == f(values) == {"x": 1, "y": 2}
