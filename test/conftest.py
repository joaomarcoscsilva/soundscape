import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runveryslow", action="store_true", default=False, help="run very slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    veryslow = config.getoption("--runveryslow")
    slow = config.getoption("--runslow") or veryslow

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_veryslow = pytest.mark.skip(reason="need --runveryslow option to run")

    for item in items:
        if "veryslow" in item.keywords and not veryslow:
            item.add_marker(skip_veryslow)
        elif "slow" in item.keywords and not slow:
            item.add_marker(skip_slow)
