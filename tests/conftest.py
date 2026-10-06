import pytest
from disco_geo.data.procedural import generate


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("objects")
    generate(root, scenes=20, views=6, size=16, seed=3)
    return root
