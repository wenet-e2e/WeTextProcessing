import pytest


@pytest.fixture(scope="session")
def shared_pipeline():
    """Builds each pipeline configuration once per session, in memory only.

    Cache publication and reloading are covered by the dedicated cache tests.
    """

    pipelines = {}

    def get(pipeline_type, **options):
        key = (pipeline_type, tuple(sorted(options.items())))
        if key not in pipelines:
            pipelines[key] = pipeline_type(cache_dir=False, **options)
        return pipelines[key]

    return get
