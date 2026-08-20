import pytest

torch = pytest.importorskip("torch")

from openpoints.models.layers import (  # noqa: E402
    get_aggregation_feature_channels,
    get_aggregation_features,
    get_aggregation_feautres,
)
from openpoints.models.layers.local_aggregation import CHANNEL_MAP  # noqa: E402


BATCH, CHANNELS, NUM_POINTS, NUM_NEIGHBORS = 2, 8, 5, 4


@pytest.fixture
def grouped():
    torch.manual_seed(0)
    p = torch.randn(BATCH, NUM_POINTS, 3)
    dp = torch.randn(BATCH, 3, NUM_POINTS, NUM_NEIGHBORS)
    f = torch.randn(BATCH, CHANNELS, NUM_POINTS)
    fj = torch.randn(BATCH, CHANNELS, NUM_POINTS, NUM_NEIGHBORS)
    return p, dp, f, fj


@pytest.mark.parametrize('feature_type', ['dp_fj', 'dp_fj_df', 'pi_dp_fj_df', 'dp_df'])
def test_legacy_feature_types_are_unchanged(grouped, feature_type):
    """The four historically supported types must keep their exact output."""
    p, dp, f, fj = grouped

    if feature_type == 'dp_fj':
        expected = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        expected = torch.cat([dp, fj, fj - f.unsqueeze(-1)], 1)
    elif feature_type == 'pi_dp_fj_df':
        pi = p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, NUM_NEIGHBORS)
        expected = torch.cat([pi, dp, fj, fj - f.unsqueeze(-1)], 1)
    else:
        expected = torch.cat([dp, fj - f.unsqueeze(-1)], 1)

    assert torch.equal(get_aggregation_features(p, dp, f, fj, feature_type), expected)
    assert torch.equal(get_aggregation_feautres(p, dp, f, fj, feature_type), expected)


@pytest.mark.parametrize('feature_type', [
    'fj', 'dp', 'pi', 'pj', 'df', 'fi',
    'pi_dp', 'pj_dp', 'dp_fi_df', 'pj_dp_fj_df', 'pj_dp_df',
    'pi_pj_dp_fi_fj_df',
])
def test_arbitrary_component_combinations(grouped, feature_type):
    p, dp, f, fj = grouped
    out = get_aggregation_features(p, dp, f, fj, feature_type)

    assert out.shape[0] == BATCH
    assert out.shape[2:] == (NUM_POINTS, NUM_NEIGHBORS)
    assert out.shape[1] == get_aggregation_feature_channels(feature_type, CHANNELS)
    assert out.shape[1] == CHANNEL_MAP[feature_type](CHANNELS)


def test_components_are_concatenated_in_written_order(grouped):
    p, dp, f, fj = grouped
    assert torch.equal(
        get_aggregation_features(p, dp, f, fj, 'fj_dp'),
        torch.cat([fj, dp], 1),
    )


def test_pj_equals_pi_plus_dp(grouped):
    p, dp, f, fj = grouped
    pi = get_aggregation_features(p, dp, f, fj, 'pi')
    pj = get_aggregation_features(p, dp, f, fj, 'pj')
    assert torch.allclose(pj, pi + dp)


def test_explicit_channel_map_entries_are_preserved():
    assert CHANNEL_MAP['assa'](CHANNELS) == CHANNELS * 3
    assert CHANNEL_MAP['assa_dp'](CHANNELS) == CHANNELS * 3 + 3


def test_unknown_component_raises(grouped):
    p, dp, f, fj = grouped
    with pytest.raises(ValueError):
        get_aggregation_features(p, dp, f, fj, 'dp_bogus')
    with pytest.raises(ValueError):
        get_aggregation_feature_channels('', CHANNELS)


@pytest.mark.parametrize('feature_type', ['fi', 'df', 'dp_df'])
def test_missing_query_features_raise(grouped, feature_type):
    p, dp, _, fj = grouped
    with pytest.raises(ValueError):
        get_aggregation_features(p, dp, None, fj, feature_type)
