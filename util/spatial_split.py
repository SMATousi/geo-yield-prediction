import json


DEFAULT_STATE_GROUPS = {
    '1': (27, 38, 46, 55),
    '2': (31, 20, 19, 29),
    '3': (17, 18, 39, 21),
}


def cross_location_split(data_file, group_indices=None, test_group='1',
                         test_year=2022, train_years=None):
    """Partition field-year samples into disjoint state groups to prevent
    spatial data leakage.

    Adapted from the cross-location evaluation protocol: counties are grouped
    into disjoint state groups; the model trains on the samples of the
    non-test groups (optionally restricted to the most recent ``train_years``),
    validates on the prior year of the held-out group, and tests on the
    held-out group for ``test_year``. This yields geographically separated
    train/valid/test sets so generalization to unseen regions and years can be
    measured without leakage.

    Args:
        data_file: path to the dataset json (list of dicts with ``state_ansi``
            and ``year`` keys).
        group_indices: dict mapping group name -> tuple of state ANSI codes.
        test_group: name of the held-out group.
        test_year: year on which the held-out group is evaluated.
        train_years: if given, only train on samples from the last
            ``train_years`` years before ``test_year``.

    Returns:
        (train_idx, valid_idx, test_idx): three lists of sample indices into
        the json list.
    """
    if group_indices is None:
        group_indices = DEFAULT_STATE_GROUPS

    test_states = set(group_indices[test_group])
    train_states = set()
    for name, states in group_indices.items():
        if name != test_group:
            train_states.update(states)

    data = json.load(open(data_file))

    train_idx, valid_idx, test_idx = [], [], []
    for i, obj in enumerate(data):
        state = int(obj['state_ansi'])
        year = int(obj['year'])
        if state in test_states:
            if year == test_year:
                test_idx.append(i)
            elif year == test_year - 1:
                valid_idx.append(i)
        elif state in train_states:
            if year < test_year:
                if train_years is None or year >= test_year - train_years:
                    train_idx.append(i)

    return train_idx, valid_idx, test_idx
