"""Initialize child Adam state without changing parameter values or rendering.

Fresh children keep shared clocks and clear only newly allocated rows. Existing
slots, including reused split-parent slots, retain history. Legacy resets are
explicit ablations; unchanged live history is preserved by the fresh policy.
"""
from typing import Any
import warp as wp

CHILD_STATE_POLICIES = ('inherit', 'fresh_children', 'legacy_reset')


@wp.kernel
def copy_state_rows(values: wp.array(dtype=Any), parents: wp.array(dtype=wp.int32),
                    children: wp.array(dtype=wp.int32)):
    i = wp.tid()
    values[children[i]] = values[parents[i]]


@wp.kernel
def clear_state_rows(values: wp.array(dtype=Any), children: wp.array(dtype=wp.int32)):
    child = children[wp.tid()]
    # Valid optimizer histories are finite; retain scalar/vector dtype.
    values[child] = values[child] * 0.0


def inherit_adam_rows(optimizer, parents, children):
    """Historical parent-state policy; shared clock remains unchanged."""
    for values in (*optimizer.m, *optimizer.v):
        wp.launch(copy_state_rows, len(children), inputs=[values, parents, children], device=values.device)


def initialize_adam_rows(optimizer, parents, children, policy):
    if policy not in CHILD_STATE_POLICIES:
        raise ValueError('Unknown child optimizer-state policy.')
    if not len(children):
        return
    if policy == 'legacy_reset':
        optimizer.reset_internal_state()
    elif policy == 'inherit':
        inherit_adam_rows(optimizer, parents, children)
    else:
        for values in (*optimizer.m, *optimizer.v):
            wp.launch(clear_state_rows, len(children), inputs=[values, children], device=values.device)
