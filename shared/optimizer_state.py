"""Inherit Adam trajectories when a live splat is cloned or split.

Moments and the optimizer's existing clock travel together: new children inherit
the parent's trajectory rather than zero moments with a mature global clock.
Sources must be live slots and destinations distinct free slots.
"""

from typing import Any

import warp as wp


@wp.kernel
def copy_state_rows(values: wp.array(dtype=Any), parents: wp.array(dtype=wp.int32),
                    children: wp.array(dtype=wp.int32)):
    i = wp.tid()
    values[children[i]] = values[parents[i]]


def inherit_adam_rows(optimizer, parents, children):
    """Copy first/second moments; leave the shared Adam clock unchanged."""
    for values in (*optimizer.m, *optimizer.v):
        wp.launch(copy_state_rows, len(children), inputs=[values, parents, children], device=values.device)
