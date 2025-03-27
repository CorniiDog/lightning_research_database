
def zig_zag_range(max_value: int, start: int):
    """
    Generates an iterator over indices in a zig-zag order within the range [0, max_value).

    The iterator starts at the specified `start` index and then alternates between
    increasing and decreasing indices. The pattern is:
    
        start, start + d, start - d, start + 2*d, start - 2*d, ...

    The initial direction is chosen based on the available space toward the range
    boundaries. If the distance to the upper boundary (max_value - 1) is smaller than
    the distance to the lower boundary (0), the iterator will first move upward; otherwise,
    it will move downward.

    Parameters:
      max_value (int): The exclusive upper bound for indices (valid indices are 0 to max_value - 1).
      start (int): The starting index within the range.

    Returns:
      Iterator[int]: An iterator yielding indices in zig-zag order.

    Example:
      >>> list(zig_zag_range(10, 5))
      [5, 6, 4, 7, 3, 8, 2, 9, 1, 0]

      >>> list(zig_zag_range(10, 2))
      [2, 1, 3, 0, 4, 5, 6, 7, 8, 9]

      >>> list(zig_zag_range(0, 0))
      []
    """
    # Return an empty iterator if max_value is 0.
    if max_value == 0:
        return

    if start < 0 or start >= max_value:
        raise ValueError("start must be within the range [0, max_value)")

    yield start
    up_max = max_value - start - 1  # maximum upward steps possible
    down_max = start              # maximum downward steps possible
    max_d = max(up_max, down_max)

    # Determine the first direction: if the space upward is smaller, go up first; otherwise, go down.
    first = 'pos' if up_max < down_max else 'neg'

    for d in range(1, max_d + 1):
        if first == 'pos':
            if start + d < max_value:
                yield start + d
            if start - d >= 0:
                yield start - d
        else:
            if start - d >= 0:
                yield start - d
            if start + d < max_value:
                yield start + d

