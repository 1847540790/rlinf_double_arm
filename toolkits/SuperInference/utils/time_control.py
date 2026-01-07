#!/usr/bin/env python3
"""
Precise time control utilities for device timing.

Author: Jun Lv
"""

import time
from typing import Callable


def precise_sleep(dt: float, slack_time: float = 0.001, time_func: Callable[[], float] = time.monotonic) -> None:
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    
    Args:
        dt: Time to sleep in seconds
        slack_time: Time to spin instead of sleep (default: 0.001 seconds)
        time_func: Time function to use (default: time.monotonic)
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack_time: float = 0.001, time_func: Callable[[], float] = time.monotonic) -> None:
    """
    Wait until a specific end time using hybrid sleep/spin approach.
    
    Args:
        t_end: Target end time
        slack_time: Time to spin instead of sleep (default: 0.001 seconds)
        time_func: Time function to use (default: time.monotonic)
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def precise_loop_timing(update_interval: float, slack_time: float = 0.001, 
                        time_func: Callable[[], float] = time.monotonic) -> Callable[[], None]:
    """
    Create a precise loop timing function that maintains consistent intervals.
    
    Args:
        update_interval: Target interval between loop iterations in seconds
        slack_time: Time to spin instead of sleep (default: 0.001 seconds)
        time_func: Time function to use (default: time.monotonic)
    
    Returns:
        A function that should be called at the end of each loop iteration
    """
    start_time = time_func()
    
    def wait_for_next_iteration():
        nonlocal start_time
        elapsed = time_func() - start_time
        sleep_time = max(0, update_interval - elapsed)
        if sleep_time > 0:
            precise_sleep(sleep_time, slack_time, time_func)
        start_time = time_func()
    
    return wait_for_next_iteration 