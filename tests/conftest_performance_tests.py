# The MIT License (MIT)
#
# Copyright (c) 2018-2026 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Performance testing framework infrastructure."""

import time
import warnings
from typing import Callable

import pytest

PERFORMANCE_LOG: dict[str, dict[str, float]] = {}


@pytest.fixture(scope="module")
def evaluate_execution_time() -> Callable:
    """Return function to evaluate execution time.

    Necessary to enable the function call through pytest fixtures.

    Returns:
        Function to evaluate execution time.
    """

    def _evaluate_execution_time(
        name: str,
        function: Callable,
        *,
        args: tuple = (),
        kwargs: dict = {},
        expected_time: float,
    ):
        """Evaluate the execution time of a function and log it.

        Args:
            name: Name of the test for logging.
            function: The function to be executed.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            expected_time: Expected execution time in seconds.
        """
        start_time = time.time()
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start_time

        PERFORMANCE_LOG[name] = {
            "execution_time": elapsed_time,
            "expected_time": expected_time,
        }

        if elapsed_time > expected_time:
            warnings.warn(
                f"Execution time for '{name}' test exceeded expected time: "
                f"{elapsed_time:.2f}s > {expected_time:.2f}s"
            )

        return result

    return _evaluate_execution_time


def sessionfinish_performance_tests(session):
    """Exit with exit code 1 if any performance test failed.

    This is utilized to ensure that the Github Actions workflow fails if performance
    tests exceed their expected execution time.
    """
    if PERFORMANCE_LOG:
        for data in PERFORMANCE_LOG.values():
            if data["execution_time"] > data["expected_time"]:
                session.exitstatus = 1
                break


def terminal_summary_performance_tests(terminalreporter):
    """Print a summary of performance tests at the end of the pytest run."""
    if PERFORMANCE_LOG:
        terminalreporter.write_sep("=", "Performance Test Summary")

        for name, data in PERFORMANCE_LOG.items():
            elapsed = data["execution_time"]
            expected = data["expected_time"]

            if data["execution_time"] < data["expected_time"]:
                terminalreporter.write_line(
                    f"{name:<70} {elapsed:.3f}s < {expected:.3f}s",
                    bold=False,
                    green=True,
                )
            else:
                terminalreporter.write_line(
                    f"{name:<70} {elapsed:.3f}s > {expected:.3f}s",
                    bold=True,
                    red=True,
                )
