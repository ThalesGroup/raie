"""Test cases for the console module."""

import pytest
from click.testing import CliRunner

from raie import cli


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> int:
    """It exits with a status code of zero."""
    result = runner.invoke(cli.cli_main)
    assert result.exit_code == 0
    return 0
