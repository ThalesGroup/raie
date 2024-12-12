# -*- coding: utf-8 -*-

"""Console script for raie."""
import logging
from typing import List, Optional

import click

from raie import __version__

CLI_LOGGER = logging.getLogger(__name__)


@click.command()
@click.version_option(version=__version__)
def cli_main(arg: Optional[List] = None) -> int:
    """Console script for test_project_two."""
    click.echo("Replace this message by putting your code into test_project_two.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    CLI_LOGGER.info("Script has terminated successfully.")
    return 0
