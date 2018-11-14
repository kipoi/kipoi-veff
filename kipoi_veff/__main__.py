import pkg_resources
import logging.config
import kipoi_veff.cli
import sys
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


def cli_main(command, raw_args):
    kipoi_veff.cli.cli_main(command, raw_args)

if __name__ == '__main__':
    command = sys.argv[1]
    raw_args = sys.argv[1:]
    cli_main(command, raw_args)