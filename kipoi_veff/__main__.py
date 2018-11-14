import pkg_resources
import logging.config
from kipoi_veff.cli import main
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()