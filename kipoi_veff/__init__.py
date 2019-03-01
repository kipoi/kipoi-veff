from __future__ import absolute_import

__author__ = 'Kipoi team'
__email__ = 'avsec@in.tum.de'
__version__ = '0.2.3'


from . import scores
from . import parsers
from . import utils
from . import snv_predict
from . import specs
# from .scores import Ref, Alt, Diff, LogitRef, LogitAlt, Logit, DeepSEA_effect
# from .parsers import KipoiVCFParser

from .snv_predict import predict_snvs, analyse_model_preds, score_variants
from .utils import ModelInfoExtractor, SnvPosRestrictedRg, SnvCenteredRg, ensure_tabixed_vcf, VcfWriter, \
    BedOverlappingRg

from .mutation_map import MutationMap, MutationMapPlotter


# Required by kipoi
from .__main__ import cli_main
from .specs import VarEffectModelArgs as ModelParser
from .specs import VarEffectDataLoaderArgs as DataloaderParser
