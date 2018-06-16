"""Postprocessing CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import sys

import kipoi
import kipoi_veff
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi_veff.specs import VarEffectFuncType
from kipoi_veff.scores import get_scoring_fns
from kipoi import writers
from kipoi.utils import cd
from kipoi.utils import parse_json_file_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _prepare_multi_model_args(args):
    assert isinstance(args.model, list)
    assert isinstance(args.source, list)
    assert isinstance(args.seq_length, list)
    assert isinstance(args.dataloader, list)
    assert isinstance(args.dataloader_source, list)
    assert isinstance(args.dataloader_args, list)

    def ensure_matching_args(ref_arg, query_arg, ref_label, query_label, allow_zero=True):
        assert isinstance(ref_arg, list)
        assert isinstance(query_arg, list)
        n = len(ref_arg)
        if allow_zero and (len(query_arg) == 0):
            ret = [None] * n
        elif len(query_arg) == 1:
            ret = [query_arg[0]] * n
        elif not len(query_arg) == n:
            raise Exception("Either give one {q} for all {r} or one {q} for every {r} in the same order.".format(
                q=query_label, r=ref_label))
        else:
            ret = query_arg
        return ret

    args.source = ensure_matching_args(args.model, args.source, "--model", "--source", allow_zero=False)
    args.seq_length = ensure_matching_args(args.model, args.seq_length, "--model", "--seq_length")
    args.dataloader = ensure_matching_args(args.model, args.dataloader, "--model", "--dataloader")
    args.dataloader_source = ensure_matching_args(args.dataloader, args.dataloader_source, "--dataloader",
                                                  "--dataloader_source")
    args.dataloader_args = ensure_matching_args(args.model, args.dataloader_args, "--model",
                                                "--dataloader_args", allow_zero=False)


def cli_score_variants(command, raw_args):
    """CLI interface to score variants
    """
    # Updated argument names:
    # - scoring -> scores
    # - --vcf_path -> --input_vcf, -i
    # - --out_vcf_fpath -> --output_vcf, -o
    # - --output -> -e, --extra_output
    # - remove - -install_req
    # - scoring_kwargs -> score_kwargs
    AVAILABLE_FORMATS = ["tsv", "hdf5", "h5"]
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    parser.add_argument('model', help='Model name.', nargs="+")
    parser.add_argument('--source', default=["kipoi"], nargs="+",
                        choices=list(kipoi.config.model_sources().keys()),
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                             " under model_sources. " +
                             "'dir' is an additional source referring to the local folder.")
    parser.add_argument('--dataloader', nargs="+", default=[],
                        help="Dataloader name. If not specified, the model's default" +
                             "DataLoader will be used")
    parser.add_argument('--dataloader_source', nargs="+", default=["kipoi"],
                        help="Dataloader source")
    parser.add_argument('--dataloader_args', nargs="+", default=[],
                        help="Dataloader arguments either as a json string:" +
                             "'{\"arg1\": 1} or as a file path to a json file")
    parser.add_argument('-i', '--input_vcf',
                        help='Input VCF.')
    parser.add_argument('-o', '--output_vcf',
                        help='Output annotated VCF file path.', default=None)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument('-r', '--restriction_bed', default=None,
                        help="Regions for prediction can only be subsets of this bed file")
    parser.add_argument('-e', '--extra_output', required=False,
                        help="Additional output file. File format is inferred from the file path ending" +
                             ". Available file formats are: {0}".format(",".join(AVAILABLE_FORMATS)))
    parser.add_argument('-s', "--scores", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--score_kwargs", default="", nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scoring. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scoring. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument('-l', "--seq_length", type=int, nargs="+", default=[],
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")
    parser.add_argument('--std_var_id', action="store_true", help="If set then variant IDs in the annotated"
                                                                  " VCF will be replaced with a standardised, unique ID.")

    args = parser.parse_args(raw_args)
    # Make sure all the multi-model arguments like source, dataloader etc. fit together
    _prepare_multi_model_args(args)

    # Check that all the folders exist
    file_exists(args.input_vcf, logger)
    dir_exists(os.path.dirname(args.output_vcf), logger)
    if args.extra_output is not None:
        dir_exists(os.path.dirname(args.extra_output), logger)

        # infer the file format
        args.file_format = args.extra_output.split(".")[-1]
        if args.file_format not in AVAILABLE_FORMATS:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(args.file_format, args.extra_output, AVAILABLE_FORMATS))
            sys.exit(1)

        if args.file_format in ["hdf5", "h5"]:
            # only if hdf5 output is used
            import deepdish

    if not isinstance(args.scores, list):
        args.scores = [args.scores]

    score_kwargs = []
    if len(args.score_kwargs) > 0:
        score_kwargs = args.score_kwargs
        if len(args.scores) >= 1:
            # Check if all scoring functions should be used:
            if args.scores == ["all"]:
                if len(score_kwargs) >= 1:
                    raise ValueError("`--score_kwargs` cannot be defined in combination will `--scoring all`!")
            else:
                score_kwargs = [parse_json_file_str(el) for el in score_kwargs]
                if not len(args.score_kwargs) == len(score_kwargs):
                    raise ValueError("When defining `--score_kwargs` a JSON representation of arguments (or the "
                                     "path of a file containing them) must be given for every "
                                     "`--scores` function.")

    keep_predictions = args.extra_output is not None

    n_models = len(args.model)

    res = {}
    for model_name, model_source, dataloader, dataloader_source, dataloader_args, seq_length in zip(args.model,
                                                                                                    args.source,
                                                                                                    args.dataloader,
                                                                                                    args.dataloader_source,
                                                                                                    args.dataloader_args,
                                                                                                    args.seq_length):
        model_name_safe = model_name.replace("/", "_")
        output_vcf_model = None
        if args.output_vcf is not None:
            output_vcf_model = args.output_vcf
            # If multiple models are to be analysed then vcfs need renaming.
            if n_models > 1:
                if output_vcf_model.endswith(".vcf"):
                    output_vcf_model = output_vcf_model[:-4]
                output_vcf_model += model_name_safe + ".vcf"

        dataloader_arguments = parse_json_file_str(dataloader_args)

        # --------------------------------------------
        # load model & dataloader
        model = kipoi.get_model(model_name, model_source)

        if dataloader is not None:
            Dl = kipoi.get_dataloader_factory(dataloader, dataloader_source)
        else:
            Dl = model.default_dataloader

        # Load effect prediction related model info
        model_info = kipoi_veff.ModelInfoExtractor(model, Dl)

        if model_info.use_seq_only_rc:
            logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
        else:
            logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

        if output_vcf_model is not None:
            logger.info('Annotated VCF will be written to %s.' % str(output_vcf_model))

        res[model_name_safe] = kipoi_veff.score_variants(model,
                                                         dataloader_arguments,
                                                         args.input_vcf,
                                                         output_vcf_model,
                                                         scores=args.scores,
                                                         score_kwargs=score_kwargs,
                                                         num_workers=args.num_workers,
                                                         batch_size=args.batch_size,
                                                         seq_length=seq_length,
                                                         std_var_id=args.std_var_id,
                                                         restriction_bed=args.restriction_bed,
                                                         return_predictions=keep_predictions)

    # tabular files
    if keep_predictions:
        if args.file_format in ["tsv"]:
            for model_name in res:
                for i, k in enumerate(res[model_name]):
                    # Remove an old file if it is still there...
                    if i == 0:
                        try:
                            os.unlink(args.extra_output)
                        except Exception:
                            pass
                    with open(args.extra_output, "w") as ofh:
                        ofh.write("KPVEP_%s:%s\n" % (k.upper(), model_name))
                        res[model_name][k].to_csv(args.extra_output, sep="\t", mode="a")

        if args.file_format in ["hdf5", "h5"]:
            deepdish.io.save(args.extra_output, res)

    logger.info('Successfully predicted samples')


def isint(qstr):
    import re
    return bool(re.match("^[0-9]+$", qstr))


# Parse the slice
def parse_filter_slice(in_str):
    if in_str.startswith("(") or in_str.startswith("["):
        in_str_els = in_str.lstrip("([").rstrip(")]").split(",")
        slices = []
        for slice_el in in_str_els:
            slice_el = slice_el.strip(" ")
            if slice_el == "...":
                slices.append(Ellipsis)
            elif isint(slice_el):
                slices.append(int(slice_el))
                if len(in_str_els) == 1:
                    return int(slice_el)
            else:
                # taken from https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
                slices.append(slice(*map(lambda x: int(x.strip()) if x.strip() else None, slice_el.split(':'))))
        return tuple(slices)
    elif isint(in_str):
        return int(in_str)
    else:
        raise Exception("Filter index slice not valid. Allowed values are e.g.: '1', [1:3,...], [:, 0:4, ...]")


def cli_grad(command, raw_args):
    """CLI interface to predict
    """
    from .main import prepare_batch
    from kipoi.model import GradientMixin
    assert command == "grad"
    from tqdm import tqdm
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Save gradients and inputs to a hdf5 file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-l", "--layer", default=None,
                        help="Which output layer to use to make the predictions. If specified," +
                             "`model.predict_activation_on_batch` will be invoked instead of `model.predict_on_batch`",
                        required=False)
    parser.add_argument("--final_layer",
                        help="Alternatively to `--layer` this flag can be used to indicate that the last layer should "
                             "be used.", action='store_true')
    parser.add_argument("--pre_nonlinearity",
                        help="Flag indicating that it should checked whether the selected output is post activation "
                             "function. If a non-linear activation function is used attempt to use its input. This "
                             "feature is not available for all models.", action='store_true')
    parser.add_argument("-f", "--filter_idx",
                        help="Filter index that should be inspected with gradients. If not set all filters will " +
                             "be used.", default=None)
    parser.add_argument("-a", "--avg_func",
                        help="Averaging function to be applied across selected filters (`--filter_idx`) in " +
                             "layer `--layer`.", choices=GradientMixin.allowed_functions, default="sum")
    parser.add_argument('--selected_fwd_node', help="If the selected layer has multiple inbound connections in "
                                                    "the graph then those can be selected here with an integer "
                                                    "index. Not necessarily supported by all models.",
                        default=None, type=int)
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                             ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    # setup the files
    if not isinstance(args.output, list):
        args.output = [args.output]
    for o in args.output:
        ending = o.split('.')[-1]
        if ending not in writers.FILE_SUFFIX_MAP:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(ending, o, writers.FILE_SUFFIX_MAP))
            sys.exit(1)
        dir_exists(os.path.dirname(o), logger)
    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)

    layer = args.layer
    if layer is None and not args.final_layer:
        raise Exception("A layer has to be selected explicitely using `--layer` or implicitely by using the"
                        "`--final_layer` flag.")

    # Not a good idea
    # if layer is not None and isint(layer):
    #    logger.warn("Interpreting `--layer` value as integer layer index!")
    #    layer = int(args.layer)

    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if not isinstance(model, GradientMixin):
        raise Exception("Model does not support gradient calculation.")

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

    filter_idx_parsed = None
    if args.filter_idx is not None:
        filter_idx_parsed = parse_filter_slice(args.filter_idx)

    # setup batching
    it = dl.batch_iter(batch_size=args.batch_size,
                       num_workers=args.num_workers)

    # Setup the writers
    use_writers = []
    for output in args.output:
        ending = output.split('.')[-1]
        W = writers.FILE_SUFFIX_MAP[ending]
        logger.info("Using {0} for file {1}".format(W.__name__, output))
        if ending == "tsv":
            assert W == writers.TsvBatchWriter
            use_writers.append(writers.TsvBatchWriter(file_path=output, nested_sep="/"))
        elif ending == "bed":
            raise Exception("Please use tsv or hdf5 output format.")
        elif ending in ["hdf5", "h5"]:
            assert W == writers.HDF5BatchWriter
            use_writers.append(writers.HDF5BatchWriter(file_path=output))
        else:
            logger.error("Unknown file format: {0}".format(ending))
            sys.exit(1)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # make the prediction
        pred_batch = model.input_grad(batch['inputs'], filter_idx=filter_idx_parsed,
                                      avg_func=args.avg_func, layer=layer, final_layer=args.final_layer,
                                      selected_fwd_node=args.selected_fwd_node,
                                      pre_nonlinearity=args.pre_nonlinearity)

        # write out the predictions, metadata (, inputs, targets)
        # always keep the inputs so that input*grad can be generated!
        # output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
        output_batch = batch
        output_batch["grads"] = pred_batch
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Gradients stored in {0}'.format(",".join(args.output)))


# --------------------------------------------
# CLI commands


command_functions = {
    'score_variants': cli_score_variants,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi postproc <command> [-h] ...

    # Available sub-commands:
    score_variants        Score variants with a kipoi model
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


def cli_main(command, raw_args):
    args = parser.parse_args(raw_args[0:1])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, raw_args[1:])
