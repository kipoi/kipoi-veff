"""CLI
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
from kipoi_veff.utils.io import SyncBatchWriter
from kipoi import writers
from kipoi_utils.utils import cd
from kipoi_utils.utils import parse_json_file_str, parse_json_file_str_or_arglist
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def get_single(x, name):
    """Make sure only a single element is used
    """
    if isinstance(x, list):
        if len(x) == 1:
            return x[0]
        else:
            raise ValueError("Only a single {} can be used with --singularity".format(name))
    else:
        return x


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
    AVAILABLE_FORMATS = [k for k in writers.FILE_SUFFIX_MAP if k != 'bed']
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi veff {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    parser.add_argument('model', help='Model name.')
    parser.add_argument('--source', default="kipoi",
                        choices=list(kipoi.config.model_sources().keys()),
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                             " under model_sources. " +
                             "'dir' is an additional source referring to the local folder.")

    add_dataloader(parser=parser, with_args=True)

    parser.add_argument('-i', '--input_vcf', required=True,
                        help='Input VCF.')
    parser.add_argument('-o', '--output_vcf',
                        help='Output annotated VCF file path.', default=None)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument('-r', '--restriction_bed', default=None,
                        help="Regions for prediction can only be subsets of this bed file")
    parser.add_argument('-e', '--extra_output', type=str, default=None, required=False,
                        help="Additional output files in other (non-vcf) formats. File format is inferred from the file path ending" +
                             ". Available file formats are: {0}".format(", ".join(["." + k for k in AVAILABLE_FORMATS])))
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
    parser.add_argument('-l', "--seq_length", type=int, default=None,
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")
    parser.add_argument('--std_var_id', action="store_true", help="If set then variant IDs in the annotated"
                                                                  " VCF will be replaced with a standardised, unique ID.")

    parser.add_argument("--model_outputs", type=str, default=None, nargs="+",
                        help="Optional parameter: Only return predictions for the selected model outputs. Naming"
                             "according to the definition in model.yaml > schema > targets > column_labels")

    parser.add_argument("--model_outputs_i", type=int, default=None, nargs="+",
                        help="Optional parameter: Only return predictions for the selected model outputs. Give integer"
                             "indices of the selected model output(s).")

    parser.add_argument("--singularity", action='store_true',
                        help="Run `kipoi predict` in the appropriate singularity container. "
                        "Containters will get downloaded to ~/.kipoi/envs/ or to "
                        "$SINGULARITY_CACHEDIR if set")

    args = parser.parse_args(raw_args)

    # OBSOLETE
    # Make sure all the multi-model arguments like source, dataloader etc. fit together
    #_prepare_multi_model_args(args)

    # Check that all the folders exist
    file_exists(args.input_vcf, logger)

    if args.output_vcf is None and args.extra_output is None:
        logger.error("One of the two needs to be specified: --output_vcf or --extra_output")
        sys.exit(1)

    if args.extra_output is not None:
        dir_exists(os.path.dirname(args.extra_output), logger)
        ending = args.extra_output.split('.')[-1]
        if ending not in AVAILABLE_FORMATS:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(ending, args.extra_output, AVAILABLE_FORMATS))
            sys.exit(1)

    # singularity_command
    if args.singularity:
        from kipoi.cli.singularity import singularity_command
        logger.info("Running kipoi veff {} in the singularity container".format(command))

        # Drop the singularity flag
        raw_args = [x for x in raw_args if x != '--singularity']

        dataloader_kwargs = parse_json_file_str_or_arglist(args.dataloader_args)

        # create output files
        output_files = []
        if args.output_vcf is not None:
            output_files.append(args.output_vcf)
        if args.extra_output is not None:
            output_files.append(args.extra_output)

        singularity_command(['kipoi', 'veff', command] + raw_args,
                            model=args.model,
                            dataloader_kwargs=dataloader_kwargs,
                            output_files=output_files,
                            source=args.source,
                            dry_run=False)
        return None

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

    # VCF writer
    output_vcf_model = None
    if args.output_vcf is not None:
        dir_exists(os.path.dirname(args.output_vcf), logger)
        output_vcf_model = args.output_vcf

    # Other writers
    if args.extra_output is not None:
        dir_exists(os.path.dirname(args.extra_output), logger)
        extra_output = args.extra_output
        writer = writers.get_writer(extra_output, metadata_schema=None)
        assert writer is not None
        extra_writers = [SyncBatchWriter(writer)]
    else:
        extra_writers = []

    dataloader_arguments = parse_json_file_str_or_arglist(args.dataloader_args)

    # --------------------------------------------
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
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

    model_outputs = None
    if args.model_outputs is not None:
        model_outputs = args.model_outputs

    elif args.model_outputs_i is not None:
        model_outputs = args.model_outputs_i

    kipoi_veff.score_variants(model,
                              dataloader_arguments,
                              args.input_vcf,
                              output_vcf=output_vcf_model,
                              output_writers=extra_writers,
                              scores=args.scores,
                              score_kwargs=score_kwargs,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              seq_length=args.seq_length,
                              std_var_id=args.std_var_id,
                              restriction_bed=args.restriction_bed,
                              return_predictions=False,
                              model_outputs=model_outputs)


    logger.info('Successfully predicted samples')


def isint(qstr):
    import re
    return bool(re.match("^[0-9]+$", qstr))


def cli_create_mutation_map(command, raw_args):
    """CLI interface to calculate mutation map data
    """
    assert command == "create_mutation_map"
    parser = argparse.ArgumentParser('kipoi veff {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-r', '--regions_file',
                        help='Region definition as VCF or bed file. Not a required input.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument('-o', '--output', required=True,
                        help="Output HDF5 file. To be used as input for plotting.")
    parser.add_argument('-s', "--scores", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--score_kwargs", default="", nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scores. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scores. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument('-l', "--seq_length", type=int, default=None,
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")

    parser.add_argument("--singularity", action='store_true',
                        help="Run `kipoi predict` in the appropriate singularity container. "
                        "Containters will get downloaded to ~/.kipoi/envs/ or to "
                        "$SINGULARITY_CACHEDIR if set")

    args = parser.parse_args(raw_args)

    # extract args for kipoi.variant_effects.predict_snvs
    print("DL ARGS",args.dataloader_args)
    dataloader_arguments = parse_json_file_str_or_arglist(args.dataloader_args)
    #dataloader_arguments = parse_json_file_str(args.dataloader_args)

    if args.output is None:
        raise Exception("Output file `--output` has to be set!")

    if args.singularity:
        from kipoi.cli.singularity import singularity_command
        logger.info("Running kipoi veff in the singularity container".format(command))
        # Drop the singularity flag
        raw_args = [x for x in raw_args if x != '--singularity']
        singularity_command(['kipoi', 'veff', command] + raw_args,
                            args.model,
                            dataloader_arguments,
                            output_files=args.output,
                            source=args.source,
                            dry_run=False)
        return None

    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model, args.source, and_dataloaders=True)
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    regions_file = os.path.realpath(args.regions_file)
    output = os.path.realpath(args.output)
    with cd(model.source_dir):
        if not os.path.exists(regions_file):
            raise Exception("Regions inputs file does not exist: %s" % args.regions_file)

        # Check that all the folders exist
        file_exists(regions_file, logger)
        dir_exists(os.path.dirname(output), logger)

        if args.dataloader is not None:
            Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
        else:
            Dl = model.default_dataloader

    if not isinstance(args.scores, list):
        args.scores = [args.scores]

    # TODO - why is this function not a method of the model class?
    dts = get_scoring_fns(model, args.scores, args.score_kwargs)

    # Load effect prediction related model info
    model_info = kipoi_veff.ModelInfoExtractor(model, Dl)
    manual_seq_len = args.seq_length

    # Select the appropriate region generator and vcf or bed file input
    args.file_format = regions_file.split(".")[-1]
    bed_region_file = None
    vcf_region_file = None
    bed_to_region = None
    vcf_to_region = None
    if args.file_format == "vcf" or regions_file.endswith("vcf.gz"):
        vcf_region_file = regions_file
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            vcf_to_region = kipoi_veff.SnvCenteredRg(model_info, seq_length=manual_seq_len)
            logger.info('Using variant-centered sequence generation.')
    elif args.file_format == "bed":
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            bed_to_region = kipoi_veff.BedOverlappingRg(model_info, seq_length=manual_seq_len)
            logger.info('Using bed-file based sequence generation.')
        bed_region_file = regions_file
    else:
        raise Exception("")

    if model_info.use_seq_only_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    from kipoi_veff.mutation_map import _generate_mutation_map
    mdmm = _generate_mutation_map(model,
                                  Dl,
                                  vcf_fpath=vcf_region_file,
                                  bed_fpath=bed_region_file,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  dataloader_args=dataloader_arguments,
                                  vcf_to_region=vcf_to_region,
                                  bed_to_region=bed_to_region,
                                  evaluation_function_kwargs={'diff_types': dts},
                                  )
    mdmm.save_to_file(output)

    logger.info('Successfully generated mutation map data')


def cli_plot_mutation_map(command, raw_args):
    """CLI interface to plot mutation map
    """
    assert command == "plot_mutation_map"
    parser = argparse.ArgumentParser('kipoi veff {}'.format(command),
                                     description='Plot mutation map in a file.')
    # TODO - rename path to fpath

    # TODO - input file should be the default
    parser.add_argument('-f', '--input_file', required=False,
                        help="Input HDF5 file produced from `create_mutation_map`")
    parser.add_argument('-o', '--output', required=False,
                        help="Output image file")
    parser.add_argument('--input_entry', required=True, type=int,
                        help="Input line for which the plot should be generated")
    parser.add_argument('--model_seq_input', required=True,
                        help="Model input name to be used for plotting. As defined in model.yaml.")
    parser.add_argument('--scoring_key', required=True,
                        help="Variant score label to be used for plotting. As defined when running "
                             "`create_mutation_map`.")
    parser.add_argument('--model_output', required=True,
                        help="Model output to be used for plotting. As defined in model.yaml.")
    parser.add_argument('--limit_region_genomic', required=False, nargs=2, type=int, default=None,
                        help="Limit to genomic region. Given as tuple without chromosome, "
                             "eg: `--limit_region_genomic 13245 12347`")
    parser.add_argument('--rc_plot', required=False, action="store_true",
                        help="Make reverse-complement plot.")
    args = parser.parse_args(raw_args)

    # Check that all the folders exist
    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    # install args
    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pylab as plt
    from kipoi_veff.mutation_map import MutationMapPlotter

    logger.info('Loading mutation map file...')

    mutmap = MutationMapPlotter(fname=args.input_file)

    fig = plt.figure(figsize=(20, 2))
    ax = plt.subplot(1, 1, 1)

    logger.info('Plotting...')

    if args.limit_region_genomic is not None:
        args.limit_region_genomic = tuple(args.limit_region_genomic)

    mutmap.plot_mutmap(args.input_entry, args.model_seq_input, args.scoring_key, args.model_output, ax=ax,
                       limit_region_genomic=args.limit_region_genomic, rc_plot=args.rc_plot)
    fig.savefig(args.output)

    logger.info('Successfully plotted mutation map')


# --------------------------------------------
# CLI commands


command_functions = {
    'score_variants': cli_score_variants,
    'plot_mutation_map': cli_plot_mutation_map,
    'create_mutation_map': cli_create_mutation_map,
    # TOOD - add and test mutation maps
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi veff plugin command line tool',
    usage='''kipoi veff <command> [-h] ...

    # Available sub-commands:
    score_variants        Score variants with a kipoi model
    create_mutation_map   Calculate variant effect scores for mutation map plotting
    plot_mutation_map     Plot mutation map from data generated in `create_mutation_map`
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


if __name__ == '__main__':
    command = sys.argv[1]
    raw_args = sys.argv[1:]
    cli_main(command, raw_args)
