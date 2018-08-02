import pytest
import subprocess
import sys
import os
import yaml
import pandas as pd
import config
# import filecmp
from utils import compare_vcfs, temp
from kipoi.readers import HDF5Reader
import numpy as np

if config.install_req:
    INSTALL_FLAG = "--install_req"
else:
    INSTALL_FLAG = ""


predict_activation_layers = {
    "rbp": "concatenate_6",
    "pyt": "3"  # two before the last layer
}


class dummy_container(object):
    pass


def test__prepare_multi_model_args():
    from kipoi_veff.cli import _prepare_multi_model_args
    any_len = ["seq_length", "dataloader", "dataloader_source"]
    keys = ["model", "source", "seq_length", "dataloader", "dataloader_source", "dataloader_args"]
    for some_empty in [True, False]:
        args = dummy_container()
        for k in keys:
            if k in any_len and some_empty:
                setattr(args, k, [])
            else:
                setattr(args, k, ["a", "b"])
        _prepare_multi_model_args(args)
        for k in keys:
            assert len(getattr(args, k)) == len(getattr(args, "model"))
            if k in any_len and some_empty:
                assert all([el is None for el in getattr(args, k)])
            else:
                assert all([el is not None for el in getattr(args, k)])
    args = dummy_container()
    for k in keys:
        setattr(args, k, ["a", "b"])
    args.model = ["a"]
    with pytest.raises(Exception):
        _prepare_multi_model_args(args)


@pytest.mark.parametrize("file_format", ["tsv", "hdf5"])
def test_predict_variants_example_multimodel(file_format, tmpdir):
    """kipoi predict ...
    """
    if sys.version_info[0] == 2:
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    examples = "rbp", "non_bedinput_model"
    example_dirs = ["tests/models/{0}/".format(ex) for ex in examples]
    main_example_dir = example_dirs[1]

    tmpdir_here = tmpdir.mkdir("example")

    # non_bedinput_model is not compatible with restricted bed files as
    # alterations in region generation have no influence on that model
    tmpfile = str(tmpdir_here.join("out.{0}".format(file_format)))
    vcf_tmpfile = str(tmpdir_here.join("out.{0}".format("vcf")))

    dataloader_kwargs = {"fasta_file": "example_files/hg38_chr22.fa",
                         "preproc_transformer": "dataloader_files/encodeSplines.pkl",
                         "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
                         "intervals_file": "example_files/variant_intervals.tsv"}
    dataloader_kwargs = {k: main_example_dir + v for k, v in dataloader_kwargs.items()}
    import json
    dataloader_kwargs_str = json.dumps(dataloader_kwargs)

    args = ["python", os.path.abspath("./kipoi_veff/cli.py"),
            "score_variants",
            # "./",  # directory
            example_dirs[0], example_dirs[1],
            "--source=dir",
            "--batch_size=4",
            "--dataloader_args='%s'" % dataloader_kwargs_str,
            "--input_vcf", temp(main_example_dir + "/example_files/variants.vcf"),
            # this one was now gone in the master?!
            "--output_vcf", vcf_tmpfile,
            "--extra_output", tmpfile]
    # run the
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)

    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(main_example_dir) + "/../../../")
    assert returncode == 0

    assert os.path.exists(tmpfile)

    for example_dir in example_dirs:
        # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)
        vcf_tmpfile_model = vcf_tmpfile[:-4] + example_dir.replace("/", "_") + ".vcf"
        assert os.path.exists(vcf_tmpfile_model)
        compare_vcfs(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile_model)

    if file_format == "hdf5":
        data = HDF5Reader.load(tmpfile)
    else:
        table_labels = []
        table_starts = []
        table_ends = []
        tables = {}
        head_line_id = "KPVEP_"
        with open(tmpfile, "r") as ifh:
            for i, l in enumerate(ifh):
                if head_line_id in l:
                    if (len(table_starts) > 0):
                        table_ends.append(i - 1)
                    table_labels.append(l.rstrip()[len(head_line_id):])
                    table_starts.append(i + 1)
            table_ends.append(i)
        for label, start, end in zip(table_labels, table_starts, table_ends):
            tables[label] = pd.read_csv(tmpfile, sep="\t", skiprows=start, nrows=end - start, index_col=0)


@pytest.mark.parametrize("example", ["rbp", "non_bedinput_model"])
@pytest.mark.parametrize("restricted_bed", [True, False])
@pytest.mark.parametrize("file_format", ["tsv", "hdf5"])
def test_predict_variants_example(example, restricted_bed, file_format, tmpdir):
    """kipoi predict ...
    """
    if (example not in {"rbp", "non_bedinput_model"}) or (sys.version_info[0] == 2):
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    example_dir = "tests/models/{0}/".format(example)

    tmpdir_here = tmpdir.mkdir("example")

    # non_bedinput_model is not compatible with restricted bed files as
    # alterations in region generation have no influence on that model
    if restricted_bed and (example != "rbp"):
        pytest.skip("Resticted_bed only available for rbp_eclip")
    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir_here.join("out.{0}".format(file_format)))
    vcf_tmpfile = str(tmpdir_here.join("out.{0}".format("vcf")))

    dataloader_kwargs = {"fasta_file": "example_files/hg38_chr22.fa",
                         "preproc_transformer": "dataloader_files/encodeSplines.pkl",
                         "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
                         "intervals_file": "example_files/variant_intervals.tsv"}
    dataloader_kwargs = {k: example_dir + v for k, v in dataloader_kwargs.items()}
    import json
    dataloader_kwargs_str = json.dumps(dataloader_kwargs)

    args = ["python", os.path.abspath("./kipoi_veff/cli.py"),
            "score_variants",
            # "./",  # directory
            example_dir,
            "--source=dir",
            "--batch_size=4",
            "--dataloader_args='%s'" % dataloader_kwargs_str,
            "--input_vcf", temp(example_dir + "/example_files/variants.vcf"),
            # this one was now gone in the master?!
            "--output_vcf", vcf_tmpfile,
            "--extra_output", tmpfile]
    # run the
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)

    if restricted_bed:
        args += ["--restriction_bed", example_dir + "/example_files/restricted_regions.bed"]

    returncode = subprocess.call(args=args, cwd=".")
    assert returncode == 0

    assert os.path.exists(tmpfile)
    assert os.path.exists(vcf_tmpfile)

    if restricted_bed:
        # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out2.vcf", vcf_tmpfile)
        compare_vcfs(example_dir + "/example_files/variants_ref_out2.vcf", vcf_tmpfile)
    else:
        # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)
        compare_vcfs(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)

    if file_format == "hdf5":
        data = HDF5Reader.load(tmpfile)
    else:
        table_labels = []
        table_starts = []
        table_ends = []
        tables = {}
        head_line_id = "KPVEP_"
        with open(tmpfile, "r") as ifh:
            for i, l in enumerate(ifh):
                if head_line_id in l:
                    if (len(table_starts) > 0):
                        table_ends.append(i - 1)
                    table_labels.append(l.rstrip()[len(head_line_id):])
                    table_starts.append(i + 1)
            table_ends.append(i)
        for label, start, end in zip(table_labels, table_starts, table_ends):
            tables[label] = pd.read_csv(tmpfile, sep="\t", skiprows=start, nrows=end - start, index_col=0)
