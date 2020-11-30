"""
Kipoi VCF parser

by Ziga Avsec
"""
import numpy as np
import re
from collections import OrderedDict


def get_info_tags(vcf):
    """Get info tags from a vcf file
    Args:
      vcf: cyvcf2.VCF object
    Returns:
      a list of ids's
    """
    return [x for x in vcf.header_iter()
            if x['HeaderType'] == 'INFO']


def get_info_ids(info_tags):
    return [x["ID"] for x in info_tags]


def get_kipoi_colnames(info_tags):
    """Get the kipoi column labels
    """

    def parse_kpvep_descr(desc):
        """Parse kipoi info tag
        """
        desc_split = re.split("Prediction from model outputs: ", desc.replace('"', ''))
        if len(desc_split) > 1:
            pred_labels = desc_split[-1].split("|")
            return pred_labels
        else:
            return None

    return OrderedDict([(x["ID"], parse_kpvep_descr(x["Description"]))
                        for x in info_tags if x["ID"].startswith("KV")])


def parse_kipoi_colname(colname):
    """Parse kipoi column name into:
    (model, version, type)
    input: "KV:dir:examples/rbp:DIFF"
    output: ("dir", "examples/rbp", "DIFF")
    """
    _, source_type, model, diff_type = colname.split(":", 3)
    return source_type, model, diff_type


def parse_kipoi_info(elem, colnames, prefix="", add_index=True):
    """Parse kipoi info field
    """
    if elem is None:
        elems = [np.nan] * len(colnames)
    else:
        elems = list(elem.split("|"))
        if colnames is None:
            colnames = ["unnamed_%d" % i for i in range(len(elems))]
        else:
            assert len(elems) == len(colnames)

    if add_index:
        return OrderedDict([(prefix + c + "_" + str(i), soft_to_float(elems[i]))
                            for i, c in enumerate(colnames)])
    else:
        return OrderedDict([(prefix + c, soft_to_float(elems[i]))
                            for i, c in enumerate(colnames)])


def variant_to_dict(variant):
    return OrderedDict([
        ('variant_id', "." if variant.ID is None else str(variant.ID)),
        ('variant_chr', variant.CHROM),
        ('variant_pos', variant.POS),
        ('variant_ref', variant.REF),
        ('variant_alt', variant.ALT[0] if variant.ALT else '.'),  # WARNING - assuming a single alternative
    ])


# TODO - convert all info tags to a nice dictionary or pandas data-frame
#         - TODO ?- should I convert it to a dictionary or a pandas.DataFrame?
#         - TODO ?- which columns do we need in the final table
#         - TODO ?- how to handle nested column names?


def soft_to_float(x):
    try:
        return float(x)
    except:
        return x


class KipoiVCFParser(object):
    """Iteratively parse a vcf file into a dictionary. This class was designed to work well with VCFs annotated by
    `kipoi-veff`. It performs automated shortening of column names.

    # Arguments
        vcf_file: .vcf file path (can be also .vcf.gz, .bcf, .bcf.gz)

    # Notes
    Iterator returns a nested dictionary with the schema:

    ```yaml
     - variant:
       - id
       - chr
       - pos
       - ref
       - alt
     - other:
       - f1
       - f2
     - kipoi:
       - model:
         - type:
           - feature1...
           - feature2...
    ```
    """

    def __init__(self, vcf_file):
        from cyvcf2 import VCF
        self.vcf_file = vcf_file
        self.vcf = VCF(vcf_file)
        self.info_tags = get_info_tags(self.vcf)
        self.info_ids = get_info_ids(self.info_tags)
        self.kipoi_colnames = get_kipoi_colnames(self.info_tags)
        self.kipoi_columns = [x for x in self.info_ids if x in self.kipoi_colnames]
        self.other_columns = [x for x in self.info_ids if x not in self.kipoi_columns]
        self.kipoi_parsed_colnames = {k: parse_kipoi_colname(k)
                                      for k in self.kipoi_colnames}

    def __iter__(self):
        return self

    def __next__(self):
        variant = next(self.vcf)
        out = variant_to_dict(variant)

        for k in self.other_columns:
            out['other_' + k] = variant.INFO.get(k)
        # out['kipoi'] = OrderedDict()
        for i, k in enumerate(self.kipoi_columns):
            source_name, model, diff_type = self.kipoi_parsed_colnames[k]
            prefix = 'KV_{model}_{diff_type}_'.format(model=model,
                                                      i=i,
                                                      diff_type=diff_type)
            out.update(
                parse_kipoi_info(variant.INFO.get(k),
                                 colnames=self.kipoi_colnames[k],
                                 prefix=prefix,
                                 add_index=True)
            )
        return out
