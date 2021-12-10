# kipoi_veff
> Variant effect prediction plugin for Kipoi.

[![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Not%20Maintained-yellow.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)

## In order to simplify this code base we have rewritten parts of it in [kipoi-veff2](https://github.com/kipoi/kipoi-veff2). If you must use this code base, please make sure the following 
- `model_sources.kipoi.auto_update` is False in `~/.kipoi/config.yaml` 
- `git checkout c82b20d` at `model_sources.kipoi.local_path` from `~/.kipoi/config.yaml`. This will ensure your local models repo remains compatible with kipoi-veff. 


<a href='https://circleci.com/gh/kipoi/kipoi-veff'>
	<img alt='CircleCI' src='https://circleci.com/gh/kipoi/kipoi-veff.svg?style=svg' style="max-height:20px;width:auto">
</a>
<a href=https://coveralls.io/github/kipoi/kipoi-veff?branch=master>
	<img alt='Coverage status' src=https://coveralls.io/repos/github/kipoi/kipoi-veff/badge.svg?branch=master style="max-height:20px;width:auto;">
</a>

 
The variant effect prediction plug-in bring VCF annotation for DNA-sequence based models to models that are integrated into [Kipoi](https://kipoi.org). Additionally it offers visualisation tools to estimate the effect of possible genetic variants in a given area (mutation maps).

## Installation

To install the variant effect prediction plug-in use the following command:

```sh
pip install kipoi_veff
```

In order to use the variant effect prediction plug-in with Kipoi the `--vep` argument has to be added when setting up the model environment with `kipoi create env <model_name> --vep`.

Alternatively, you can use `conda` to install kipoi_veff:

```sh
conda install -c bioconda kipoi_veff
```

## Usage example

Main function of this package is `score_variants` accessible from the command line or python. It annotates the vcf file using model predictions for the reference and alternative alleles.

### CLI
```bash
$ kipoi veff score_variants -h
usage: kipoi veff score_variants [-h]
                                 [--source {kipoi,dir,github-permalink} [{kipoi,dir,github-permalink} ...]]
                                 [--dataloader DATALOADER [DATALOADER ...]]
                                 [--dataloader_source DATALOADER_SOURCE [DATALOADER_SOURCE ...]]
                                 [--dataloader_args DATALOADER_ARGS [DATALOADER_ARGS ...]]
                                 [-i INPUT_VCF] [-o OUTPUT_VCF]
                                 [--batch_size BATCH_SIZE] [-n NUM_WORKERS]
                                 [-r RESTRICTION_BED] [-e EXTRA_OUTPUT]
                                 [-s SCORES [SCORES ...]]
                                 [-k SCORE_KWARGS [SCORE_KWARGS ...]]
                                 [-l SEQ_LENGTH [SEQ_LENGTH ...]]
                                 [--std_var_id]
                                 [--model_outputs MODEL_OUTPUTS [MODEL_OUTPUTS ...]]
                                 [--model_outputs_i MODEL_OUTPUTS_I [MODEL_OUTPUTS_I ...]]
                                 model [model ...]

Predict effect of SNVs using ISM.

positional arguments:
  model                 Model name.

optional arguments:
  -h, --help            show this help message and exit
  --source {kipoi,dir,github-permalink} [{kipoi,dir,github-permalink} ...]
                        Model source to use. Specified in ~/.kipoi/config.yaml
                        under model_sources. 'dir' is an additional source
                        referring to the local folder.
  --dataloader DATALOADER [DATALOADER ...]
                        Dataloader name. If not specified, the model's
                        defaultDataLoader will be used
  --dataloader_source DATALOADER_SOURCE [DATALOADER_SOURCE ...]
                        Dataloader source
  --dataloader_args DATALOADER_ARGS [DATALOADER_ARGS ...]
                        Dataloader arguments either as a json string:'{"arg1":
                        1} or as a file path to a json file
  -i INPUT_VCF, --input_vcf INPUT_VCF
                        Input VCF.
  -o OUTPUT_VCF, --output_vcf OUTPUT_VCF
                        Output annotated VCF file path.
  --batch_size BATCH_SIZE
                        Batch size to use in prediction
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of parallel workers for loading the dataset
  -r RESTRICTION_BED, --restriction_bed RESTRICTION_BED
                        Regions for prediction can only be subsets of this bed
                        file
  -e EXTRA_OUTPUT, --extra_output EXTRA_OUTPUT
                        Additional output file. File format is inferred from
                        the file path ending. Available file formats are:
                        tsv,hdf5,h5
  -s SCORES [SCORES ...], --scores SCORES [SCORES ...]
                        Scoring method to be used. Only scoring methods
                        selected in the model yaml file areavailable except
                        for `diff` which is always available. Select scoring
                        function by the`name` tag defined in the model yaml
                        file.
  -k SCORE_KWARGS [SCORE_KWARGS ...], --score_kwargs SCORE_KWARGS [SCORE_KWARGS ...]
                        JSON definition of the kwargs for the scoring
                        functions selected in --scoring. The definiton can
                        either be in JSON in the command line or the path of a
                        .json file. The individual JSONs are expected to be
                        supplied in the same order as the labels defined in
                        --scoring. If the defaults or no arguments should be
                        used define '{}' for that respective scoring method.
  -l SEQ_LENGTH [SEQ_LENGTH ...], --seq_length SEQ_LENGTH [SEQ_LENGTH ...]
                        Optional parameter: Model input sequence length -
                        necessary if the model does not have a pre-defined
                        input sequence length.
  --std_var_id          If set then variant IDs in the annotated VCF will be
                        replaced with a standardised, unique ID.
  --model_outputs MODEL_OUTPUTS [MODEL_OUTPUTS ...]
                        Optional parameter: Only return predictions for the
                        selected model outputs. Namingaccording to the
                        definition in model.yaml > schema > targets >
                        column_labels
  --model_outputs_i MODEL_OUTPUTS_I [MODEL_OUTPUTS_I ...]
                        Optional parameter: Only return predictions for the
                        selected model outputs. Give integerindices of the
                        selected model output(s).
```

### Python

```python
from kipoi_veff import score_variants

# Signature
score_variants(model,
               dl_args,
               input_vcf,
               output_vcf,
               scores=["logit_ref", "logit_alt", "ref", "alt", "logit", "diff"],
               score_kwargs=None,
               num_workers=0,
               batch_size=32,
               source='kipoi',
               seq_length=None,
               std_var_id=False,
               restriction_bed=None,
               return_predictions=False,
               model_outputs = None)
```

Method arguments:
- model: model string or a model class instance
- dl_args: dataloader arguments as a dictionary
- input_vcf: input vcf file path
- output_vcf: output vcf file path
- scores: list of score names to compute. See kipoi_veff.scores
- score_kwargs: optional, list of kwargs that corresponds to the entries in score. For details see 
- num_workers: number of paralell workers to use for dataloading
- batch_size: batch_size for dataloading
- source: model source name
- std_var_id: If true then variant IDs in the annotated VCF will be replaced with a standardised, unique ID.
- seq_length: If model accepts variable input sequence length then this value has to be set!
- restriction_bed: If dataloader can be run with regions generated from the VCF then only variants that overlap
- regions defined in `restriction_bed` will be tested.
- return_predictions: return generated predictions also as pandas dataframe.
- model_outputs: If set then either a boolean filter or a named filter for model outputs that are reported.


## Development setup

If you want to help develop the Kipoi variant effect prediction plug-in, you are more than welcome to join. You should then install and test kipoi-veff as follows: 

```sh
git clone https://github.com/kipoi/kipoi-veff.git
cd kipoi-veff
pip install -e '.[develop]'
py.test -n 8 # Run tests using 8 workers
```


## Release History

* 0.1.0
    * First release to PyPI
