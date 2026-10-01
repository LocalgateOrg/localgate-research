# Getting started

[Research guide](../README.md) · [Project overview](../../README.md)

## Installation

Run these commands from the repository root. The package requires Python 3.12
or later; these commands use Python 3.13 and the locked scientific dependencies
from the recorded calculation environment.
Figure regeneration has additional [rendering dependencies](figures.md#rendering-dependencies).

```bash
uv sync --python 3.13 --locked
install -d -m 700 output
uv run localgate-analysis --help
```

## Optional dependencies

The ordinary installation covers analysis, grading and plotting. Install extras
only for the workflow you need; the corresponding guide includes the command.

| Extra | Purpose |
|---|---|
| `corpus` | Obtain and prepare the original benchmark answer key. |
| `model` | Prepare, train and run the classifier. |
| `measurement` | Run classifier inference and measure or estimate its electricity use. |

Generation uses the separate runtime files documented in
[Questions and grading](questions-and-grading.md#prepare-questions-and-generate-answers).

## Commands and source code

| Command | Purpose | Implementation |
|---|---|---|
| `localgate-data` | Prepare questions, generate answers, build labels and draw samples. | [Data preparation](../data/) |
| `localgate-grade` | Obtain judge verdicts or reduce completed verdict files. | [Grading](../grading/judge.py) |
| `localgate-model` | Prepare development data, train, predict and measure classifier electricity. | [Model commands](../model_source/cli.py) |
| `localgate-analysis` | Calculate human, classifier, audit, electricity, power and reference-label results. | [Analyses](../analysis/) |
| `localgate-figures` | Render the six figures from newly calculated results. | [Figure generation](../figures/reproduce.py) |

## Inputs and access

The [versioned datasets, model adapter and grading protocol](../../README.md#data-and-models)
are public. Some analyses also require separately supplied study inputs, including
the 30-generation labels and generation records. Participant submissions and
background responses are restricted; anonymised responses may be requested from
the authors.

Each guide lists the files its commands need.
Existing records can be analysed without model calls. Generating new answers
requires the corresponding model runtime; live conversion and grading require
provider credentials and incur charges.

The finished figures can be viewed without these inputs or a package installation.
Recalculating participant-dependent results requires the restricted study inputs.

## Output directories

All commands write generated results under the ignored `output/` directory.
Commands handling participant or other private inputs require a new output
directory inside a private parent (mode 0700). Choose a new directory name when
repeating a run.
