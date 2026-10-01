# Research guide

The study follows questions through three stages: estimating local-model
success, comparing human and classifier forecasts, and examining the electricity
and answer-quality consequences of routing on those forecasts. These guides
bring each part of the study together with its supporting results and commands.

## Where to start

| I want to… | Guide |
|---|---|
| Install the package and identify the inputs I need | [Getting started](docs/getting-started.md) |
| Understand or repeat conversion, generation and grading | [Questions and grading](docs/questions-and-grading.md) |
| Examine participant forecasts and the human analyses | [Human study](docs/human-study.md) |
| Load, train or evaluate the classifier | [Classifier](docs/classifier.md) |
| Examine electricity estimates or measure classifier inference | [Electricity](docs/electricity.md) |
| View, download or regenerate the figures | [Figures](docs/figures.md) |

## Reproduction workflow

Existing versioned datasets and predictions can enter at the relevant stage.
New generations and verdicts require the corresponding model runtime or
provider access. The diagram links to the instructions for each step.

```mermaid
---
config:
  layout: dagre
  flowchart:
    curve: linear
---
flowchart TD
    accTitle: Commands for reproducing the LocalGate study
    accDescr: Public source questions or versioned inputs feed question preparation, answer generation and grading to obtain five-answer references. These support study assignment and classifier development. The study interface collects participant responses, which require authorised access. Separate study reference labels use 30 answers per question. Train a classifier or load the published adapter, then predict. Both sets of reference labels, participant responses and predictions feed analysis, whose results feed figure generation. Solid arrows show data flow; the dashed arrow supplies classifier predictions.
    subgraph inputs["Prepare inputs"]
        I[("Public source questions<br/>or versioned inputs")]:::data
        D("localgate-data<br/>Prepare questions and<br/>generate answers"):::data
        G("localgate-grade<br/>Grade answers and<br/>reduce verdicts"):::data
        L[("Five-answer references")]:::data
        I --> D --> G --> L
    end
    subgraph forecasts["Obtain forecasts"]
        S("localgate-data<br/>Draw questions and<br/>assign participants"):::data
        H[("Participant responses<br/><b>Restricted access</b>")]:::excluded
        T("localgate-model<br/>Prepare and train"):::model
        P[("Published adapter")]:::model
        V[("Pinned base model<br/>and fit configuration")]:::model
        M("localgate-model<br/>Predict"):::model
        C("Study interface<br/>Collect forecasts"):::evaluation
        K[("Study reference labels<br/>30 answers per question")]:::data
        S --> C --> H
        T --> M
        P -->|Load adapter| M
        V --> M
    end
    subgraph results["Calculate results and figures"]
        A("localgate-analysis<br/>Calculate results"):::evaluation
        F[["localgate-figures<br/>Render figures"]]:::evaluation
        A --> F
    end
    L --> S
    L --> T
    L --> A
    H --> A
    K --> A
    M -. predictions .-> A
    style inputs fill:transparent,stroke:#89939e
    style forecasts fill:transparent,stroke:#a68abd
    style results fill:transparent,stroke:#68a99e
    classDef data fill:#203e56,stroke:#8bb8db,color:#ffffff
    classDef model fill:#423552,stroke:#bba0d0,color:#ffffff
    classDef evaluation fill:#194b46,stroke:#86c6bc,color:#ffffff
    classDef excluded fill:#383d43,stroke:#b0b7bf,color:#ffffff
    click I "https://github.com/LocalgateOrg/localgate-research#data-and-models" "Versioned input datasets" _blank
    click D "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/questions-and-grading.md#prepare-questions-and-generate-answers" "Data preparation and generation commands" _blank
    click G "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/questions-and-grading.md#grading" "Judge execution and reduction commands" _blank
    click L "https://huggingface.co/datasets/localgate/judge-panel-verdicts/tree/7d5bb21170294a4895bb9e3408794d7aa3fd55a1" "Published five-generation panel labels" _blank
    click S "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/human-study.md#draw-and-assign-study-questions" "Question sampling and participant assignment" _blank
    click C "https://github.com/LocalgateOrg/Study_LocalModel" "Study interface for collecting forecasts" _blank
    click K "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/getting-started.md" "Separate 30-generation study inputs" _blank
    click H "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/getting-started.md" "Participant-input access requirements" _blank
    click T "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/classifier.md#selected-classifier-refit" "Selected-model training commands" _blank
    click V "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/classifier.md#use-the-published-adapter" "Required base model and fit configuration" _blank
    click P "https://huggingface.co/localgate/larpbert/tree/fd9ac7db0b68c9d2817e0ad9783f6f8bd8c6a115" "Published classifier adapter" _blank
    click M "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/classifier.md#selected-classifier-refit" "Prediction commands" _blank
    click A "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/getting-started.md#commands-and-source-code" "Analysis commands" _blank
    click F "https://github.com/LocalgateOrg/localgate-research/blob/main/research/docs/figures.md#regenerate-the-figures" "Figure-generation commands" _blank
```

Solid arrows show data flow; dashed arrows show classifier predictions.

The 280 study questions and the remaining 1,375 test questions do not overlap.
On the study questions, both human and classifier forecasts are compared with
30 fresh generations. The broader classifier evaluation uses five-generation
estimates. The overview is also available as a
[study-map PDF](../artifacts/figures/01_study_map.pdf), alongside the
[complete figure collection](../artifacts/figures/).

## Paper and study materials

- [Submitted paper](../paper/Meissner_Bullard_Mizzaro_LocalGate_AI_and_Sustainability.pdf)
- [Preregistration](../paper/LARP_Preregistration_Human_Judgment_of_Local_AI_Model_Solvability.pdf)
- [Original participant-facing materials](../study/README.md)
- [Versioned datasets and model](../README.md#data-and-models)

The [input requirements](docs/getting-started.md#inputs-and-access) distinguish
public resources from the separate study inputs and restricted participant responses.
