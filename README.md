# 📋 Compositional Complexity in Text and Images

This functional magnetic resonance imaging (fMRI) study aims at studying compositional processing across text and image modalities based on text-image pairs from the [Common Objects in Context-Actions (COCO-A)](https://www.vision.caltech.edu/~mronchi/projects/Cocoa/) dataset. Specifically, we will test the following main and alternative hypotheses:

1. **Main hypothesis — Modality-independent compositional processing**: There is a shared neural basis for processing compositional complexity across text and image modalities, reflected in shared regions in the univariate analysis or above chance multivariate pattern analysis (MVPA) decoding performance for low versus high compositional complexity for classifiers trained on one modality and tested on the other.
2. **Alternative hypothesis — Modality-specific compositional processing**: Distinct brain regions are involved in processing compositional complexity within each modality (i.e., text or images).

The experiment follows a 2 x 2 design: high vs. low compositional complexity & image vs. text modality. For a full explanation of the graph-based compositional complexity measure and experimental design, please see the study manuscript (see below).

![Example stimuli](https://github.com/helena-balabin/compositionality-study/raw/main/docs/source/study_overview.png)
This figure shows example text-image pairs with low and high compositional complexity, respectively. The text-image pairs, selected from the COCO-A dataset, were chosen based on the graph-based complexity measures of their corresponding action graphs (left) and abstract meaning representation (AMR) graphs (right). During the experiment (gray box), subjects will only see the raw images or text without any graph overlays.

## 🔬 Reproducibility

This study sets out to achieve a high degree of reproducibility through open science practices. We make our code publicly available and publish our research as a registered report, a publication format where the study design, hypotheses, and planned analyses are peer-reviewed and accepted before data collection begins. 

**Registered Report — Stage 1: In-Principle Acceptance (IPA)**: This study has received Stage 1 IPA. The methodology and proposed analyses were peer-reviewed and provisionally accepted for publication in [Neurobiology of Language (NoL)](https://direct.mit.edu/nol/pages/registered-reports), pending successful execution of the approved protocol.

- **OSF Project**: [Link to OSF repository](https://osf.io/ta45d/overview)
- **Stage 1 Manuscript**: [Link to manuscript](https://osf.io/ta45d/files/4nd6h)

Upon study completion, we plan to make the (anonymized/defaced) fMRI dataset publicly available.

## ⬇️ Installation

To install the code, use a virtual environment manager of your choice (poetry/conda/virtualenv) and install the requirements listed in the ```pyproject.toml```, e.g.:

```poetry install```
