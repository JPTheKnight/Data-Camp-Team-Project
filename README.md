# Predicting Legal Assistance Costs Using Machine Learning

## Introduction

### Data Source

The data used in this challenge originates from financial and social assistance records. It includes various features such as regional distributions, household characteristics, and the total amount of legal aid provided. This dataset provides insights into financial assistance trends and the factors influencing them.

### Task Objective

The primary goal of this challenge is to develop a predictive model that accurately estimates the **total legal assistance amount (Montant total NDURINT)** based on available features. Participants must preprocess the dataset, engineer meaningful features, and build regression models to enhance prediction performance.

### Why This Matters

Predicting legal assistance costs is crucial for optimizing budget allocations, identifying regions with higher assistance needs, and making data-driven policy decisions. An accurate model can help government agencies and social organizations better manage resources and improve financial planning.

---

## Getting Started

### Install

To run a submission and execute the notebook, you need to install the dependencies listed in `requirements.txt`. Use the following command:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, you can set up your environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ramp-env
```

---

### Challenge Description

Get started with this RAMP challenge by exploring the provided
[dedicated notebook](submissions/starting_kit/estimator.py), which walks through the dataset, preprocessing steps, and baseline model implementation.

---

### Test a Submission

Submissions should be stored in the `submissions` folder. For example, a submission named `my_submission` should be placed in:

```
submissions/my_submission
```

To test a specific submission, use the `ramp-test` command:

```bash
ramp-test --submission starting_kit
```

For additional options and help, use:

```bash
ramp-test --help
```

---

### To Go Further

For more details on the RAMP framework and submission workflow, refer to the
[official documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html).
