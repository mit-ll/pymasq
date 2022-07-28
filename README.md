# PyMASq

<p align="center">
    <img src="./assets/images/masq_logo_light.svg" width="150px"/>
</p>

## Python-based Mitigation Application and Assessment (MASq)

## Introduction

In recent years, the advancement of computational technologies and artificial intelligence/machine learning (AI/ML) capabilities have resulted in vast amounts of data becoming publicly available (intentionally disclosed or not) degrading the privacy of individuals and organizations potentially leading to hacking, discrimination, ransoming, and exploitation.  To gain decision advantage over the US government and financial leverage over US persons adversaries have disclosed sensitive information. In addition, data aggregation efforts expose patterns of sensitive activities by providing additional context about sensitive records and attributes.  As more datasets are involved in transparency and aggregation, institutions often do not have enough subject matter experts to assess and mitigate risks of disclosure, the proper tools to effectively and efficiently de-identify data, or the time to fully vet risk and utility of the data to missions and stakeholders. Academic research contains a number of viable approaches for mitigating risk of exposing sensitive information, but tools are either not automated, no longer supported, or designed for experts in the field.  An automated decision support tool is required to empower non-expert users to explore and mitigate risk in their data to protect the privacy of individuals and groups.

With funding from the Department of Defense and Lincoln Laboratory’s New Technologies Initiative (NTI), a team of researchers developed the Mitigation Application and Assessment (MASq) software tool which provides situational awareness to data owners and mission stakeholders about the disclosure risk contained within their dataset and provides methods for mitigating said risk. MASq combines standard and novel techniques in Artificial Intelligence and Statistical Disclosure Control (SDC) to facilitate some – or all – of the procedures and workflows associated with data de-identification prior to release, including:

* identifying which data elements reveal an organization’s activities as a group in their dataset, thus creating risk for their mission
* providing a comprehensive collection of mitigation techniques for generalizing or suppressing elements within their dataset
* providing quantitative metrics, grounded in well-supported literature, to evaluate the disclosure risk contained within a dataset and the information loss associated with mitigations (modifications) made to the dataset which reduce disclosure risk.
Furthermore, MASq can automate the aforementioned procedures by applying hundreds of combinations of mitigations, evaluating their impact with respect to disclosure risk and information loss, and generating a report which ranks the most effective mitigations strategies identified for a particular dataset. To date, MASq has been transitioned to a government sponsor and is in the process of being released as an open-source software package.

## Link to User Guide

[PyMASq User Guide](https://github.com/mit-ll/pymasq/wiki)

## Installating from Git

```sh
pip install .
git clone git@github.com:mit-ll/pymasq.git
cd pymasq
```

### Installing into a Conda Environment

```sh
conda create -n masq python=3.8 -y
conda activate masq
pip install .
```

To generate the docs

```bash
python -m pip install -r ./doc-requirements.txt
```

<p align="center">
    <img src="./assets/images/Lincoln_Lab_icon.png" width="150px"/>
</p>

## Distribution Statement

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2021 Massachusetts Institute of Technology.

Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: MIT

This material is based upon work supported under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Air Force.

The software/firmware is provided to you on an As-Is basis
