Vignette Template
=================

Vignettes contain ipython notebooks that were created iteratively following the steps outlined below.

Example scenarios consist of:

1. Find a dataset (e.g., kaggle) with a properly defined regression or classification task

2. Perform cursory analysis on dataset to understand data values, relationships, patterns, etc.. Perform data cleaning as needed (e.g., remove NaNs or unecessary columns).

3. Identify a Sensitive Variable (SV) which you intend to remove and who's relationship amongst remaining data features (columns) you wish to obfuscate. Currently, a SV must be categorical/discrete.

4. Perform regression or classification task on original dataset, having removed the SV feature.

5. Perform initial Risk and/or Utility evaluations to be used as a baseline.

6. Perform Key Variable Exploration (KVE) with `pymasq` to identify features with highest correlation to SV. Determine relevant features (columns) to be considered Key Variables (KVs).

7. From here, you have the option to perform step 7 manually (7.m), automatically (7.a), or both.

    7.m. Manually mitigate KVs via trial-and-error.

        i. Select a KV and identify a relevant mitigation to apply to it based on data type, value, ranges, etc.

            >> Note that one or more KVs can be selected and one or more mitigations can be applied to them.

        ii. Perform new Risk and/or Utility evaluations on the mitigated dataset.

        iii. Compare against baseline Risk and/or Utility evaluations. 
        
        Repeat steps 7.m.i - 7.m.iii until scores are acceptable.

    7a. Auatomically mitigate KVs via optimization or search procedure.

        i. Define all KVs that can be modified during the optimization or search procedure. 

        ii. Define all mitigations and their respective parameters that can be applied during the optimization or search procedure.

        iii. Define the parameters of the optimization or search procedure and run.

8. Perform regression or classification task on modified dataset, having removed the SV feature.

    Repeat steps 7 and 8 until scores are acceptable.