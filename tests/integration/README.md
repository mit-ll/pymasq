Integration Testing
-------------------

The integration tests will test each `pymasq.mitigation` and `pymasq.metric` available using `pymasq.optimization` procedures.

A core configuration is used to run all mitigations and metrics that have been vetted previously.
Any new functionality that is to be tested must be specified in its own configuration file.

The `template_config.yaml` file describes the expected format of the configurations.

User Guide
----------

The `integration.py` script is run via the command line.

    $ python integration.py

For top-level use:

    $ python integration.py [-h] [-v] [--test-config TEST_CFG] [-i ITERS]

### Help

Display available actions.
    
    $ python integration.py [ -h | --help ]

### Verbose

Display additional logging info to terminal. _Optional: default is False_.
    
    $ python integration.py [ -v | --verbose ]

### Test Configuration

Set the complete file path of the test configuration YAML file to use.

    $ python integration.py [ --test-config ]

### Iterations 

Set the number of `iters` to run the optimization procedures. This will **not** overwrite `iters` if set in the config file. _Optional: default is 1000000000_.

    $  python integration.py [ -i | --iters ] <int>


Configuration Files
-------------------

The integration tests run with the parameters specified in two YAML configuration files, `core_config.yaml` and `test_config.yaml`.
These files should define the configuration of all tests to be run. A third configuration file, `template_config.yaml`, is also included
and provides the schema for how proper configuration files can be defined.

- The `core_config.yaml` contains the configurations for the mitigations and metrics that
have been vetted previously. 
    > **This file should only be modified when adding a new mitigation
or metric that has already been tested.**

- The `test_config.yaml` is intended to include an example configuration file for new functionality. 
The configuration in this file will add to or update/overwrite the configuration loaded from `config_core.yaml`. Use the
`--test-config` flag to specify the file path to a different configuration file to be tested.

Note that comments to the YAML files can be included by adding "`#`" in any part of the file.

Default Behavior
----------------
If no optimization procedure is defined the in the configuration file to be tested (e.g., `test_config.yaml`), then 
only the `pymasq.optimization.ExhaustiveSearch` procedure will be run. This procedure will test 
all permutations in `pymasq.mitigations` and may be time-consuming. In this case, you can use the `--iters` flag to 
constraint the number of iterations to run.

Note that permutations are not applied and evaluated all at once, but rather incrementally. 
That is, a mitigation strategy composed of 3 mitigations, will have 6 permutations and will run for 18 iterations, 
while a mitigation strategy composed of 6 mitigations will have 720 permutations and will run for 4,320 iterations.

