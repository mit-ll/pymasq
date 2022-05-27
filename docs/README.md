# Documentation README

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Air Force.

© 2021 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

## Usage Instructions for update_docs.sh

[Source](https://tgsmith61591.github.io/2018-12-23-automate-gh-builds/) for script

1. Ensure that pymasq is installed in your environment

1. Make sure that `update_docs.sh` has execute permissions  

    ```sh
    chmod +x update_docs.sh 
    ```  
  
1. Ensure that the snapshot-\<version\> branch you want to generate documentation for exists
(You do not have to currently be on that branch)  
    Example:

    ```bash
    (pymasq) workstation:docs user$ git branch
      user-0.0-1
    * user-0.0-2
      gh-pages
      master
      snapshot-0.0
      snapshot-test
    ```

1. Change to the `pymasq/docs/` directory:  
    Example:

    ```sh
    cd ./docs/
    ```

1. Run the script, passing the snapshot version number with the "-v=\<VERSION #\>" or "--version=\<VERSION #\>" flag.
(The script will not run if the snapshot version does not exist or if there is no argument passed). If hosting the
documentation on gh-pages is not required, you can pass the optional flag "-o" or "--offline"
which will generate the html documentation in `pymasq/docs/build/`.  
    Example:

    ```sh
    ./update_docs.sh -v=0.6.3-SNAPSHOT
    ```

1. View newly generated documentation [here](https://mit-ll.github.io/pymasq/index.html)
