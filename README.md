# COMP5329-assignment1
Deep learning from scratch

Dependencies:
 * scipy (for running code): `pip install scikit-learn`
 * latex (for generating report)
 * make (for running Makefile)
 * bash (for building zip file)
 * yapf, pycodestyle (for linting): `pip install yapf pycodestyle`

Usage:

    # run code
    ./assignment1.py
    
    # usage info
    ./assignment1.py -h
    
    # check code style
    make lint
    
    # generate the report and create the zip file
    make
    
    # generate just the report
    make report/report.pdf
