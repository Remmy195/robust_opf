"""Risk-aware optimal power flow by cutting planes.

Companion code for the manuscript.  The entry point is `ropf.cli`; the pieces it
composes are:

    network    MATPOWER case files to the network model
    risk       the risk functionals of Section 2.3 and their separation
    model      the AMPL boundary: (M), (M^ac) and (D)
    algorithm  Algorithm 1 and the Section 3.4 AC stage
    config     the configuration file and its key table
    results    the three artifacts a run leaves behind
    log        the run transcript

    counterfactual  Section 4: disfigurements, the frequency screen, and (D)
    study           the ladder study driver
    data            fetching the ACTIVSg cases
"""

__version__ = "1.0.0"
