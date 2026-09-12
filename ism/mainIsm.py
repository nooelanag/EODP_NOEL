
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\EODP\eodp_students\auxiliary'
indir = r"C:\EODP\EODP_TER_2021_working\EODP-TS-ISM\input\gradient_alt100_act150" # small scene
outdir = r"C:\EODP\EODP_TER_2021_working\EODP-TS-ISM\myoutput"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
