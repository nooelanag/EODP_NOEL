
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO + CIBER\\ACADEMICO\\26-27\\1er cuatri\\EODP\\EODP_NOEL\\auxiliary'
indir = r"C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO + CIBER\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-L1B\\input"
outdir = r"C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO + CIBER\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-L1B\\output_noel"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
