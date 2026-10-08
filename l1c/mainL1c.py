
# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c


# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\EODP_NOEL\\auxiliary'
# GM dir + L1B dir
indir = 'C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-L1C\\input\\gm_alt100_act_150,C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-L1C\\input\\l1b_output'
outdir = r"C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-L1C\\output_noel"


# Initialise the ISM
myL1c = l1c(auxdir, indir, outdir)
myL1c.processModule()
