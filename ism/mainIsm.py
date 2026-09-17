
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\EODP_NOEL\\auxiliary'
indir = r"C:\\Users\\Noel\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-ISM\\input\\gradient_alt100_act150" # small scene
outdir = r"C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-ISM\\output_noel"



# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
