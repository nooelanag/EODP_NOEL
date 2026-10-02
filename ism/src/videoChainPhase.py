
from ism.src.initIsm import initIsm
import numpy as np
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF

class videoChainPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, toa, band):
        self.logger.info("EODP-ALG-ISM-3000: Video Chain")

        # Electrons to Voltage - read-out & amplification
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-3010: Electrons to Voltage – Read-out and Amplification")
        toa = self.electr2Volt(toa,
                         self.ismConfig.OCF,
                         self.ismConfig.ADC_gain)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [V]")

        # Digitisation
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-3020: Voltage to Digital Numbers – Digitisation")
        toa = self.digitisation(toa,
                          self.ismConfig.bit_depth,
                          self.ismConfig.min_voltage,
                          self.ismConfig.max_voltage)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [DN]")

        # Plot
        if self.ismConfig.save_vcu_stage:
            saveas_str = self.globalConfig.ism_toa_vcu + band
            title_str = 'TOA after the VCU phase [DN]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa

    def electr2Volt(self, toa, OCF, gain_adc):
        """
        Electron to Volts conversion.
        Simulates the read-out and the amplification
        (multiplication times the gain).
        :param toa: input toa in [e-]
        :param OCF: Output Conversion factor [V/e-]
        :param gain_adc: Gain of the Analog-to-digital conversion [-]
        :return: output toa in [V]
        """
        #TODO
        # Read-out: electrons -> volts with the output conversion factor,
        # then amplification with the ADC gain
        toa_v = toa * OCF * gain_adc
        return toa_v

    def digitisation(self, toa, bit_depth, min_voltage, max_voltage):
        """
        Digitisation - conversion from Volts to Digital counts
        :param toa: input toa in [V]
        :param bit_depth: bit depth
        :param min_voltage: minimum voltage
        :param max_voltage: maximum voltage
        :return: toa in digital counts
        """
        #TODO
        # Saturation: the signal cannot go above the maximum voltage of the ADC
        # Saturation: the signal cannot go above the maximum voltage of the ADC
        toa = np.minimum(toa, max_voltage)

        # Quantisation: map [min_voltage, max_voltage] onto 0 ... 2^bit_depth - 1
        # and round to the nearest integer count
        levels = 2 ** bit_depth - 1
        toa_dn = np.round((toa - min_voltage) / (max_voltage - min_voltage) * levels)

        # Counts cannot be negative
        toa_dn = np.maximum(toa_dn, 0)
        return toa_dn

    #crossvalidation of the outputs
    #MTF PLot -> dimensioning MTF? Explicar las dos gráficas y decir cual es la MTF dominante
    # What are the unit conversions?
    #isrf_toa_optical [mW/(m^{2}*sr)]
    #rad2irrad -> [FACTOR]
    #IRR2PH
    #PH2E
    #E2V
    #V2DigitalNumbers
    #From central pixel:  toa[50,75]

    #Check if the central picture after the convertion we get the same outputs. PER BAND (VNIR0 to VNIR-4)
    
    
#First question:
#    - Do a crossvalidation of all the outputs, comparing the files from my own outputs ("C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-ISM\\output_noel") and the teachers outputs ("C:\\Users\\Noel\\Documents\\NOEL\\UNIVERSIDAD\\MASTER\\TELECO\\ACADEMICO\\26-27\\1er cuatri\\EODP\\FILES\\EODP_TER_2021\\EODP_TER_2021\\EODP-TS-ISM\\output"). Lets make these directories as variables so that I can modify in case they are not exactly like that. After crossvalidation is done, generate a brief report with the conclussions (if the data in each file is the same as the teachers data).
#
#Second question:
#   - Do the MTF plot. What is the dimensioning of the MTF? Explain both graphs and say which is the dominant one (and why). 
#
#Third question:
#   - Do a table checking the unit conversion:
#       * isrf_toa_optical [mW/(m^{2}*sr)]
#       * rad2irrad -> [FACTOR]
#       * IRR2PH
#       * PH2E
#       * E2V
#       * V2DigitalNumbers
#   - From the central pixel: ¿¿¿toa[50,75]???
#   - Check if the central picture after the cconvertion we get the same outputs. PER BAND (VNIR0 to VNIR4).
#
#
#
#THIS IS WHAT I UNDERSTOOD, SO SOME THINGS MIGHT BE MISTAKEN!!!