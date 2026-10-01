from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF

        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hdet * Hsmear * Hmotion * Hwfe # dummy

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO
        eps = 1e-6

        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w



        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        fnAct = fAct / (1/w)
        fnAlt = fAlt / (1/w)

        [fnAltxx, fnActxx] = np.meshgrid(fnAlt, fnAct, indexing='ij')  # Please use ‘ij’ indexing or you
        fn2D = np.sqrt(fnAltxx * fnAltxx + fnActxx * fnActxx)

        fc = D / (lambd * focal)
        fr2D = fn2D / (fc)*(1 / w)



        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        Hdiff = np.zeros_like(fr2D)

        ind = fr2D <= 1
        Hdiff[ind] = (2 / pi) * (np.arccos(fr2D[ind])- fr2D[ind] * np.sqrt(1 - fr2D[ind]**2))

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        x = pi * defocus * fr2D * (1 - fr2D)

        Hdefoc = np.ones_like(fr2D)
        nonzero = x != 0
        Hdefoc[nonzero] = 2 * j1(x[nonzero]) / x[nonzero]
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        Hwfe = np.exp(-fr2D * (1 - fr2D) * (kLF * (wLF / lambd) ** 2 + kHF * (wHF / lambd) ** 2))

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO
        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        #TODO
        HsmearAlt = np.sinc(ksmear * fnAlt)
        Hsmear = repmat(HsmearAlt.reshape(-1,1), 1, ncolumns)
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO
        Hmotion = np.sinc(kmotion * fn2D)


        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
               Plotting the system MTF and all of its contributors
               :param Hdiff: Diffraction MTF
               :param Hdefoc: Defocusing MTF
               :param Hwfe: Wavefront electronics MTF
               :param Hdet: Detector MTF
               :param Hsmear: Smearing MTF
               :param Hmotion: Motion blur MTF
               :param Hsys: System MTF
               :param nlines: Number of lines in the TOA
               :param ncolumns: Number of columns in the TOA
               :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
               :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
               :param directory: output directory
               :param band: band
               :return: N/A
               """
        # TODO

        """Plotting the system MTF and all of its contributors."""
        mid_line = nlines // 2
        mid_col = ncolumns // 2

        mtf_dict = {
            "Diffraction MTF": Hdiff,
            "Defocus MTF": Hdefoc,
            "WFE Aberrations MTF": Hwfe,
            "Detector MTF": Hdet,
            "Smearing MTF": Hsmear,
            "Motion blur MTF": Hmotion,
            "System MTF": Hsys,
        }

        # Como hay que hacer 2 plot, definimos una función para representar ALT y ACT
        def _plot_slice(x_freq, direction_label, filename):
            plt.figure(figsize=(9, 6))

            # Manejo de dimensiones de frecuencia
            if x_freq.ndim == 2:
                x_vec = (
                    x_freq[mid_line, :]
                    if direction_label == "ACT"
                    else x_freq[:, mid_col]
                )
            else:
                x_vec = x_freq

            # Filtrar solo frecuencias no negativas (hasta Nyquist u otro límite)
            mask = x_vec >= 0
            sort_idx = np.argsort(x_vec[mask])
            x_plot = x_vec[mask][sort_idx]

            # Representación de cada uno de los mtf
            for label, h_data in mtf_dict.items():
                if h_data.ndim == 2:
                    slice_data = (
                        h_data[mid_line, :]
                        if direction_label == "ACT"
                        else h_data[:, mid_col]
                    )
                else:
                    slice_data = h_data

                y_plot = slice_data[mask][sort_idx]

                if label == "System MTF":
                    plt.plot(
                        x_plot, y_plot, label=label, color="black", linewidth=2.0
                    )
                else:
                    plt.plot(x_plot, y_plot, label=label, linewidth=1.5, alpha=0.85)

            # Línea vertical de Nyquist en 0.5
            plt.axvline(
                x=0.5, color="black", linestyle="--", label="f Nyquist", linewidth=1.5
            )

            plt.title(f"System MTF - slice {direction_label}")
            plt.xlabel("Spatial frequencies f/(1/w) [-]")
            plt.ylabel("MTF")
            plt.xlim(0.0, 0.51)
            plt.ylim(-0.02, 1.05)
            plt.grid(True, linestyle="-", alpha=0.4)
            plt.legend(loc="lower left", fontsize="small")
            plt.tight_layout()

            # Guardar en directorio original
            plt.savefig(os.path.join(directory, filename), dpi=300)

            # Mostrar imagen
            plt.show()

        # Se llama a la función anterior creada para los dos plot
        _plot_slice(fnAct, "ACT", "mtf_act.png")
        _plot_slice(fnAlt, "ALT", "mtf_alt.png")
