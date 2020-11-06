"""
------------------------------------------------------------------------

Description   :
    Digital Down Converters often use CIC + FIR scheme for downsampling
    and filter input signal.
    This script calculate FIR filter compensator to correct frequency
    response after CIC filter.

    Parameters :
    ----------
    R : int
        CIC decimation factor
    N : int
        Number of stages (CIC filter order)
    M : int
        Differential delay (for FPGAs - 1 or 2)

    NFIR : int
        FIR Filter order, must be odd when Fo = 0.5
    NWDT : int
        Number of bits for fixed-point coeffs
    BETA : float
        Beta for Kaiser Window
    Fo : float
        Normalized Cutoff: 0.2 < Fo < 0.5

    NFFT : int
         Number of FFT points (spectrum duration)

    IS_COE : bool
        Create Xilinx *.COE file if True
    IS_HDR : bool
        Create *.H file (header) if True

------------------------------------------------------------------------

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (c) 2019 Kapitanov Alexander

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT
WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND
PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE
DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR
OR CORRECTION.

------------------------------------------------------------------------
"""

# Title         : CIC filter
# Author        : Alexander Kapitanov
# E-mail        :
# Company       :

from sys import version
import datetime as dt

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# #####################################################################
# Input parameters
# #####################################################################

# CIC paramteres:
R = 8                     # Decimation factor
N = 6                     # Number of stages (filter order)
M = 1                     # Differential delay (for FPGAs - 1 or 2)

# FIR paramteres:
NFIR = 128                # Filter order, must be odd when Fo = 0.5
NWDT = 16                 # Number of bits for fixed-point coeffs
BETA = 7                  # Beta for Kaiser Window
Fo = 0.22                 # Normalized Cutoff: 0.2 < Fo < 0.5;

# Chirp parameters
IS_COE = True             # Create Xilinx *.COE file if True
IS_HDR = True             # Create *.H file (header) if True

NFFT = 2**13              # FFT points for Spectrum


# Main section: Calculate
def cic_calc():
    """
    Main function: calculate fir filter compensator

    """
    # Cutoff frequency after decimation by R:
    fr = Fo / R

    flin = np.linspace(0, 1, NFFT//2)
    fhlf = np.linspace(0, 0.5, NFFT//2)

    hcic = np.zeros(NFFT)
    hfir = np.zeros(NFFT//2)

    # Calculate CIC responce:
    for i in range(NFFT):
        if i == 0:
            hcic[i] = 1
        else:
            hcic[i] = R**(-N) * M**N * np.abs(np.sin(np.pi * M * R * i/NFFT) /
                                              np.sin(np.pi * i/NFFT))**N
    hcic_db = 20 * np.log10(hcic)

    # Calculate Ideal FIF filter responce:
    # NB! You can change FIR corrector freq bandwidth. Just multiply sine
    # arguments for the next equation by 2.
    for i in range(NFFT//2):
        if i < int(Fo * NFFT):
            if i == 0:
                hfir[i] = 1
            else:
                hfir[i] = np.abs(M * R * np.sin(np.pi * i/R/NFFT)
                                 / np.sin(np.pi * M * i/NFFT)) ** N
        else:
            hfir[i] = 0

    # Calculate FIR by using firwin2() function:
    fir_win = signal.firwin2(numtaps=NFIR,
                             freq=flin,
                             gain=hfir,
                             window=('kaiser', BETA)
                             )

    fir_win /= np.max(fir_win)

    # Calculate freq responce for FIR filter
    fir_fft = np.abs(fft(fir_win, int(np.ceil(NFFT/R))))
    fir_fft[fir_fft == 0] = 1e-10  # For log(x) function

    fir_db2 = 20 * np.log10(fir_fft)
    fir_db2 -= np.max(fir_db2)

    fir_rep = np.tile(fir_db2, R)
    fir_dif = hcic_db + fir_rep
    fir_dif -= np.max(fir_dif)

    if IS_COE:
        fix_coe = np.round(fir_win * ((2 ** NWDT - 1) - 1))
        with open('fir_taps.coe', 'w') as fl:
            fl.write("; XILINX FIR filter coefficient (.COE) File\n")
            fl.write("; Generated by Python " + version[0:5] + "\n")
            fl.write("; Generated on " +
                     dt.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
            fl.write("Radix = 10;\nCoefficient_Width = %d;\n" % NWDT)
            fl.write("CoefData = \n")
            for i in range(NFIR):
                fl.write("%s" % int(fix_coe[i]))
                if i == (NFIR - 1):
                    fl.write(";\n")
                else:
                    fl.write(",\n")
            fl.close()

    if IS_HDR:
        with open('fir_taps.h', 'w') as fl:
            fl.write("/*\n * Filter Coefficients (C Source) file header\n"
                     " * Generated by Python " + version[0:5] + "\n")
            fl.write(" * Generated on " +
                     dt.datetime.now().strftime("%Y-%m-%d %H:%M") +
                     "\n */\n\n")
            fl.write("const int BL = %d;\n" % NWDT)
            fl.write("const float B[%d] = {\n" % NWDT)
            for i in range(NFIR):
                fl.write("\t%f" % fir_win[i])
                if i == (NFIR-1):
                    fl.write("\n};")
                else:
                    fl.write(",\t")
                if (i+1) % 4 == 0:
                    fl.write("\n")
            fl.close()

    # Passband irregularity: calculate mean value and freq error
    p_band = int(np.ceil(0.9 * fr * fir_dif.size))
    i_band = fir_dif[2:p_band]
    i_mean = np.mean(i_band) * np.ones(int(NFFT/2/R))
    i_stdx = 0.5 * (np.max(i_band) - np.min(i_band))

    # Plot results
    plt.figure('FIR Filter Compensator ideal response')
    plt.subplot(3, 2, 1)
    plt.plot(np.linspace(0, 0.5, NFFT//2), hfir,
             linewidth=0.75,
             label="Fo = %0.2f" % Fo)
    plt.title('Ideal freq response')
    plt.grid()
    plt.xlim([0, 0.5])
    plt.legend(loc=4)

    plt.subplot(3, 2, 2)
    plt.plot(np.linspace(-NFIR/2, NFIR/2, NFIR), fir_win,
             linewidth=0.75,
             label="N = %d" % NFIR)
    plt.title('FIR impulse response')
    plt.grid()
    plt.xlim([-NFIR/2, NFIR/2])
    plt.legend(loc=1)

    plt.subplot(3, 2, 3)
    plt.plot(np.linspace(0, 1, NFFT), hcic_db, '-',
             linewidth=0.90,
             label="R = %d\nN = %d" % (R, N))
    plt.title('CIC freq responce')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([-140, 5])
    plt.legend(loc=1)

    plt.subplot(3, 2, 4)
    plt.plot(fhlf, hcic_db[0:NFFT//2], '-.', linewidth=0.95)
    plt.plot(fhlf, fir_rep[0:NFFT//2], '--', linewidth=0.95)
    plt.plot(fhlf, fir_dif[0:NFFT//2], '-', linewidth=1.20)
    plt.title('CIC, FIR, SUM')
    plt.grid()
    plt.xlim([0, 0.5])
    plt.ylim([-120, 5])

    plt.subplot(3, 2, 5)
    plt.plot(fhlf, hcic_db[0:NFFT//2], '-.', linewidth=0.90)
    plt.plot(fhlf, fir_rep[0:NFFT//2], '--', linewidth=0.90)
    plt.plot(fhlf, fir_dif[0:NFFT//2], '-', linewidth=1.20)
    plt.title('Zoom result')
    plt.grid()
    plt.xlim([0, 0.5/R])
    plt.ylim([-120, 5])

    plt.subplot(3, 2, 6)
    plt.plot(fhlf, fir_dif[0:NFFT//2], '--', linewidth=0.75)
    plt.plot(i_mean, '-',
             linewidth=1.00,
             label="Error = %0.4f" % i_stdx)
    # label="Mean = %0.4f \nError = %0.4f" % (i_mean[0], i_stdx))
    plt.title('Passband irregularity')
    plt.grid()
    plt.xlim([0, fr])
    plt.ylim([i_mean[0]-3*i_stdx, i_mean[0]+3*i_stdx])
    plt.legend(loc=4)

    plt.tight_layout()
    plt.show()


# Execute CIC function
cic_calc()
