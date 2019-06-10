"""
------------------------------------------------------------------------

Title         : dsp_fir.py
Author        : Alexander Kapitanov
E-mail        : sallador@bk.ru
Lang.         : python
Company       :
Release Date  : 2019/06/08

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
IS_HDR = False            # Create *.H file (header) if True

NFFT = 2**13              # FFT points for Spectrum

# #####################################################################
# Main section: Calculate
# #####################################################################

# Calculate cutoff frequency before and after decimation:
Fc = 0.5 / R
Fr = Fo / R

flin = np.linspace(0, 1, NFFT//2)
fhlf = np.linspace(0, 0.5, NFFT//2)

# Calculate CIC responce:
HCIC = np.zeros(NFFT)

for i in range(NFFT):
    if i == 0:
        HCIC[i] = 1
    else:
        HCIC[i] = R**(-N) * M**N * np.abs(np.sin(np.pi * M * R * i/NFFT) /
                                          np.sin(np.pi * i/NFFT))**N

HCIC_DB = 20 * np.log10(HCIC)

# Calculate Ideal FIF filter responce:
# NB! You can change FIR corrector freq bandwidth. Just multiply sine
# arguments for the next equation by 2.
FIR_IDL = np.zeros(NFFT//2)
for i in range(NFFT//2):
    if i < int(Fo * NFFT):
        if i == 0:
            FIR_IDL[i] = 1
        else:
            FIR_IDL[i] = np.abs(M * R * np.sin(np.pi * i/R/NFFT)
                                / np.sin(np.pi * M * i/NFFT)) ** N
    else:
        FIR_IDL[i] = 0

# Calculate FIR by using firwin2() function:
FIR_COR = signal.firwin2(numtaps=NFIR,
                         freq=flin,
                         gain=FIR_IDL,
                         window=('kaiser', BETA)
                         )

FIR_COR /= np.max(FIR_COR)

# Calculate freq responce for FIR filter
FIR_FFT = np.abs(fft(FIR_COR, int(np.ceil(NFFT/R))))
FIR_FFT[FIR_FFT == 0] = 1e-10

FIR_DB2 = 20 * np.log10(FIR_FFT)
FIR_DB2 -= np.max(FIR_DB2)

FIR_REP = np.tile(FIR_DB2, R)
FIR_DIF = HCIC_DB + FIR_REP
FIR_DIF -= np.max(FIR_DIF)

if IS_COE:
    FIX_COE = np.round(FIR_COR * ((2 ** NWDT - 1) - 1))
    with open('fir_taps.coe', 'w') as fl:
        for el in FIX_COE:
            fl.write("%s\n" % int(el))
        fl.close()

# Passband irregularity: calculate mean value and freq error
PBAND = int(np.ceil(0.9*Fr*FIR_DIF.size))
IBAND = FIR_DIF[2:PBAND]
IMEAN = np.mean(IBAND) * np.ones(int(NFFT/2/R))
IBSTD = 0.5 * (np.max(IBAND) - np.min(IBAND))

# #####################################################################
# Plot results
# #####################################################################
plt.figure('FIR Filter Compensator ideal response')
plt.subplot(3, 2, 1)
plt.plot(np.linspace(0, 0.5, NFFT//2), FIR_IDL, linewidth=0.75)
plt.title('Ideal freq response')
plt.grid()
plt.xlim([0, 0.5])

plt.subplot(3, 2, 2)
plt.plot(np.linspace(-NFIR/2, NFIR/2, NFIR), FIR_COR, linewidth=0.75)
plt.title('Impulse response')
plt.grid()
plt.xlim([-NFIR/2, NFIR/2])

plt.subplot(3, 2, 3)
plt.plot(fhlf, HCIC_DB[0:NFFT//2], '-.', linewidth=0.90)
plt.plot(fhlf, FIR_REP[0:NFFT//2], '--', linewidth=0.90)
plt.plot(fhlf, FIR_DIF[0:NFFT//2], '-', linewidth=1.20)
plt.title('CIC, FIR, SUM')
plt.grid()
plt.xlim([0, 0.5])
plt.ylim([-120, 5])

plt.subplot(3, 2, 4)
plt.plot(fhlf, HCIC_DB[0:NFFT//2], '-.', linewidth=0.90)
plt.plot(fhlf, FIR_REP[0:NFFT//2], '--', linewidth=0.90)
plt.plot(fhlf, FIR_DIF[0:NFFT//2], '-', linewidth=1.20)
plt.title('FIR order = %d' % NFIR)
plt.grid()
plt.xlim([0, 0.5/R])
plt.ylim([-120, 5])

plt.subplot(3, 2, 5)
plt.plot(fhlf, FIR_DIF[0:NFFT//2], '--', linewidth=0.75)
plt.plot(IMEAN, '-', linewidth=1.00, label="Mean = %0.4f \nError = %0.4f" % (IMEAN[0], IBSTD))
plt.title('Passband irregularity')
plt.grid()
plt.xlim([0, Fr])
plt.ylim([np.min(IBAND), np.max(IBAND)])
plt.legend(loc=3, ncol=1, borderaxespad=0.)

plt.tight_layout()
plt.show()
