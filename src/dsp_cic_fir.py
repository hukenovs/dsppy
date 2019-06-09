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
    NFFT : integer
        Number of FFT points (signal duration)

    Asig : float
        Signal magnitude (should be positive)
    Fsig : float
        Signal frequency (linear part of chirp)
    Beta : float
        Beta (bandwidth of chirp, max = 0.5)
    Bstd : float
        Simulate a little clipping of chirp (set it from 0.9 to 1.1)

    SNR  : float
        Signal to noise ratio [dB]

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
NFIR = 256                # Filter order, must be odd when Fo = 0.5
NWDT = 16                 # Number of bits for fixed-point coeffs
Fo = 0.22                 # Normalized Cutoff: 0.2 < Fo < 0.5;
BETA = 9                  # Beta for Kaiser Window (if win is enabled)

# Chirp parameters
IS_COE = True             # Create Xilinx *.COE file if True
IS_HDR = False            # Create *.H file (header) if True

NFFT = 2**12              # FFT points for Spectrum

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
FIR_FFT = 20 * np.log10(np.abs(fft(FIR_COR, int(np.ceil(NFFT/R)))))
FIR_FFT -= np.max(FIR_FFT)

FIR_REP = np.tile(FIR_FFT, R)
FIR_DIF = HCIC_DB + FIR_REP
FIR_DIF -= np.max(FIR_DIF)

if IS_COE:

    FIX_COE = np.round(FIR_COR * ((2 ** NWDT - 1) - 1))
    with open('fir_taps.coe', 'w') as fl:
        for el in FIX_COE:
            fl.write("%s\n" % int(el))
        fl.close()

# #####################################################################
# Plot results
# #####################################################################
# plt.figure('FIR Filter Compensator ideal response')
# plt.subplot(2, 2, 1)
# plt.plot(np.linspace(0, 0.5, NFFT//2), FIR_IDL)
# plt.title('CFIR filter ideal response')
# plt.grid()
# plt.xlim([0, 0.5])
#
# plt.subplot(2, 2, 2)
# plt.plot(np.linspace(-NFIR/2, NFIR/2, NFIR), FIR_COR)
# plt.title('FIR filter impulse response')
# plt.grid()
# plt.xlim([-NFIR/2, NFIR/2])
#
# plt.subplot(2, 2, 3)
# plt.plot(fhlf, HCIC_DB[0:NFFT//2])
# plt.plot(fhlf, FIR_REP[0:NFFT//2])
# plt.plot(fhlf, FIR_DIF[0:NFFT//2])
# plt.title('Total filter responce (Order = %d)' % NFIR)
# plt.grid()
# plt.ylim([-120, 5])
#
# plt.tight_layout()
# plt.show()
