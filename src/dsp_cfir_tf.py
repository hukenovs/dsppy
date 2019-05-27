"""
------------------------------------------------------------------------

Title         : dsp_cfir_tf.py
Author        : Alexander Kapitanov
E-mail        : sallador@bk.ru
Lang.         : python
Company       :
Release Date  : 2019/05/24

------------------------------------------------------------------------

Description   :
    Test Complex filtration of signal by using two algorithms:
    time method and frequency method.
    Input sequence is a long duration chirp signal, output sequence is
    filtered data. It is a narrow chirp pulse of greatly increased
    amplitude.

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

import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftshift
from src.signalgen import *

# #####################################################################
# Input parameters
# #####################################################################
# Number of sample points
NFFT = 2**9                 # FFT points
Tsig = 1.0 / NFFT           # Time set

# Chirp parameters
Asig = 1.0                  # Signal magnitude
Fsig = 20.0                 # Signal frequency
beta = 0.25                 # Normalized magnitude for 1st sine

# Noise parameters (Normal Gaussian distribution)
mean = 0.00                 # Mean value (DC bias for signals)
std = 0.101                 # Standard deviation of the distribution

SNR = 20
# #####################################################################

imit_data = signal_chirp(amp=Asig, period=NFFT, beta=beta, is_complex=False, is_modsine=True)
awgn_data = noise_gauss(mean=mean, std=std, period=NFFT)

calc_data = calc_awgn(sig=imit_data, snr=SNR)

fft_signal = fft(calc_data)
fft_abssig = abs(fft_signal)
fft_shifts = fftshift(fft_abssig)

plt.figure('Pass chirp signal from filter (time / freq methods)')
plt.subplot(2, 1, 1)
plt.plot(calc_data)
plt.title('Signal')
plt.grid()
plt.xlabel('time')
plt.ylabel('Magnitude')
plt.subplot(2, 1, 2)
plt.plot(fft_abssig)
plt.title('Signal')
plt.grid()
plt.xlabel('time')
plt.ylabel('Magnitude')
plt.show()
