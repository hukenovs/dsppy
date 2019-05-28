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
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

from src.signalgen import signal_chirp, calc_awgn

# #####################################################################
# Input parameters
# #####################################################################
# Number of sample points
NFFT = 2**9                 # FFT points
Tsig = 1.0 / NFFT           # Time set

# Chirp parameters
Asig = 1.0                  # Signal magnitude
Fsig = 15.0                 # Signal frequency
Beta = 0.50                 # Normalized magnitude for 1st sine

# Noise parameters (Normal Gaussian distribution)

SNR = 20
# #####################################################################

imit_data = signal_chirp(amp=Asig, beta=Beta, period=NFFT, is_complex=True, is_modsine=True)
calc_data = calc_awgn(sig=imit_data, snr=SNR)

sfun_data = signal_chirp(amp=Asig, freq=Fsig, beta=Beta, period=NFFT, is_complex=True, is_modsine=True)

fft_signal = fft(calc_data)
fft_real = fft_signal.real / np.max(np.abs(fft_signal.real))
fft_imag = fft_signal.imag / np.max(np.abs(fft_signal.imag))

fft_abssig = np.abs(fft_signal)
fft_logscl = 20*np.log10(fft_abssig / np.max(np.abs(fft_abssig)))

fft_sfunc = fft(sfun_data)
fft_sconj = np.conj(fft_sfunc)

comp_mult = fft_signal * fft_sconj
comp_real = comp_mult.real # / np.max(np.abs(comp_mult.real))
comp_imag = comp_mult.imag # / np.max(np.abs(comp_mult.imag))

ifft_signal = ifft(comp_mult)

ifft_real = ifft_signal.real
ifft_imag = ifft_signal.imag

# plt_fonts = {
#     'family': 'cursive',
#     'style': 'italic',
#     'size': 10}
# plt.rc('font', **plt_fonts)

plt.figure('Pass chirp signal from filter (time / freq methods)')
plt.subplot(2, 2, 1)
plt.plot(calc_data.real)
plt.plot(calc_data.imag)
plt.title('Input Chirp Signal')
plt.grid()
plt.xlabel('time')
plt.ylabel('Magnitude')

plt.subplot(2, 2, 2)
plt.plot(fft_real)
plt.plot(fft_imag)
plt.title('Chirp Spectrum (I/Q)')
plt.grid()
plt.xlabel('freq')
plt.ylabel('Magnitude')

plt.subplot(2, 2, 3)
plt.plot(comp_real)
plt.plot(comp_imag)
plt.title('Signal after complex multiplier')
plt.grid()
plt.xlabel('freq')
plt.ylabel('Magnitude')

plt.subplot(2, 2, 4)
plt.plot(ifft_real)
plt.plot(ifft_imag)
plt.title('Output data (freq method)')
plt.grid()
plt.xlabel('freq')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
