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
    filtered data. Output signal is a narrow chirp pulse of greatly
    increased amplitude.

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
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import convolve

from src.signalgen import signal_chirp, calc_awgn, complex_minmax

# #####################################################################
# Input parameters
# #####################################################################

# Number of sample points
NFFT = 2**9                 # Number of FFT points (signal duration)

# Chirp parameters
Asig = 1.100                # Signal magnitude
Fsig = 32.00                # Signal frequency
Beta = 0.250                # Beta (bandwidth of chirp, max = 0.5)
Bstd = 1.050                # Simulate a little clipping of chirp

# Noise parameters (AWGN)
SNR = -10                   # SNR (Signal to noise ratio) in dB

# #####################################################################
# Function declaration
# #####################################################################


def filter_conv(xx, yy):
    """
    Calculate convolution of two complex signals

    Parameters
    ----------
    xx : complex
        1st one-dimensional input array.
    yy : complex
        2nd one-dimensional input array.

    """
    # Step 1: Conjugate and flip core function
    yy_inv = np.flip(np.conj(yy))
    # Step 2: Calculate partial convolution
    flt_re2re = convolve(xx.real, yy_inv.real, mode='same')
    flt_re2im = convolve(xx.real, yy_inv.imag, mode='same')
    flt_im2re = convolve(xx.imag, yy_inv.real, mode='same')
    flt_im2im = convolve(xx.imag, yy_inv.imag, mode='same')
    # Step 3: Complex conv partial operations
    flt_real = flt_re2re - flt_im2im
    flt_imag = flt_im2re + flt_re2im
    # Step 4: Flip and shift
    return fftshift(np.flip(flt_real + 1j*flt_imag))


# #####################################################################
# Main section: Calculate
# #####################################################################

# Signal + Noise, FFT
imit_data = signal_chirp(amp=Asig, freq=Fsig, beta=Bstd*Beta, period=NFFT, is_complex=True, is_modsine=True)
calc_data = calc_awgn(sig=imit_data, snr=SNR)

fft_signal = fft(calc_data, NFFT)
fft_real = fft_signal.real / np.max(np.abs(fft_signal.real))
fft_imag = fft_signal.imag / np.max(np.abs(fft_signal.imag))

fft_abs = np.abs(fft_signal)
fft_log = 20*np.log10(fft_abs / np.max(np.abs(fft_abs)))

# Sup. Function & Compl Mult
sfun_data = signal_chirp(amp=Asig, freq=0, beta=Beta, period=NFFT, is_complex=True, is_modsine=True)
fft_sfunc = np.conj(fft(sfun_data, NFFT))     # FFT and conjugate

comp_mult = fft_signal * fft_sfunc
comp_real = comp_mult.real
comp_imag = comp_mult.imag

# IFFT
ifft_signal = np.flip(ifft(comp_mult, NFFT))

# Time complex conv
time_signal = filter_conv(xx=calc_data, yy=sfun_data)

# Difference
diff_signal = (ifft_signal-time_signal) / NFFT

# #####################################################################
# Plot results
# #####################################################################

# Min and Max for Y axis
axis_inp = complex_minmax(calc_data)
axis_cmp = complex_minmax(comp_mult)
axis_res = complex_minmax(ifft_signal)
axis_tms = complex_minmax(time_signal)
axis_dff = complex_minmax(diff_signal)

# plt_fonts = {
#     'family': 'cursive',
#     'style': 'italic',
#     'size': 10}
# plt.rc('font', **plt_fonts)

plt.figure('Filter chirp signal freq method)')
plt.subplot(3, 2, 1)
plt.plot(calc_data.real)
plt.plot(calc_data.imag)
plt.title('1. Input Chirp Signal')
plt.grid()
plt.axis([0, NFFT-1, axis_inp[0], axis_inp[1]])
plt.xlabel('time')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 3)
plt.plot(fft_log)
plt.title('2. Chirp Spectrum')
plt.grid()
plt.axis([0, NFFT-1, -50, 0])
plt.xlabel('freq')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 5)
plt.plot(comp_mult.real)
plt.plot(comp_mult.imag)
plt.title('3. Complex multiplier')
plt.grid()
plt.axis([0, NFFT-1, axis_cmp[0], axis_cmp[1]])
plt.xlabel('freq')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 2)
plt.plot(ifft_signal.real)
plt.plot(ifft_signal.imag)
plt.title('4. Output (freq method)')
plt.grid()
plt.axis([0, NFFT-1, axis_res[0], axis_res[1]])
plt.xlabel('time')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.plot(time_signal.real)
plt.plot(time_signal.imag)
plt.title('5. Output (time method)')
plt.grid()
plt.axis([0, NFFT-1, axis_tms[0], axis_tms[1]])
plt.xlabel('time')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 6)
plt.plot(np.abs(diff_signal.real))
plt.plot(np.abs(diff_signal.imag))
plt.title('6. Difference error')
plt.grid()
plt.axis([0, NFFT-1, 0, axis_dff[1]])
plt.xlabel('time')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
