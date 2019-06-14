"""
------------------------------------------------------------------------

Title         : dsp_maf.py
Author        : Alexander Kapitanov
E-mail        : sallador@bk.ru
Lang.         : python
Company       :
Release Date  : 2019/06/11

------------------------------------------------------------------------

Description   :
    Moving average filter (MAF) as simple FIR filter.

    Input and output data: one-dimensional sequences. M - moving average
    filter parameter.
    Frequency response of the moving average filter is:

    >> H[f] = sin(pi * f * M) / (M * sin(pi * f)).

    Note, H[0] = 1.

    Parameters :
    ----------
    M : int
        moving average filter parameter
    N : int
        Number of samples

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
from scipy.fftpack import fft

N = 512             # Number of samples
M = (3, 8, 24)      # Moving average step

LM = len(M)         # Size of M


# Moving average function
def maf(sig, m=2):
    """
    Calculate moving average filter

    Parameters
    ----------
    sig : ndarray
        one-dimensional input array.
    m : int
        moving average step

    """
    coe = np.ones(m) / m
    return np.convolve(sig, coe, mode='same')


# Main section: Calculate
def maf_calc():
    """
    Main function: calculate moving average filter

    """

    # Input signal w/ noise:
    mag = 5         # Magnitude

    # This is my signal switcher. Change it for your purposes
    # if 0:
    #     k = 16  # Number of repeats
    #     sig = np.tile(
    #         A=np.concatenate((np.zeros(int(N/k)), mag*np.ones(int(N/k)))),
    #         reps=int(k/2)
    #     )
    # else:
    #     sig = np.concatenate(
    #         (
    #             np.zeros(int(N/2)),
    #             mag * np.ones(int(N/4)),
    #             np.zeros(int(N/2))
    #         )
    #     )

    sig = np.concatenate(
        (
            np.zeros(int(N/2)),
            np.ones(int(N/4)) * mag,
            np.zeros(int(N/2)))
    )

    lns = sig.size  # Size of signal

    # Add some noise and peaks
    np.random.seed(1)
    sig += np.random.randn(lns)             # Add Gaussian noise
    rnd = np.random.randint(0, lns, 15)     # Add random numbers for index
    sig[rnd] = 2*mag                        # Add peaks

    # Calculate Moving Average filter:
    res = np.zeros((lns, LM))
    for i in range(LM):
        res[:, i] = maf(sig, m=M[i])

    # The second way for calculating over 2d array:
    # for i, j in enumerate(res.T):
    #     res[:, i] = maf(sig, m=M[i])

    # Calculate Frequency responce:
    hfq = np.zeros((lns, LM))
    for j in range(LM):
        for i in range(lns):
            if i == 0:
                hfq[i, j] = 1
            else:
                hfq[i, j] = np.abs(np.sin(np.pi * M[j] * i / 2 / lns) / M[j] /
                                   np.sin(np.pi * i / 2 / lns))

    # Calculate spectrum of input signal:
    fft_sig = np.abs(fft(sig))
    fft_sig /= np.max(fft_sig)

    # Calculate spectrum of output signal:
    fft_out = np.zeros((lns, LM))
    for i in range(LM):
        fft_out[:, i] = np.abs(fft(res[:, i]))
        fft_out[:, i] /= np.max(fft_out[:, i])

    # Plot results:
    plt.figure('FIR Filter Compensator ideal response')
    plt.subplot(3, 2, 1)
    plt.plot(sig, linewidth=1.25)
    plt.title('Input signal')
    plt.grid()
    plt.xlim([0, lns-1])

    plt.subplot(3, 2, 3)
    for i in range(LM):
        plt.plot(hfq[:, i], linewidth=1.25, label="M=%d" % M[i])
    plt.title('MA filter responce')
    plt.grid()
    plt.legend(loc=1)
    plt.xlim([0, lns-1])

    plt.subplot(3, 2, 5)
    for i in range(LM):
        plt.plot(res[:, i], linewidth=1.0, label="M=%d" % M[i])
    plt.title('Output signal')
    plt.grid()
    plt.legend(loc=2)
    plt.xlim([0, N-1])

    for i in range(LM):
        plt.subplot(3, 2, 2*i+2)
        plt.plot(sig, '-', linewidth=0.5)
        plt.plot(res[:, i], linewidth=1.5)
        plt.title('Moving average, M = %d' % M[i])
        plt.grid()
        plt.xlim([0, lns-1])

    plt.tight_layout()
    plt.show()


# Execute Moving average function
maf_calc()
