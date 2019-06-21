"""
------------------------------------------------------------------------

Title         : dsp_ulfft.py
Author        : Alexander Kapitanov
E-mail        : sallador@bk.ru
Lang.         : python
Company       :
Release Date  : 2019/05/28

------------------------------------------------------------------------

Description   :
    Ultra-long FFT calculation via 2D-FFT method (N1- and N2-length
    Radix-2 FFT with multiplier on twiddle factor before second stage)

Datapath:
    In -> Shuffle0 -> FFT0 -> Shuffle1 -> Twd -> FFT1 -> Shuffle2 -> Out

    > Shuffle 0, 1, 2 - mix data between rows and colomns.
    > FFT0, FFT1 - partial FFTs (by Rows, by Colomns)
    > Twd - complex multiplying data w/ twiddle factor (sine, cosine)

    Total samples : NFFT = N1 × N2

    Parameters :
    ----------
    N1 : int
        Number of FFT0 points (first stage)
    N2 : int
        Number of FFT1 points (second stage)
    NFFT : int
        Total number of FFT points = N1 * N2

    ----------
    Shuffle reordering example:

    Parameters: N1 = N2 = 4, NFFT = 16
    Input 1D array:  ( 0 1 2 3 4 5 6 7 8 9 A B C D E F )
    or
    Input 2D array: (
        0 1 2 3
        4 5 6 7
        8 9 A B
        C D E F )

    Output 2D array: (
        0 4 8 C
        1 5 9 D
        2 6 A E
        3 7 B F )

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


# Ultra-long FFT function
def calc_ulfft(sig, n1=32, n2=32):
    """
    Calculate Ultra-Long FFT

    Parameters
    ----------
    sig : ndarray
        One-dimensional input array, can be complex
    n1 : int
        Rows (number of 1st FFTs)
    n2 : int
        Columns (number of 2ns FFTs), so NFFT = N1 * N2

    """
    # Twiddle factor:
    t_d = np.reshape(np.array([
        np.exp(-1j * 2 * np.pi * (k1 * k2) / (n1 * n2))
        for k1 in range(n1) for k2 in range(n2)
    ]), (n1, n2))

    # 1 Step: Shuffle 0
    s_d = np.array([sig[k2*n1+k1] for k1 in range(n1) for k2 in range(n2)])
    # 2 Step: Calculate FFT0
    f_d = np.array([fft(s_d[n2*k1:n2*(k1+1)]) for k1 in range(n1)])
    # 3 Step: Complex multiplier
    s_d = np.reshape(np.array(f_d * t_d), n1 * n2)
    # 4 Step: Shuffle 1
    s_d = np.array([s_d[k1*n2+k2] for k2 in range(n2) for k1 in range(n1)])
    # 5 Step: Calculate FFT1
    f_d = np.array([fft(s_d[n1*k2:n1*(k2+1)]) for k2 in range(n2)])
    # 6 Step: Shuffle 2
    s_d = np.reshape(np.array(f_d), n1*n2)
    # Output result
    return np.array([s_d[k2*n1+k1] for k1 in range(n1) for k2 in range(n2)])


# Input parameters
# Number of sample points
N1 = 32                     # Colomn (FFT1)
N2 = 32                     # Rows (FFT2)
NFFT = N1 * N2              # Number of total FFT points
# Signal parameters
FSIG = 2                    # Signal frequency


# Main section: Calculate
def fft_calc(n1=32, n2=32):
    """
    Main function: calculate ultra-long FFT step by step

    Parameters
    ----------
    n1 : int
        Rows (number of 1st FFTs)
    n2 : int
        Columns (number of 2ns FFTs), so NFFT = N1 * N2

    """
    sh_dat = np.zeros((8, NFFT), dtype=complex)
    # Input signal:
    sh_dat[0] = np.array(
        [1+1j if i in (FSIG, NFFT-FSIG) else 0 for i in range(NFFT)]
    )
    # Twiddles for complex multiplier:
    twd_data = np.reshape(np.array([
        np.exp(-1j * 2 * np.pi * (k1 * k2) / (n1 * n2))
        for k1 in range(n1) for k2 in range(n2)
    ]), (n1, n2))

    # 1 Step: Shuffle [0] - input sequence
    sh0_data = np.reshape(
        a=np.array(
            [sh_dat[0][k2*n1 + k1] for k1 in range(n1) for k2 in range(n2)]
        ),
        newshape=(n1, n2)
    )
    # 2 Step: Calculate FFT0 and shuffle
    res_fft0 = np.array([fft(sh0_data[k1, ...]) for k1 in range(n1)])
    # 3 Step: Complex multiplier
    cmp_data = res_fft0 * twd_data
    # 4 Step: Calculate FFT1 and shuffle
    res_fft1 = np.array([fft(cmp_data[..., k2]) for k2 in range(n2)])

    # Internal Sequences:
    sh_dat[1] = np.array(
        [sh0_data[k1, k2] for k1 in range(n1) for k2 in range(n2)]
    )
    sh_dat[2] = np.array(
        [res_fft0[k1, k2] for k1 in range(n1) for k2 in range(n2)]
    )
    sh_dat[3] = np.array(
        [res_fft0[k1, k2] for k2 in range(n2) for k1 in range(n1)]
    )
    sh_dat[4] = np.array(
        [twd_data[k1, k2] for k1 in range(n1) for k2 in range(n2)]
    )
    sh_dat[5] = np.array(
        [cmp_data[k1, k2] for k2 in range(n2) for k1 in range(n1)]
    )
    sh_dat[6] = np.array(
        [res_fft1[k1, k2] for k1 in range(n1) for k2 in range(n2)]
    )
    sh_dat[7] = np.array(
        [res_fft1[k1, k2] for k2 in range(n2) for k1 in range(n1)]
    )

    # Plot titles:
    plt_titles = (
        '1. Input Signal',
        '2. Shuffle [0]',
        '3. FFT0, n1 dots',
        '4. Shuffle [1]',
        '5. Twiddle (Exp)',
        '6. Complex Multiplier',
        '7. FFT1, n2 dots',
        '8. Output signal'
    )

    # Plot results:
    plt.figure('Ultra-long FFT')
    for i in range(8):
        plt.subplot(4, 2, i+1)
        plt.plot(sh_dat[i].real, linewidth=1.15)
        plt.plot(sh_dat[i].imag, linewidth=1.15)
        plt.title(plt_titles[i])
        plt.grid()
        plt.xlim([0, NFFT-1])

    plt.tight_layout()
    plt.show()


# Execute main function: Ultra-long FFT
fft_calc(n1=N1, n2=N2)
