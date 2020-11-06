"""
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

# Title         : Ultra-long FFT
# Author        : Alexander Kapitanov
# E-mail        :
# Company       :

from datetime import datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

N1 = 8  # Colomn (FFT1)
N2 = 16  # Rows (FFT2)


# Ultra-long FFT function
def calc_ulfft(sig: np.ndarray, n1: int = 32, n2: int = 32) -> np.ndarray:
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
    t_d = np.reshape(
        np.array([np.exp(-1j * 2 * np.pi * (k1 * k2) / (n1 * n2)) for k1 in range(n1) for k2 in range(n2)]), (n1, n2)
    )

    # 1 Step: Shuffle 0
    s_d = np.array([sig[k2 * n1 + k1] for k1 in range(n1) for k2 in range(n2)])
    # 2 Step: Calculate FFT0
    f_d = np.array([fft(s_d[n2 * k1 : n2 * (k1 + 1)]) for k1 in range(n1)])
    # 3 Step: Complex multiplier
    s_d = np.reshape(np.array(f_d * t_d), n1 * n2)
    # 4 Step: Shuffle 1
    s_d = np.array([s_d[k1 * n2 + k2] for k2 in range(n2) for k1 in range(n1)])
    # 5 Step: Calculate FFT1
    f_d = np.array([fft(s_d[n1 * k2 : n1 * (k2 + 1)]) for k2 in range(n2)])
    # 6 Step: Shuffle 2
    s_d = np.reshape(np.array(f_d), n1 * n2)
    # Output result
    return np.array([s_d[k2 * n1 + k1] for k1 in range(n1) for k2 in range(n2)])


class SignalGenerator:
    def __init__(
        self, nfft: int, freq: float, alpha: float = 0.01,
    ):
        """Generate some useful kind of signals: harmonic or linear freq. modulated.

        Parameters
        ----------
        nfft : int
            Total lenght of FFT (NFFT = N1 * N2).
        freq : float
            Signal frequency.
        alpha : float
            Add Gaussian noise if alpha not equal to zero. Should be positive.

        """

        self.nfft = nfft
        self.freq = freq
        self.alpha = alpha

    def awgn(self):
        np.random.seed(42)
        return self.alpha * np.random.rand(self.nfft)

    def input_harmonic(self):
        """Generate input singal"""
        return (
            np.array([1 + 1j if i in (self.freq, self.nfft - self.freq) else 0 for i in range(self.nfft)]) + self.awgn()
        )

    def input_linfreq(self):

        tt = np.linspace(0, 1, self.nfft, endpoint=False)

        sig_re = np.cos(self.freq * tt ** 2 * np.pi) * np.sin(tt * np.pi) + self.awgn()
        sig_im = np.sin(self.freq * tt ** 2 * np.pi) * np.sin(tt * np.pi) + self.awgn()
        return sig_re + 1j * sig_im


class UltraLongFFT:
    """Calculate ultra-long FFT via 2D FFT scheme with 3 shufflers: step by step.

    Parameters
    ----------
    n1 : int
        Rows (number of points for 1st FFTs)
    n2 : int
        Columns (number of points for 2ns FFTs)

    where NFFT = N1 * N2
    """

    _plt_titles = (
        "1. Input Signal",
        "2. Shuffle [1]",
        "3. FFT1, N1",
        "4. Shuffle [2]",
        "5. Twiddles",
        "6. Complex Multiplier",
        "7. FFT2, N2",
        "8. Shuffle [3]. Output",
    )

    def __init__(self, n1: int = 32, n2: int = 32):
        self.n1 = n1
        self.n2 = n2
        self.__nfft = n1 * n2

    @property
    @lru_cache(maxsize=4)
    def twiddles(self):
        """Twiddle factors for rotating internal sequence."""
        twd = np.array(
            [np.exp(-1j * 2 * np.pi * (k1 * k2) / self.__nfft) for k1 in range(self.n1) for k2 in range(self.n2)]
        )
        return np.reshape(twd, (self.n1, self.n2))

    def fft_calculate(self, signal: np.ndarray) -> np.ndarray:
        """Calculate signals for each stage of Ultra-long FFT Algorithm

        Parameters
        ----------
        signal : np.ndarray
            Input signal. Can be complex.

        Returns
        -------
        np.ndarray
            List of arrays for each stage of Ultra-long FFT.
        """

        # ---------------- ULFFT Algorithm ----------------
        # 1 Step: Shuffle input sequence
        sh0_data = np.reshape(
            a=np.array([signal[k2 * self.n1 + k1] for k1 in range(self.n1) for k2 in range(self.n2)]),
            newshape=(self.n1, self.n2),
        )
        # 2 Step: Calculate FFT N1 and shuffle
        res_fft0 = np.array([fft(sh0_data[k1, ...]) for k1 in range(self.n1)])
        # 3 Step: Complex multiplier
        cmp_data = res_fft0 * self.twiddles
        # 4 Step: Calculate FFT N2 and shuffle
        res_fft1 = np.array([fft(cmp_data[..., k2]) for k2 in range(self.n2)])

        # Internal Sequences:
        return np.vstack(
            [
                signal,
                sh0_data.reshape(-1, self.__nfft),
                res_fft0.reshape(-1, self.__nfft),
                res_fft0.T.reshape(-1, self.__nfft),
                self.twiddles.reshape(-1, self.__nfft),
                cmp_data.T.reshape(-1, self.__nfft),
                res_fft1.reshape(-1, self.__nfft),
                res_fft1.T.reshape(-1, self.__nfft),
            ]
        )

    def plot_result(self, data: np.ndarray, save_fig: bool = False):
        """Plot signals for each stage of Ultra-long FFT Algorithm"""
        _ = plt.figure("Ultra-long FFT", figsize=(16, 12))
        for i, v in enumerate(data):
            plt.subplot(4, 2, i + 1)

            plt.plot(v.real, linewidth=1.15, color="C2")
            plt.plot(v.imag, linewidth=1.15, color="C4")
            plt.title("1", fontsize=14)
            plt.grid(True)
            plt.xlim([0, self.__nfft - 1])

        plt.tight_layout()
        if save_fig:
            plt.savefig(f"figure_{datetime.now()}"[:-7])
        plt.show()


if __name__ == "__main__":
    _input = SignalGenerator(N1 * N2, freq=16, alpha=0.001)
    # _array = _input.input_harmonic(freq=2)
    _array = _input.input_linfreq()
    _ulfft = UltraLongFFT(N1, N2)
    _datas = _ulfft.fft_calculate(_array)
    _ulfft.plot_result(_datas, save_fig=True)
