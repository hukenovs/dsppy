"""
------------------------------------------------------------------------

Description   :

    Cascaded Integrator-Comb (CIC) filter is an optimized class of
    finite impulse response (FIR) filter.
    CIC filter combines an interpolator or decimator, so it has some
    parameters:

    R - decimation or interpolation ratio,
    N - number of stages in filter (or filter order)
    M - number of samples per stage (1 or 2)*

    * for this realisation of CIC filter just leave M = 1.

    CIC filter is used in multi-rate processing. In hardware
    applications CIC filter doesn't need multipliers, just only
    adders / subtractors and delay lines.

    Equation for 1st order CIC filter:
    y[n] = x[n] - x[n-RM] + y[n-1].

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

# Title         : Cascaded Integrator-Comb (CIC)
# Author        : Alexander Kapitanov
# E-mail        :
# Company       :

import numpy as np


class CicFilter:
    """
    Cascaded Integrator-Comb (CIC) filter is an optimized class of
    finite impulse response (FIR) filter.

    Parameters
    ----------
    x : np.ndarray
        input signal
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def decimator(self, r: int, n: int):
        """
        CIC decimator: Integrator + Decimator + Comb

        Parameters
        ----------
        r : int
            decimation rate
        n : int
            filter order
        """

        # integrator
        y = self.x[:]
        for i in range(n):
            y = np.cumsum(y)

        # decimator

        y = y[::r]
        # comb stage
        return np.diff(y, n=n, prepend=np.zeros(n))

    def interpolator(self, r: int, n: int, mode: bool = False):
        """
        CIC inteprolator: Comb + Decimator + Integrator

        Parameters
        ----------
        r : int
            interpolation rate
        n : int
            filter order
        mode : bool
            False - zero padding, True - value padding.
        """

        # comb stage
        y = np.diff(self.x, n=n,
                    prepend=np.zeros(n), append=np.zeros(n))

        # interpolation
        if mode:
            y = np.repeat(y, r)
        else:
            y = np.array([i if j == 0 else 0 for i in y for j in range(r)])

        # integrator
        for i in range(n):
            y = np.cumsum(y)

        if mode:
            return y[1:1 - n * r]
        else:
            return y[r - 1:-n * r + r - 1]


# main function
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_cic_filter():

        # Number os samples
        n1 = 160
        n2 = 10

        t1 = np.linspace(0, 1, n1)
        t2 = np.linspace(0, 1, n2)

        # Decimation / Interpolation
        flt_r = [3, 5, 10]
        flt_n = [2, 4, 4]

        # Decimator
        x1 = np.sin(4 * np.pi * t1)
        clf = CicFilter(x1)
        zdec = [clf.decimator(flt_r[i],
                              flt_n[i]
                              ) for i in range(3)]

        # Interpolator
        x2 = np.sin(4 * np.pi * t2)
        clf = CicFilter(x2)
        zint = [clf.interpolator(flt_r[i],
                                 flt_n[i],
                                 mode=False
                                 ) for i in range(3)]

        # Plot figure
        plt.figure(figsize=(12, 6), dpi=80)
        for i in range(2):
            plt.subplot(4, 2, 1+i)
            plt.title('Decimator' if i == 0 else 'Interpolator')
            plt.plot(x1 if i == 0 else x2, '-', color='C0')
            plt.xlim([0, n1 if i == 0 else n2-1])
            plt.grid(True)

            for j in range(3):
                plt.subplot(4, 2, 3+2*j+i)
                plt.stem(zdec[j] if i == 0 else zint[j],
                         use_line_collection=True,
                         linefmt='C2',
                         basefmt='C0',
                         label=f'R = {flt_r[j]}, N = {flt_n[j]}'
                         )
                plt.grid(True)
                plt.legend(loc='upper right')
        plt.tight_layout(True)
        plt.show()

    test_cic_filter()
