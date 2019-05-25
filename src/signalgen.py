"""
------------------------------------------------------------------------

Title         : signalgen.py
Author        : Alexander Kapitanov
E-mail        : sallador@bk.ru
Lang.         : python
Company       :
Release Date  : 2019/05/24

------------------------------------------------------------------------

Description   :
    Signal generator for digital signal processing

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


def signal_simple(amp=1.0, freq=10.0, period=100, mode='sin'):
    """
    Create simple waves: sine, cosine or complex combination

    Parameters
    ----------
    amp : float
        Signal magnitude
    freq : float
        Signal frequency
    period : integer
        Number of points for signal (same as period)
    mode : str
        Output mode: sine, cosine or complex signal
    """
    tt = freq * 2.0 * np.pi * np.linspace(0.0, 1.0, period)
    if mode == 'sin':
        return amp * np.sin(tt)
    if mode == 'cos':
        return amp * np.cos(tt)
    if mode == 'cmp':
        return amp * (np.sin(tt) + 1j*np.cos(tt))
    raise ValueError('Wrond signal mode')


def signal_am(amp=1.0, km=0.25, fc=10.0, fs=2.0, period=100):
    """
    Create Amplitude modulation (AM) signal

    Parameters
    ----------
    amp : float
        Signal magnitude
    km : float
        Modulation coeff: amplitude sensitivity 0 <= km < 1
    fc : float
        Carrier frequency
    fs : float
        Signal frequency
    period : integer
        Number of points for signal (same as period)
    """
    tt = 2.0 * np.pi * np.linspace(0.0, 1.0, period)
    return amp * (1 + km * np.cos(fs * tt)) * np.cos(fc * tt)


def signal_chirp(amp=1.0, beta=0.25, period=100, is_complex=False, is_modsine=False):
    """
    Create Chirp signal

    Parameters
    ----------
    amp : float
        Signal magnitude
    period : integer
        Number of points for signal (same as period)
    beta : float
        Modulation bandwidth: beta < 1 for complex, beta < 0.5 for real
    is_complex : bool
        Complex signal if True
    is_modsine : bool
        Modulated by half-sine wave it True
    """
    tp = np.linspace(0.0, 1.0, period)
    tt = np.pi * (beta * period * tp ** 2)
    ts = np.pi * tp
    if is_complex is True:
        res = amp * (np.cos(tt) + 1j * np.sin(tt))
    else:
        res = amp * np.cos(tt)

    if is_modsine is True:
        return res * np.sin(ts)
    return res


def noise_gauss(mean=0.0, std=0.5, period=100):
    """
    Create Gaussian white noise as array of floating-point data

    Parameters
    ----------
    mean : float
        Mean value or signal magnitude offset
    std : float
        Standard deviation
    period : integer
        Number of points for noise (same as signal period)
    """
    return np.random.normal(mean, std, np.linspace(0.0, 1.0, period))
