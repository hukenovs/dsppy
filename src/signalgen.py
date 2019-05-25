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


class DataVector:
    """
    Return data array for time / freq domains

    Parameters
    ----------
    nlen : integer
        Number of points for signal
    """
    def __init__(self, nlen=10):
        self.nlen = nlen

    def get_vector(self):
        return np.linspace(0.0, 1.0, self.nlen)


class SimpleSignal:
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

    Methods
    -------
    get_data()
        Return harmonic signal as array of floating-point data
    """
    def __init__(self, amp=1.0, freq=10.0, period=100, mode='sin'):
        self.amp = amp
        self.freq = freq
        self.period = period
        self.mode = mode
        self.__tt = DataVector(self.period).get_vector()

    def get_data(self):
        tt = self.freq * 2.0 * np.pi * self.__tt
        if self.mode == 'sin':
            return self.amp * np.sin(tt)
        if self.mode == 'cos':
            return self.amp * np.cos(tt)
        if self.mode == 'cmp':
            return self.amp * (np.sin(tt) + 1j*np.cos(tt))
        raise ValueError('Wrond signal mode')


class AmSignal:
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

    Methods
    -------
    get_data()
        Return AM signal as array of floating-point data
    """
    def __init__(self, amp=1.0, km=10.0, fc=10.0, fs=2.0, period=100):
        self.amp = amp
        self.km = km
        self.fc = fc
        self.fs = fs
        self.period = period
        self.__tt = DataVector(self.period).get_vector()

    def get_data(self):
        tt = 2.0 * np.pi * self.__tt
        return self.amp * (1 + self.km * np.cos(self.fs * tt)) * np.cos(self.fc * tt)


class ChirpSignal:
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

    Methods
    -------
    get_data()
        Return Chirp signal as array of floating-point data
    """
    def __init__(self, amp=1.0, period=100, beta=0.125, is_complex=True, is_modsine=True):
        self.amp = amp
        self.period = period
        self.beta = beta
        self.is_complex = is_complex
        self.is_modsine = is_modsine
        self.__tt = DataVector(self.period).get_vector()

    def get_data(self):
        tt = np.pi * (self.beta * self.period * self.__tt ** 2)
        ts = np.pi * self.__tt
        if self.is_complex is True:
            res = self.amp * (np.cos(tt) + 1j * np.sin(tt))
        else:
            res = self.amp * np.cos(tt)

        if self.is_modsine is True:
            return res * np.sin(ts)
        return res


class GaussNoise:
    """
    Create Gaussian white noise

    Parameters
    ----------
    mean : float
        Mean value or signal magnitude offset
    std : float
        Standard deviation
    period : integer
        Number of points for noise (same as signal period)

    Methods
    -------
    get_noise()
        Return white noise as array of floating-point data
    """
    def __init__(self, mean=0.0, std=0.5, period=100):
        self.mean = mean
        self.std = std
        self.period = period

    def get_noise(self):
        return np.random.normal(self.mean, self.std, self.period)
