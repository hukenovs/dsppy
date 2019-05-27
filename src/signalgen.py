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

Functions :

    Name:              Return:
    signal_period    - sequence [0.0, 1.0), step = 1 / period
    signal_simple    - sine or cosine
    signal_am        - AM modulation
    signal_fm        - FM modulation
    signal_chirp     - Chirp modulation
    noise_gauss      - Gauss white noise

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


def signal_period(period=100):
    """
    Create array of data sequence [0.0, 1.0), step = 1 / period

    Parameters
    ----------
    period : integer
        Number of points for signal (same as period)
    """
    tlast = (period - 1.0) / period
    return np.linspace(0.0, tlast, int(period))


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
        Output mode: sine or cosine
    """

    tt = freq * 2.0 * np.pi * signal_period(period)
    if mode == 'sin':
        return amp * np.sin(tt)
    if mode == 'cos':
        return amp * np.cos(tt)
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
    tt = 2.0 * np.pi * signal_period(period)
    return amp * (1 + km * np.cos(fs * tt)) * np.cos(fc * tt)


def signal_fm(amp=1.0, kd=0.25, fc=10.0, fs=2.0, period=100):
    """
    Create Frequency modulation (FM) signal

    Parameters
    ----------
    amp : float
        Signal magnitude
    kd : float
        Frequency deviation, kd < period/4, (e.g. fc = 0, fs = 1, kd = 16)
    fc : float
        Carrier frequency
    fs : float
        Signal frequency
    period : integer
        Number of points for signal (same as period)
    """
    tt = 2.0 * np.pi * signal_period(period)
    return amp * np.cos(fc * tt + kd/fs * np.sin(fs * tt))


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
    tt = np.pi * beta * period * signal_period(period) ** 2
    ts = np.pi * signal_period(period)
    if is_complex is True:
        res = amp * (np.cos(tt) + 1j * np.sin(tt))
    else:
        res = amp * np.cos(tt)

    if is_modsine is True:
        return res * np.sin(ts)
    return res


def noise_gauss(mean=0.0, std=0.5, seed=0, period=100):
    """
    Create Gaussian white noise as array of floating-point data

    Parameters
    ----------
    mean : float
        Mean value or signal magnitude offset
    std : float
        Standard deviation
    seed : int
        Seed used to initialize a pseudorandom number generator
    period : integer
        Number of points for noise (same as signal period)
    """
    np.random.seed(seed=seed)
    return np.random.normal(mean, std, period)


def calc_awgn(sig, snr=0.0, seed=1):
    """
    Create Gaussian white noise as array of floating-point data

    Parameters
    ----------
    sig : float
        Array of floating point data (real or complex signal)
    snr : float
        Standard deviation
    seed : int
        Seed used to initialize a pseudorandom number generator
    """
    pwr_sig = calc_power(sig=sig)
    pwr_2db = pwr_sig / calc_idb(db=snr, is_power=True)
    chk_cmp = any(np.iscomplex(sig))

    np.random.seed(seed=seed)
    get_wgn = np.random.randn(1, np.size(sig))[0] / 3
    if chk_cmp is True:
        return sig + np.sqrt(pwr_2db/2) * (get_wgn + 1j * get_wgn)
    return sig + np.sqrt(pwr_2db) * get_wgn


def calc_energy(sig):
    """
    Calculate Energy of signal.
    Formula: y = sum( abs( x(i)^2 ) ), where i = 0..N-1

    Parameters
    ----------
    sig : float
        Array of floating point data

    """
    return np.sum(np.abs(sig ** 2))


def calc_power(sig):
    """
    Calculate Power of signal.
    Formula: y = (1/N) * sum( abs( x(i)^2 ) ), where i = 0..N-1

    Parameters
    ----------
    sig : float
        Array of floating point data

    """
    return (1/np.size(sig)) * np.sum(np.abs(sig ** 2))


def calc_db(amp=10.0, is_power=False):
    """
    Calculate the dB (Energy or Power)
    Enegry: y = 20 * log10(x),
    Power:  y = 10 * log10(x).

    Parameters
    ----------
    amp : float
        Signal magnitude (energy or power)
    is_power : bool
        if True - calc as Power
    """
    if is_power is True:
        return 10.0 * np.log10(amp)
    else:
        return 20.0 * np.log10(amp)


def calc_idb(db=10.0, is_power=False):
    """
    Calculate the Energy or Power from dB value
    Enegry: y = 10 ** (x / 20)
    Power:  y = 10 ** (x / 10)

    Parameters
    ----------
    db : float
        Signal ratio (dB)
    is_power : bool
        if True - calc the Power magnitude
    """
    if is_power is True:
        return 10.0 ** (db / 10.0)
    else:
        return 10.0 ** (db / 20.0)


def calc_rms(xx):
    """
    Calculate root mean square (RMS) as the square root of the mean square

    Parameters
    ----------
    xx : float
        Data sequence (real or complex)
    """
    return np.sqrt(np.mean(np.abs(xx) ** 2)) / np.size(xx)


def calc_snr(xx, yy):
    """
    Calculate SNR (signal-noise ratio) value in dB

    Parameters
    ----------
    xx : float array
        Input signal sequence (real or complex)
    yy : float array
        The noise sequence
    """
    return calc_db(calc_rms(xx) / calc_rms(yy), is_power=False)
