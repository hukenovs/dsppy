# #####################################################################
#
# Title         : signalgen.py
# Author        : Alexander Kapitanov
# E-mail        : sallador@bk.ru
# Lang.         : python
# Company       :
# Release Date  : 2019/05/24
#
#
# #####################################################################
#
# Description :
#    Signal generator for digital signal processing
#
#
#
# #####################################################################
#
# GNU GENERAL PUBLIC LICENSE
# Version 3, 29 June 2007
#
# Copyright (c) 2019 Kapitanov Alexander
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
# APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
# HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT
# WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND
# PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE
# DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR
# OR CORRECTION.
#
# #####################################################################
import numpy as np
from scipy.signal import chirp


class DataVector:
    def __init__(self, period=10):
        self.period = period

    # Create time vector
    def get_vector(self):
        return np.linspace(0.0, 1.0, self.period)


class SignalGen:
    def __init__(self, amp=1.0, freq=10.0, period=100, mode='sin'):
        self.amp = amp
        self.freq = freq
        self.period = period
        self.mode = mode
        self.__tt = DataVector(self.period).get_vector()

    def __get_vec(self):
        return self.freq * 2.0 * np.pi * self.__tt

    def __gen_sin(self):
        return self.amp * np.sin(self.__get_vec())

    def __gen_cos(self):
        return self.amp * np.cos(self.__get_vec())

    def __gen_cmp(self):
        return self.amp * (self.__gen_cos() + 1j*self.__gen_sin())

    modes = {
        'sin': __gen_sin,
        'cos': __gen_cos,
        'cmp': __gen_cmp,
    }

    def get_signal(self):
        method = self.modes.get(self.mode)
        if not method:
            raise ValueError('Wrond signal mode')
        return method(self)
# class SignalGen:
#     def __init__(self, amp=1.0, freq=10.0, period=100, mode='sin'):
#         self.amp = amp
#         self.freq = freq
#         self.period = period
#         self.mode = mode
#         self.__tt = DataVector(self.period).get_vector()
#
#     def __get_vec(self, freq):
#         return freq * 2.0 * np.pi * self.__tt
#
#     def __gen_sin(self, amp, freq):
#         return amp * np.sin(self.__get_vec(freq))
#
#     def __gen_cos(self, amp, freq):
#         return amp * np.cos(self.__get_vec(freq))
#
#     def __gen_cmp(self, amp, freq):
#         return amp * (self.__gen_cos(amp, freq) + 1j*self.__gen_sin(amp, freq))
#
#     __modes = {
#         'sin': __gen_sin,
#         'cos': __gen_cos,
#         'cmp': __gen_cmp,
#     }
#
#     def get_signal(self, mode):
#         method = self.__modes.get(mode)
#         if not method:
#             raise ValueError('Wrond signal mode')
#         return method()


class GetGaussNoise:
    def __init__(self, mean=0.0, std=0.5, period=100):
        self.mean = mean
        self.std = std
        self.period = period

    def get_noise(self):
        return np.random.normal(self.mean, self.std, self.period)


def vector_sig(period=10):
    return np.linspace(0.0, 1.0, period)


def harm_sig(amp=1.0, freq=10.0, period=100, mode='sin'):

    tt = freq * 2.0 * np.pi * vector_sig(period)

    if mode == 'cos':
        return amp * np.cos(tt)
    if mode == 'sin':
        return amp * np.sin(tt)
    if mode == 'cmp':
        return amp * (np.cos(tt) + 1j * np.sin(tt))
    # print("Mode set error! Change mode property to: sin, cos OR cmp")


def am_sig(amp=1.0, km=0.3, fc=10.0, fm=5.0, period=100):
    tt = 2.0 * np.pi * vector_sig(period)
    tc = fc * tt
    tm = fm * tt
    return amp * (1 + km * np.cos(tm)) * np.cos(tc)


def chirp_sig(amp=1.0, period=100, beta=0.125, is_complex=True, is_sine=True):
    tt = 2.0 * np.pi * (beta * period * vector_sig(period) ** 2)
    ts = 1.0 * np.pi * vector_sig(period)
    if is_complex is True:
        res = amp * (np.cos(tt) + 1j*np.sin(tt))
    else:
        res = amp * np.cos(tt)

    if is_sine is True:
        return res * np.sin(ts)

    return res
