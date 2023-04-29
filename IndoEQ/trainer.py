#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
last update: 06/25/2020

---

Edited on 2023/04/18
https://github.com/ganjarammar/earthquake_dl_mka

Source:
https://github.com/seoungheong/LEQNet/blob/main/LEQNet/JS_EQTransformer/core/trainer.py

"""

from __future__ import print_function
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
import os
import shutil
import multiprocessing
from .LEQNet_utils import DataGenerator, _lr_schedule, LEQNetCopy, PreLoadGenerator, data_reader
from .EqT_utils import EqTransformerCopy
from .IndoEQ_utils import IndoEQ
import datetime
from tqdm import tqdm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class DataGenerator(keras.utils.Sequence):

    """

    Keras generator with preprocessing

    Parameters
    ----------
    list_IDsx: str
        List of trace names.

    file_name: str
        Name of hdf5 file containing waveforms data.

    dim: tuple
        Dimension of input traces.

    batch_size: int, default=32
        Batch size.

    n_channels: int, default=3
        Number of channels.

    phase_window: int, fixed=40
        The number of samples (window) around each phaset.

    shuffle: bool, default=True
        Shuffeling the list.

    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.

    label_type: str, default=gaussian
        Labeling type: 'gaussian', 'triangle', or 'box'.

    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.

    add_event_r: {float, None}, default=None
        Chance for randomly adding a second event into the waveform.

    add_gap_r: {float, None}, default=None
        Add an interval with zeros into the waveform representing filled gaps.

    shift_event_r: {float, None}, default=0.9
        Rate of augmentation for randomly shifting the event within a trace.

    add_noise_r: {float, None}, default=None
        Chance for randomly adding Gaussian noise into the waveform.

    drop_channel_r: {float, None}, default=None
        Chance for randomly dropping some of the channels.

    scale_amplitude_r: {float, None}, default=None
        Chance for randomly amplifying the waveform amplitude.

    pre_emphasis: bool, default=False
        If True, waveforms will be pre emphasized.

    Returns
    --------
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.

    """

    def __init__(self,
                 list_IDs,
                 file_name,
                 dim,
                 batch_size=32,
                 n_channels=3,
                 phase_window= 40,
                 shuffle=True,
                 norm_mode = 'max',
                 label_type = 'gaussian',
                 augmentation = False,
                 add_event_r = None,
                 add_gap_r = None,
                 shift_event_r = None,
                 add_noise_r = None,
                 drop_channel_r = None,
                 scale_amplitude_r = None,
                 pre_emphasis = True):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.phase_window = phase_window
        self.list_IDs = list_IDs
        self.file_name = file_name
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type
        self.augmentation = augmentation
        self.add_event_r = add_event_r
        self.add_gap_r = add_gap_r
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.drop_channel_r = drop_channel_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.augmentation:
            return 2*int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.augmentation:
            indexes = self.indexes[index*self.batch_size//2:(index+1)*self.batch_size//2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y1, y2, y3 = self.__data_generation(list_IDs_temp)
        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _normalize(self, data, mode = 'max'):
        'Normalize waveforms in each batch'

        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'

        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate:
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate):
        'Randomly add gaps (zeros) of different sizes into waveforms'

        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate:
            data[gap_start:gap_end,:] = 0
        return data

    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'

        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])
        else:
            data_noisy = data
        return data_noisy

    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'

        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):
        'Used for triangolar labeling'

        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate):
        'Add a scaled version of the event into the empty part of the trace'

        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM
                spt_secondEV = secondEV_strt
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:
                    additions = [spt_secondEV, sst_secondEV]
                    data = added

        return data, additions

    def _shift_event(self, data, addp, adds, coda_end, snr, rate):
        'Randomly rotate the array to shift the event location'

        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None
        if np.random.uniform(0, 1) < rate:
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]

            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate
            else:
                addp2 = None
            if adds+nrotate >= 0 and adds+nrotate < org_len:
                adds2 = adds+nrotate
            else:
                adds2 = None
            if coda_end+nrotate < org_len:
                coda_end2 = coda_end+nrotate
            else:
                coda_end2 = org_len
            if addp2 and adds2:
                data = data2
                addp = addp2
                adds = adds2
                coda_end= coda_end2
        return data, addp, adds, coda_end

    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels):
            bpf = data[:, ch]
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

    def __data_generation(self, list_IDs_temp):
        'read the waveforms'
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        y1 = np.zeros((self.batch_size, self.dim, 1))
        y2 = np.zeros((self.batch_size, self.dim, 1))
        y3 = np.zeros((self.batch_size, self.dim, 1))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            additions = None
            # dataset = fl.get(str(ID) + '/data') # comment if using STEAD 
            dataset = fl.get('data/' + str(ID)) 

            # if ID.split('_')[-1] == 'EV':
            # ------ comment if using STEAD 
            # data = np.array(dataset).T[:6000]
            # spt = int(dataset.attrs['p_'])
            # sst = int(dataset.attrs['s_'])
            # coda_end = int(dataset.attrs['coda_'])
            # ------

            # ------ comment if not using STEAD 
            data = np.array(dataset)[:6000]
            try:
                spt = int(dataset.attrs['p_arrival_sample'])
                sst = int(dataset.attrs['s_arrival_sample'])
                # coda_end = int(dataset.attrs['coda_end_sample'])[0][0]
            except Exception as e:
                # print(i, str(ID), f'| {e}')
                continue
            # ------ comment if not using STEAD 

            snr = None
            # snr = dataset.attrs['snr_db']

            # elif ID.split('_')[-1] == 'NO':
            #     data = np.array(dataset)

            ## augmentation
            # if self.augmentation == True:
            #     if i <= self.batch_size//2:
            #         if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
            #             data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2)
            #         if self.norm_mode:
            #             data = self._normalize(data, self.norm_mode)
            #     else:
            #         if dataset.attrs['trace_category'] == 'earthquake_local':
            #             if self.shift_event_r:
            #                 data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r)

            #             if self.add_event_r:
            #                 data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r)

            #             if self.add_noise_r:
            #                 data = self._add_noise(data, snr, self.add_noise_r)

            #             if self.drop_channel_r:
            #                 data = self._drop_channel(data, snr, self.drop_channel_r)
            #                 data = self._adjust_amplitude_for_multichannels(data)

            #             if self.scale_amplitude_r:
            #                 data = self._scale_amplitude(data, self.scale_amplitude_r)

            #             if self.pre_emphasis:
            #                 data = self._pre_emphasis(data)

            #             if self.norm_mode:
            #                 data = self._normalize(data, self.norm_mode)

            #         elif dataset.attrs['trace_category'] == 'noise':
            #             if self.drop_channel_r:
            #                 data = self._drop_channel_noise(data, self.drop_channel_r)

            #             if self.add_gap_r:
            #                 data = self._add_gaps(data, self.add_gap_r)

            #             if self.norm_mode:
            #                 data = self._normalize(data, self.norm_mode)

            # elif self.augmentation == False:
            #     if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local':
            #         data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2)
            #     if self.norm_mode:
            #         data = self._normalize(data, self.norm_mode)

            X[i, :, :] = data

            ## labeling
            # if dataset.attrs['trace_category'] == 'earthquake_local':
            if self.label_type  == 'gaussian':
                sd = None
                if spt and sst:
                    sd = sst - spt

                if sd and sst:
                    if sst+int(0.4*sd) <= self.dim:
                        y1[i, spt:int(sst+(0.4*sd)), 0] = 1
                    else:
                        y1[i, spt:self.dim, 0] = 1

                if spt and (spt-20 >= 0) and (spt+20 < self.dim):
                    y2[i, spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]
                elif spt and (spt-20 < self.dim):
                    y2[i, 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:self.dim-(spt-20)]

                if sst and (sst-20 >= 0) and (sst-20 < self.dim):
                    y3[i, sst-20:sst+20, 0] = np.exp(-(np.arange(sst-20,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]
                elif sst and (sst-20 < self.dim):
                    y3[i, 0:sst+20, 0] = np.exp(-(np.arange(0,sst+20)-sst)**2/(2*(10)**2))[:self.dim-(sst-20)]

                if additions:
                    add_sd = None
                    add_spt = additions[0]
                    add_sst = additions[1]
                    if add_spt and add_sst:
                        add_sd = add_sst - add_spt

                    if add_sd and add_sst+int(0.4*add_sd) <= self.dim:
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1
                    else:
                        y1[i, add_spt:self.dim, 0] = 1

                    if add_spt and (add_spt-20 >= 0) and (add_spt+20 < self.dim):
                        y2[i, add_spt-20:add_spt+20, 0] = np.exp(-(np.arange(add_spt-20,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]
                    elif add_spt and (add_spt+20 < self.dim):
                        y2[i, 0:add_spt+20, 0] = np.exp(-(np.arange(0,add_spt+20)-add_spt)**2/(2*(10)**2))[:self.dim-(add_spt-20)]

                    if add_sst and (add_sst-20 >= 0) and (add_sst+20 < self.dim):
                        y3[i, add_sst-20:add_sst+20, 0] = np.exp(-(np.arange(add_sst-20,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]
                    elif add_sst and (add_sst+20 < self.dim):
                        y3[i, 0:add_sst+20, 0] = np.exp(-(np.arange(0,add_sst+20)-add_sst)**2/(2*(10)**2))[:self.dim-(add_sst-20)]


            elif self.label_type  == 'triangle':
                sd = None
                if spt and sst:
                    sd = sst - spt

                if sd and sst:
                    if sst+int(0.4*sd) <= self.dim:
                        y1[i, spt:int(sst+(0.4*sd)), 0] = 1
                    else:
                        y1[i, spt:self.dim, 0] = 1

                if spt and (spt-20 >= 0) and (spt+21 < self.dim):
                    y2[i, spt-20:spt+21, 0] = self._label()
                elif spt and (spt+21 < self.dim):
                    y2[i, 0:spt+spt+1, 0] = self._label(a=0, b=spt, c=2*spt)
                elif spt and (spt-20 >= 0):
                    pdif = self.dim - spt
                    y2[i, spt-pdif-1:self.dim, 0] = self._label(a=spt-pdif, b=spt, c=2*pdif)

                if sst and (sst-20 >= 0) and (sst+21 < self.dim):
                    y3[i, sst-20:sst+21, 0] = self._label()
                elif sst and (sst+21 < self.dim):
                    y3[i, 0:sst+sst+1, 0] = self._label(a=0, b=sst, c=2*sst)
                elif sst and (sst-20 >= 0):
                    sdif = self.dim - sst
                    y3[i, sst-sdif-1:self.dim, 0] = self._label(a=sst-sdif, b=sst, c=2*sdif)

                if additions:
                    add_spt = additions[0]
                    add_sst = additions[1]
                    add_sd = None
                    if add_spt and add_sst:
                        add_sd = add_sst - add_spt

                    if add_sd and add_sst+int(0.4*add_sd) <= self.dim:
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1
                    else:
                        y1[i, add_spt:self.dim, 0] = 1

                    if add_spt and (add_spt-20 >= 0) and (add_spt+21 < self.dim):
                        y2[i, add_spt-20:add_spt+21, 0] = self._label()
                    elif add_spt and (add_spt+21 < self.dim):
                        y2[i, 0:add_spt+add_spt+1, 0] = self._label(a=0, b=add_spt, c=2*add_spt)
                    elif add_spt and (add_spt-20 >= 0):
                        pdif = self.dim - add_spt
                        y2[i, add_spt-pdif-1:self.dim, 0] = self._label(a=add_spt-pdif, b=add_spt, c=2*pdif)

                    if add_sst and (add_sst-20 >= 0) and (add_sst+21 < self.dim):
                        y3[i, add_sst-20:add_sst+21, 0] = self._label()
                    elif add_sst and (add_sst+21 < self.dim):
                        y3[i, 0:add_sst+add_sst+1, 0] = self._label(a=0, b=add_sst, c=2*add_sst)
                    elif add_sst and (add_sst-20 >= 0):
                        sdif = self.dim - add_sst
                        y3[i, add_sst-sdif-1:self.dim, 0] = self._label(a=add_sst-sdif, b=add_sst, c=2*sdif)


            elif self.label_type  == 'box':
                sd = None
                if sst and spt:
                    sd = sst - spt

                if sd and sst+int(0.4*sd) <= self.dim:
                    y1[i, spt:int(sst+(0.4*sd)), 0] = 1
                else:
                    y1[i, spt:self.dim, 0] = 1
                if spt:
                    y2[i, spt-20:spt+20, 0] = 1
                if sst:
                    y3[i, sst-20:sst+20, 0] = 1

                if additions:
                    add_sd = None
                    add_spt = additions[0]
                    add_sst = additions[1]
                    if add_spt and add_sst:
                        add_sd = add_sst - add_spt

                    if add_sd and add_sst+int(0.4*add_sd) <= self.dim:
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1
                    else:
                        y1[i, add_spt:self.dim, 0] = 1
                    if add_spt:
                        y2[i, add_spt-20:add_spt+20, 0] = 1
                    if add_sst:
                        y3[i, add_sst-20:add_sst+20, 0] = 1

        fl.close()

        return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32')


def _get_model_class(class_name):
    if class_name == 'indoeq':
        return IndoEQ
    elif class_name == 'eqt':
        return EqTransformerCopy
    elif class_name == 'leq':
        return LEQNetCopy
    else:
        raise ValueError('Invalid model class name. Should be one of [indoeq, eqt, leq]')


def trainer(input_hdf5=None,
            input_csv=None,
            output_name=None,
            model_class='indoeq',
            input_dimension=(6000, 3),
            cnn_blocks=5,
            lstm_blocks=2,
            padding='same',
            activation = 'relu',
            drop_rate=0.1,
            use_prelu=False,
            shuffle=True,
            label_type='gaussian',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3,
            drop_channel_r=0.5,
            add_gap_r=0.2,
            scale_amplitude_r=None,
            pre_emphasis=False,
            loss_weights=[0.05, 0.40, 0.55],
            loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            train_valid_test_split=[0.85, 0.05, 0.10],
            mode='generator',
            batch_size=200,
            epochs=200,
            monitor='val_loss',
            patience=12,
            multi_gpu=False,
            number_of_gpus=4,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True,
            key_dim=16,
            num_heads=8):

    """

    Generate a model and train it.

    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of data with NumPy arrays containing 3 component waveforms each 1 min long.

    input_csv: str, default=None
        Path to a CSV file with one column (trace_name) listing the name of all datasets in the hdf5 file.

    output_name: str, default=None
        Output directory.

    input_dimension: tuple, default=(6000, 3)
        OLoss types for detection, P picking, and S picking respectively.

    cnn_blocks: int, default=5
        The number of residual blocks of convolutional layers.

    lstm_blocks: int, default=2
        The number of residual blocks of BiLSTM layers.

    padding: str, default='same'
        Padding type.

    activation: str, default='relu'
        Activation function used in the hidden layers.

    drop_rate: float, default=0.1
        Dropout value.

    shuffle: bool, default=True
        To shuffle the list prior to the training.

    label_type: str, default='triangle'
        Labeling type. 'gaussian', 'triangle', or 'box'.

    normalization_mode: str, default='std'
        Mode of normalization for data preprocessing, 'max': maximum amplitude among three components, 'std', standard deviation.

    augmentation: bool, default=True
        If True, data will be augmented simultaneously during the training.

    add_event_r: float, default=0.6
        Rate of augmentation for adding a secondary event randomly into the empty part of a trace.

    shift_event_r: float, default=0.99
        Rate of augmentation for randomly shifting the event within a trace.

    add_noise_r: float, defaults=0.3
        Rate of augmentation for adding Gaussian noise with different SNR into a trace.

    drop_channel_r: float, defaults=0.4
        Rate of augmentation for randomly dropping one of the channels.

    add_gap_r: float, defaults=0.2
        Add an interval with zeros into the waveform representing filled gaps.

    scale_amplitude_r: float, defaults=None
        Rate of augmentation for randomly scaling the trace.

    pre_emphasis: bool, defaults=False
        If True, waveforms will be pre-emphasized. Defaults to False.

    loss_weights: list, defaults=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.

    loss_types: list, defaults=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
        Loss types for detection, P picking, and S picking respectively.

    train_valid_test_split: list, defaults=[0.85, 0.05, 0.10]
        Precentage of data split into the training, validation, and test sets respectively.

    mode: str, defaults='generator'
        Mode of running. 'generator', or 'preload'.

    batch_size: int, default=200
        Batch size.

    epochs: int, default=200
        The number of epochs.

    monitor: int, default='val_loss'
        The measure used for monitoring.

    patience: int, default=12
        The number of epochs without any improvement in the monitoring measure to automatically stop the training.

    multi_gpu: bool, default=False
        If True, multiple GPUs will be used for the training.

    number_of_gpus: int, default=4
        Number of GPUs uses for multi-GPU training.

    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None.

    gpu_limit: float, default=None
        Set the maximum percentage of memory usage for the GPU.

    use_multiprocessing: bool, default=True
        If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction.

    Returns
    --------
    output_name/models/output_name_.h5: This is where all good models will be saved.

    output_name/final_model.h5: This is the full model for the last epoch.

    output_name/model_weights.h5: These are the weights for the last model.

    output_name/history.npy: Training history.

    output_name/X_report.txt: A summary of the parameters used for prediction and performance.

    output_name/test.npy: A number list containing the trace names for the test set.

    output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.

    output_name/X_learning_curve_loss.png: The learning curve of loss.

    Notes
    --------
    'generator' mode is memory efficient and more suitable for machines with fast disks.
    'pre_load' mode is faster but requires more memory and it comes with only box labeling.

    """


    args = {
    "input_hdf5": input_hdf5,
    "input_csv": input_csv,
    "output_name": output_name,
    "model_class": model_class,
    "input_dimension": input_dimension,
    "cnn_blocks": cnn_blocks,
    "lstm_blocks": lstm_blocks,
    "padding": padding,
    "activation": activation,
    "use_prelu": use_prelu,
    "drop_rate": drop_rate,
    "shuffle": shuffle,
    "label_type": label_type,
    "normalization_mode": normalization_mode,
    "augmentation": augmentation,
    "add_event_r": add_event_r,
    "shift_event_r": shift_event_r,
    "add_noise_r": add_noise_r,
    "add_gap_r": add_gap_r,
    "drop_channel_r": drop_channel_r,
    "scale_amplitude_r": scale_amplitude_r,
    "pre_emphasis": pre_emphasis,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "train_valid_test_split": train_valid_test_split,
    "mode": mode,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,
    "multi_gpu": multi_gpu,
    "number_of_gpus": number_of_gpus,
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "use_multiprocessing": use_multiprocessing,
    "key_dim": key_dim,
    "num_heads": num_heads
    }

    def train(args):
        """

        Performs the training.

        Parameters
        ----------
        args : dic
            A dictionary object containing all of the input parameters.

        Returns
        -------
        history: dic
            Training history.

        model:
            Trained model.

        start_training: datetime
            Training start time.

        end_training: datetime
            Training end time.

        save_dir: str
            Path to the output directory.

        save_models: str
            Path to the folder for saveing the models.

        training size: int
            Number of training samples.

        validation size: int
            Number of validation samples.

        """

        save_dir, save_models, last_model_path = _make_dir(args['output_name'], args['model_class'])
        training, validation = _split(args, save_dir)
        callbacks = _make_callback(args, save_models)
        model = _build_model(args)
        if last_model_path is not None:
            # load saved Keras model
            model = tf.keras.models.load_model(last_model_path)

        if args['gpuid']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)
            else:
                print("tensorflow can't activate GPU and change using CPU")

        start_training = time.time()

        if args['mode'] == 'generator':

            params_training = {'file_name': str(args['input_hdf5']),
                              'dim': args['input_dimension'][0],
                              'batch_size': args['batch_size'],
                              'n_channels': args['input_dimension'][-1],
                              'shuffle': args['shuffle'],
                              'norm_mode': args['normalization_mode'],
                              'label_type': args['label_type'],
                              'augmentation': args['augmentation'],
                              'add_event_r': args['add_event_r'],
                              'add_gap_r': args['add_gap_r'],
                              'shift_event_r': args['shift_event_r'],
                              'add_noise_r': args['add_noise_r'],
                              'drop_channel_r': args['drop_channel_r'],
                              'scale_amplitude_r': args['scale_amplitude_r'],
                              'pre_emphasis': args['pre_emphasis']}

            params_validation = {'file_name': str(args['input_hdf5']),
                                 'dim': args['input_dimension'][0],
                                 'batch_size': args['batch_size'],
                                 'n_channels': args['input_dimension'][-1],
                                 'shuffle': False,
                                 'norm_mode': args['normalization_mode'],
                                 'augmentation': False}

            training_generator = DataGenerator(training, **params_training)
            validation_generator = DataGenerator(validation, **params_validation)

            print('Started training in generator mode ...')
            history = model.fit(training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=args['use_multiprocessing'],
                                workers=multiprocessing.cpu_count(),
                                callbacks=callbacks,
                                epochs=args['epochs'])
                                # class_weight={0: 0.11, 1: 0.89})

        elif args['mode'] == 'preload':
            X, y1, y2, y3 = data_reader(list_IDs=training+validation,
                                       file_name=str(args['input_hdf5']),
                                       dim=args['input_dimension'][0],
                                       n_channels=args['input_dimension'][-1],
                                       norm_mode=args['normalization_mode'],
                                       augmentation=args['augmentation'],
                                       add_event_r=args['add_event_r'],
                                       add_gap_r=args['add_gap_r'],
                                       shift_event_r=args['shift_event_r'],
                                       add_noise_r=args['add_noise_r'],
                                       drop_channel_r=args['drop_channel_r'],
                                       scale_amplitude_r=args['scale_amplitude_r'],
                                       pre_emphasis=args['pre_emphasis'])

            print('Started training in preload mode ...', flush=True)
            history = model.fit({'input': X},
                                {'detector': y1, 'picker_P': y2, 'picker_S': y3},
                                epochs=args['epochs'],
                                validation_split=args['train_valid_test_split'][1],
                                batch_size=args['batch_size'],
                                callbacks=callbacks,
                                class_weight={0: 0.11, 1: 0.89})
        else:
            print('Please specify training_mode !', flush=True)
        end_training = time.time()

        return history, model, start_training, end_training, save_dir, save_models, len(training), len(validation)

    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)





def _make_dir(output_name, model_class=None):

    """

    Make the output directories.

    Parameters
    ----------
    output_name: str
        Name of the output directory.

    Returns
    -------
    save_dir: str
        Full path to the output directory.

    save_models: str
        Full path to the model directory.

    """
    model_class = model_class or ''
    if output_name == None:
        print('Please specify output_name!')
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
        save_models = os.path.join(save_dir, 'models')
        if os.path.isdir(save_dir):
            # if save dir contains `outputs` folder in its tree
            if 'outputs' in os.listdir(save_dir):
                # take last file in the path
                out_path = os.path.join(save_dir, 'outputs', model_class)
                last_model_name = sorted(os.listdir(out_path))[-1]
                last_model_path = os.path.join(out_path, last_model_name)
                return save_dir, save_models, last_model_path
            else:
                shutil.rmtree(save_dir)
                os.makedirs(save_models)
    return save_dir, save_models, None


def _build_model(args):

    """

    Build and compile the model.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    Returns
    -------
    model:
        Compiled model.

    """

    class_name = args['model_class']
    model_class = _get_model_class(class_name)
    inp = Input(shape=args['input_dimension'], name='input')

    kwargs = dict(
        padding=args['padding'],
        activationf =args['activation'],
        cnn_blocks=args['cnn_blocks'],
        BiLSTM_blocks=args['lstm_blocks'],
        drop_rate=args['drop_rate'],
        loss_weights=args['loss_weights'],  
        loss_types=args['loss_types'],
        kernel_regularizer=keras.regularizers.l2(1e-6),
        bias_regularizer=keras.regularizers.l1(1e-4),
        num_heads=args['num_heads'],
        key_dim=args['key_dim']
    )
    if class_name == 'indoeq':
        kwargs.update({'use_prelu': args['use_prelu']})

    model = model_class(**kwargs)(inp)

    model.summary()
    return model




def _split(args, save_dir):

    """

    Split the list of input data into training, validation, and test set.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    save_dir: str
       Path to the output directory.

    Returns
    -------
    training: str
        List of trace names for the training set.
    validation : str
        List of trace names for the validation set.

    """

    df = pd.read_csv(args['input_csv'])
    ev_list = df.trace_name.tolist()
    np.random.shuffle(ev_list)
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
    test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(save_dir+'/test', test)
    return training, validation



def _make_callback(args, save_models):

    """

    Generate the callback.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    save_models: str
       Path to the output directory for the models.

    Returns
    -------
    callbacks: obj
        List of callback objects.


    """

    m_name=str(args['output_name'])+'_{epoch:03d}.h5'
    filepath=os.path.join(save_models, m_name)
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'],
                                           patience=args['patience'])
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'],
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True)
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks




def _pre_loading(args, training, validation):

    """

    Load data into memory.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.

    training: str
        List of trace names for the training set.

    validation: str
        List of trace names for the validation set.

    Returns
    -------
    training_generator: obj
        Keras generator for the training set.

    validation_generator: obj
        Keras generator for the validation set.


    """

    training_set={}
    fl = h5py.File(args['input_hdf5'], 'r')

    print('Loading the training data into the memory ...')
    pbar = tqdm(total=len(training))
    for ID in training:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('earthquake/local/'+str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('non_earthquake/noise/'+str(ID))
        training_set.update( {str(ID) : dataset})

    print('Loading the validation data into the memory ...', flush=True)
    validation_set={}
    pbar = tqdm(total=len(validation))
    for ID in validation:
        pbar.update()
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('earthquake/local/'+str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('non_earthquake/noise/'+str(ID))
        validation_set.update( {str(ID) : dataset})

    params_training = {'dim':args['input_dimension'][0],
                       'batch_size': args['batch_size'],
                       'n_channels': args['input_dimension'][-1],
                       'shuffle': args['shuffle'],
                       'norm_mode': args['normalization_mode'],
                       'label_type': args['label_type'],
                       'augmentation': args['augmentation'],
                       'add_event_r': args['add_event_r'],
                       'add_gap_r': args['add_gap_r'],
                       'shift_event_r': args['shift_event_r'],
                       'add_noise_r': args['add_noise_r'],
                       'drop_channel_r': args['drop_channel_r'],
                       'scale_amplitude_r': args['scale_amplitude_r'],
                       'pre_emphasis': args['pre_emphasis']}

    params_validation = {'dim': args['input_dimension'][0],
                         'batch_size': args['batch_size'],
                         'n_channels': args['input_dimension'][-1],
                         'shuffle': False,
                         'norm_mode': args['normalization_mode'],
                         'augmentation': False}

    training_generator = PreLoadGenerator(training, training_set, **params_training)
    validation_generator = PreLoadGenerator(validation, validation_set, **params_validation)

    return training_generator, validation_generator




def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args):

    """

    Write down the training results.

    Parameters
    ----------
    history: dic
        Training history.

    model:
        Trained model.

    start_training: datetime
        Training start time.

    end_training: datetime
        Training end time.

    save_dir: str
        Path to the output directory.

    save_models: str
        Path to the folder for saveing the models.

    training_size: int
        Number of training samples.

    validation_size: int
        Number of validation samples.

    args: dic
        A dictionary containing all of the input parameters.

    Returns
    --------
    ./output_name/history.npy: Training history.

    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.

    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.

    ./output_name/X_learning_curve_loss.png: The learning curve of loss.


    """

    np.save(save_dir+'/history',history.history)
    model.save(save_dir+'/final_model.h5')
    model.to_json()
    model.save_weights(save_dir+'/model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--')
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss',
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png')))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_f1'])
    ax.plot(history.history['picker_P_f1'])
    ax.plot(history.history['picker_S_f1'])
    try:
        ax.plot(history.history['val_detector_f1'], '--')
        ax.plot(history.history['val_picker_P_f1'], '--')
        ax.plot(history.history['val_picker_S_f1'], '--')
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1', 'val_detector_f1', 'val_picker_P_f1', 'val_picker_S_f1'], loc='lower right')
    except Exception:
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1'], loc='lower right')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_f1.png')))

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta

    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))

    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file:
        the_file.write('================== Overal Info =============================='+'\n')
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')
        the_file.write('input_csv: '+str(args['input_csv'])+'\n')
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')
        the_file.write('================== Model Parameters ========================='+'\n')
        the_file.write('input_dimension: '+str(args['input_dimension'])+'\n')
        the_file.write('cnn_blocks: '+str(args['cnn_blocks'])+'\n')
        the_file.write('lstm_blocks: '+str(args['lstm_blocks'])+'\n')
        the_file.write('padding_type: '+str(args['padding'])+'\n')
        the_file.write('activation_type: '+str(args['activation'])+'\n')
        the_file.write('drop_rate: '+str(args['drop_rate'])+'\n')
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n')
        the_file.write('================== Training Parameters ======================'+'\n')
        the_file.write('mode of training: '+str(args['mode'])+'\n')
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')
        the_file.write('train_valid_test_split: '+str(args['train_valid_test_split'])+'\n')
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n')
        the_file.write('multi_gpu: '+str(args['multi_gpu'])+'\n')
        the_file.write('number_of_gpus: '+str(args['number_of_gpus'])+'\n')
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')
        the_file.write('================== Training Performance ====================='+'\n')
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_f1: '+str(history.history['detector_f1'][-1])+'\n')
        the_file.write('last picker_P_f1: '+str(history.history['picker_P_f1'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['picker_S_f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('label_type: '+str(args['label_type'])+'\n')
        the_file.write('augmentation: '+str(args['augmentation'])+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('add_event_r: '+str(args['add_event_r'])+'\n')
        the_file.write('add_noise_r: '+str(args['add_noise_r'])+'\n')
        the_file.write('shift_event_r: '+str(args['shift_event_r'])+'\n')
        the_file.write('drop_channel_r: '+str(args['drop_channel_r'])+'\n')
        the_file.write('scale_amplitude_r: '+str(args['scale_amplitude_r'])+'\n')
        the_file.write('pre_emphasis: '+str(args['pre_emphasis'])+'\n')