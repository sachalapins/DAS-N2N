from DASpy.IO import utils
import os, sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import obspy
import gc
from scipy.signal import periodogram


def SNR_traditional(st_data, pow_vs_amp='power', return_all_channels=False):
    """Function to calculate traditional SNR (mean^2 / stdev^2).
    """
    # Calculate SNR:
    SNR_all_traces = np.zeros(len(st_data[200:500]))
    # Loop over all traces:
    for i in range(len(st_data[200:500])):
        abs_data_curr = np.abs(st_data[200:500][i].data)
        if pow_vs_amp == 'amp':
            S = np.mean(abs_data_curr)
            N = np.std(abs_data_curr)
        else:
            S = np.mean(abs_data_curr) ** 2
            N = np.std(abs_data_curr) ** 2
        # Find SNR:
        SNR_all_traces[i] = S / N

    if return_all_channels:
        return SNR_all_traces
    else:
        # And calc. average SNR:
        average_SNR = np.mean(SNR_all_traces)
        return average_SNR

def SNR_explicit(st_data, st_noise, power_vs_amp='power', return_all_channels=False):
    """Function to calculate the SNR of a data window given 
    a window of noise.
    """
    # Check if noise is same length as data:
    if not len(st_data[0].data) == len(st_noise[0].data):
        print('Error: data is not same length as noise. Exiting.')
        sys.exit()
    # Get sampling rate:
    fs = st_data[0].stats.sampling_rate
    # Get data spectra:
    data_arr = utils.stream2array(st_data) # axes are (space, time)
    f, Pxx_data_all_channels = periodogram(data_arr, fs=fs, axis=1)
    del data_arr
    gc.collect()
    # Get noise spectra:
    noise_arr = utils.stream2array(st_noise) # axes are (space, time)
    f, Pxx_noise_all_channels = periodogram(noise_arr, fs=fs, axis=1)    
    del noise_arr
    gc.collect()
    
    # Calculate SNR:
    # Integrate PSD:
    df = f[1] - f[0]
    S_and_N_all_channels = np.sum(Pxx_data_all_channels, axis=1) * df
    N_all_channels = np.sum(Pxx_noise_all_channels, axis=1) * df
    SNR_all_channels = ( S_and_N_all_channels ) / N_all_channels # ( S+N - N ) / N or ~ (S+N) / N?
    # Note: The above is approximation, since S_and_N_all_channels could have noise enhancing or 
    # decreasing amplitude.
    
    # And convert to amplitude rather than power, if specified:
    if power_vs_amp == 'amp':
        SNR_all_channels = np.sqrt(SNR_all_channels)
    
    if return_all_channels:
        return SNR_all_channels
    else:
        # And calc. average SNR:
        average_SNR = np.mean(SNR_all_channels)
        return average_SNR


def SNR_event(st, event, phase='S', nsamp_sig_win=100, nsamp_noise_win=100, return_all_channels=False):
    """Function to calculate the SNR of an event from its picks and windows around the 
    signal and the noise. The SNR is defined here as the rms amplitude of the signal 
    window divided by the rms amplitude of the noise window, as in Stork et al 2020.

    Arguments:
    Required:
    st - Stream containing data associated with event arrivals and sufficient time before to 
            window noise. (obspy stream)
    event - DASpy.detect.detect event object containing phase picks for the event. This 
            function will only use the phase picks associated with the phase specified 
            as an optional input, <phase>. (DASpy event object)
    Optional:
    phase - The phase to use (P or S). This controls what phase arrival times to use from 
            event_phase_picks. Default is 'S' (str)
    nsamp_sig_win - The number of samples to use for the signal window. (int)
    nsamp_noise_win - The number of samples to use for the noise window. (int)
    return_all_channels - If True, returns array containing SNR for each individual channel.
                        (bool)

    Returns:
    average_SNR - Average SNR for all channels combined. If return_all_channels = True, this 
                value is not returned.
    OR:
    SNR_all_channels - If return_all_channels = True, will return array of SNR values for 
                        each indivudual channel.

    """
    fs = st[0].stats.sampling_rate
    # Loop over event picks, calculating SNR:
    SNR_all_channels = []
    channel_labels = []
    for station in list(event.phase_data.keys()):
        st_tmp = st.select(station=station)
        
        # Get detection data idx:
        phase_arrival_time = event.phase_data[station][phase]['arrival_time']
        detection_idx = int((phase_arrival_time - st_tmp[0].stats.starttime) * fs)
        
        # Check if input stream has suffient length to calculate SNR, otherwise throw error:
        if detection_idx < int((nsamp_sig_win / 2.) + nsamp_sig_win + nsamp_noise_win):
            print('Error: input stream length insufficient to deal with signal and noise windows. \
                Specify new stream length to include all event pick data. Exiting.')
            sys.exit()
        
        # Calculate SNR from windows:
        signal_window_data = st_tmp[0].data[detection_idx - int(nsamp_sig_win / 2) : detection_idx + int(nsamp_sig_win / 2)]
        noise_win_idx = int(detection_idx - 1.5*nsamp_sig_win)
        noise_window_data = st_tmp[0].data[noise_win_idx - int(nsamp_noise_win) : noise_win_idx]
        sig_rms = np.average(signal_window_data ** 2) 
        noise_rms = np.average(noise_window_data ** 2) 
        SNR_curr_channel = sig_rms / noise_rms

        # And append data:
        SNR_all_channels.append(SNR_curr_channel)
        channel_labels.append(station)

        # And clear up:
        del st_tmp
        gc.collect()
    
    # And calculate overall SNR:
    average_SNR = np.average(np.array(SNR_all_channels))

    if return_all_channels:
        return SNR_all_channels, channel_labels
    else:
        return average_SNR
