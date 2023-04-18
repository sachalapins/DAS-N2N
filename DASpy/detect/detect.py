import sys, os
import numpy as np
import obspy
import pandas as pd 
from numba import jit
import gc
import multiprocessing as mp
from obspy.signal.trigger import classic_sta_lta
import matplotlib.pyplot as plt 


class event:
    """Class to store event detection data."""
    def __init__(self, event_id):
        # Initialise event parameters:
        self.event_id = event_id
        self.phase_data = {}

    def assign_phase_picks(self, stations, phase_time_picks, phase_labels, weights=[]):
        """Function to assign phase pick data to event.
        Arguments:
        stations - List of station labels. (list of strs)
        phase_time_picks - List of phase time picks for associated stations, 
                            in UTCDateTime format. (list of UTCDateTime objects)
        phase_labels - List of phase labels associated with phase time picks. 
                        These can be <P> or <S>. e.g. ['P', 'S', 'S']. (list of 
                        specific strs)
        """
        # Perform initial checks:
        if not (len(stations) + len(phase_time_picks) + len(phase_labels)) / 3. == len(stations):
            print("Error: len(stations) =! len(phase_time_picks) =! len(phase_labels). Please \
                    make sure the lengths of the inputs to event.assign_phase_picks() are all equal. Exiting.")
            sys.exit()
        if len(weights) > 0:
            if not len(weights) == len(stations):
                print("Error: len(stations) =! len(weights). Please \
                    make sure the lengths of the inputs to event.assign_phase_picks() are all equal. Exiting.")

        # Loop over entries:
        for i in range(len(stations)):
            # Check if station exists already or not:
            if stations[i] in list(self.phase_data.keys()):
                self.phase_data[stations[i]][phase_labels[i]] = {}
                self.phase_data[stations[i]][phase_labels[i]]['arrival_time'] = phase_time_picks[i]
                if len(weights) > 0:
                    self.phase_data[stations[i]][phase_labels[i]]['Weight'] = weights[i]
            else:
                self.phase_data[stations[i]] = {}
                self.phase_data[stations[i]][phase_labels[i]] = {}
                self.phase_data[stations[i]][phase_labels[i]]['arrival_time'] = phase_time_picks[i]
        




@jit(nopython=True)
def _ncc(data, template):
    """Function performing cross-correlation between long waveform data (data) and template.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients."""
    n_samp_template = len(template)
    n_iters = len(data) - n_samp_template + 1
    ncc = np.zeros(n_iters)
    for i in range(len(ncc)):
        ncc[i] = np.sum(data[i:n_samp_template+i] * template / (np.std(data[i:n_samp_template+i]) * np.std(template))) / n_samp_template
    return(ncc)


def _cc_single_time_window(st_curr_win, stations, max_samp_shift=10):
    """
    Function to perform cross-correlation detection lgorithm that shifts 
    across a stream of channels in the spatial axis, for one time window, 
    over the whole stream, st, input.

    Arguemnts:
    Required:
    st_curr_win - Stream of DAS data, containing channels, labelled D???. (obspy 
        stream)
    stations - The stations to process for. This dictates the order over 
                which the spatial cross-correlation is undertaken. (list 
                of strs).
    Optional:
    max_samp_shift - The maximum shift to apply in samples. Default is 10.
                    (int)
    
    Returns:
    norm_cc_all_channels - The normallised cc shift for all channels combined.
    """
    # Setup data stores:
    cc_vals_ind_DAS_channels = np.zeros(len(stations) - 1)
    cc_shifts_ind_DAS_channels = np.zeros(len(stations))    

    # Loop through DAS channels (i.e. stations), correlating one with the next:
    for i in range(len(stations) - 1):
        chan_1_curr = st_curr_win.select(station=stations[i])[0].data
        chan_2_curr = st_curr_win.select(station=stations[i+1])[0].data
        # Perform shift:
        cc_vals_all_t_shifts_curr = []
        for j in range(-int(max_samp_shift), int(max_samp_shift)+1):
            chan_2_curr = np.roll(chan_2_curr, j)
            ncc_curr = _ncc(chan_1_curr, chan_2_curr)
            cc_vals_all_t_shifts_curr.append(ncc_curr)
        cc_vals_all_t_shifts_curr = np.array(cc_vals_all_t_shifts_curr)
        # And find max ncc value and associated shift:
        cc_vals_ind_DAS_channels[i] = np.max(cc_vals_all_t_shifts_curr)
        cc_shifts_ind_DAS_channels[i+1] = np.argmax(cc_vals_all_t_shifts_curr) - int(max_samp_shift)
    
    # And get normallised cc shift for all channels combined:
    norm_cc_all_channels = np.sum(cc_vals_ind_DAS_channels) / len(cc_vals_ind_DAS_channels)

    return(norm_cc_all_channels)




def spatial_cc_event_detector(st, win_len_secs=1.0, max_samp_shift=10, nproc=1):
    """
    Function to detect events based on a spatial cross-correlation detection 
    algorithm that shifts across a stream of channels in the spatial axis, 
    one by one. Cross-correlation will be undertaken starting with station 
    D001 to D???.

    Arguemnts:
    Required:
    st - Stream of DAS data, containing channels, labelled D???. (obspy 
        stream)
    Optional:
    win_len_secs - Length of the moving window, in seconds. Default is 1 s.
                    (float)
    max_samp_shift - The maximum shift to apply in samples. Default is 10.
                    (int)    
    nproc - The number of processors to use. Default is 1. (int)

    Returns:

    """
    # Do initial setup:
    fs = st[0].stats.sampling_rate

    # Perform initial checks:
    # Check if sufficient data to implement moving window:
    if len(st[0].data) / fs <= 1.0:
        print('Error: Stream, st, is too short to perform a moving \
                window analysis for a window length of ', win_len_secs, 's. \
                    Exiting.')
        sys.exit()
    # Check that all stream traces are same length:
    len_check_val = len(st[0].data)
    for i in range(len(st)-1):
        if not len(st[i+1].data) == len_check_val:
            print('Error: Stream length not equal for all traces. Exiting.')
            sys.exit()
    # Check that number of processors isn't more than number available:
    nproc = int(nproc)
    print('Note: Multiprocessing currently not implemented!')
    if nproc > mp.cpu_count():
        print('Error: number of specified processors is greater than number of \
            processors available (', mp.cpu_count(), ')')

    # Get list of stations, in ascending order:
    stations = []
    for tr in st:
        if tr.stats.station not in stations:
            stations.append(tr.stats.station)
    stations = sorted(stations)

    # Loop over (not currently moving) time windows:
    n_wins = int((len(st[0].data) / fs) / win_len_secs) #len(st[0].data) - int( win_len_secs * fs )
    norm_cc_with_t = np.zeros(n_wins)
    multi_proc_count = 0
    for i in range(n_wins):
        # Trim st for current window:
        # (Note: Speed performance gains could be made here!!!)
        st_curr_win = st.copy()
        st_curr_win.trim(starttime=st[0].stats.starttime + (i * win_len_secs), endtime=st[0].stats.starttime + ((i+1) * win_len_secs))
        print('Window', i+1, '/', n_wins,'(', st_curr_win[0].stats.starttime, ' to ', st_curr_win[0].stats.endtime, ')')

        # Perform cross-correlation for current window:
        norm_cc_with_t[i] = _cc_single_time_window(st_curr_win, stations, max_samp_shift=max_samp_shift)

        # And clear up:
        del st_curr_win
        gc.collect()

    # And convert data for output:
    ncc_st = obspy.Stream()
    tr = obspy.Trace()
    tr.data = norm_cc_with_t
    tr.stats.station = 'ncc'
    tr.stats.network = st[0].stats.network
    tr.stats.sampling_rate = int( (st[0].stats.endtime - st[0].stats.starttime) / n_wins) #fs
    ncc_st.append(tr)

    return(ncc_st)


def spectral_detector(st, win_len_secs=1.0):
    """
    Function to perform detection of events by calculating spectrum through time, 
    and if there is an amplitude 

    Arguemnts:
    Required:
    st - Stream of DAS data, containing channels, labelled D???. (obspy 
        stream)
    Optional:
    win_len_secs - Length of the moving window, in seconds. Default is 1 s.
                    (float)

    Returns:

    """


def mad(x, scale=1.4826):
    """
    Calculates the Median Absolute Deviation (MAD) values for the input array x.

    Returns MAD value with scaling.
    """
    # Calculate median and mad values:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return scale * mad


def sta_lta_detector(st, sta_win=0.05, lta_win=0.25, MAD_multiplier=10.0, min_channels_trig=10, phase='S'):
    """
    Function to detect phase arrivals using an STA/LTA trigger with a Median Average Deviation 
    trigger threshold.
    Note: This is currently a very simple trigger, which will just pick hte highest STA/LTA value 
    within the data for each channel.

    Arguments:

    """
    # Get number of samples in sta and lta windows:
    fs = st[0].stats.sampling_rate
    nsta = sta_win * fs
    nlta = lta_win * fs

    # Loop over traces, calculating STA/LTA values:
    pick_times = []
    pick_channel_labels = []
    assigned_phase_labels = []
    onset_values = []
    for i in range(len(st)):
        sta_lta = classic_sta_lta(st[i].data, nsta, nlta)
        sta_lta_mad = mad(sta_lta, scale=MAD_multiplier*1.4826)
        # And if triggers phase detection:
        if np.max(sta_lta) > sta_lta_mad:
            pick_times.append( (st[i].stats.starttime + (np.argmax(sta_lta) / fs)) )
            pick_channel_labels.append(st[i].stats.station)
            assigned_phase_labels.append(phase)
            onset_values.append(np.max(sta_lta))
        # else:
        #     pick_times.append( st[i].stats.starttime )
        #     pick_channel_labels.append(st[i].stats.station)
        #     assigned_phase_labels.append(phase)
        #     onset_values.append(np.nan)

    # Check if minimum number of channels have trigger:
    if len(pick_times) > min_channels_trig:
        trigger = True
    else:
        trigger = False

    # And write event picks to new event object:
    if trigger:
        event_out = event(str(st[0].stats.starttime))
        event_out.assign_phase_picks(pick_channel_labels, pick_times, assigned_phase_labels, weights=onset_values)
    else:
        event_out = event('no_trigger')

    # (Shift sta/lta functions to find peak STA/LTA)?

    return trigger, event_out



        

