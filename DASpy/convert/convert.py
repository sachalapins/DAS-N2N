import numpy as np 
import matplotlib.pyplot as plt
from DASpy.IO import utils
from scipy import ndimage
from scipy import signal
import gc



# Define functions:

def vel_to_strain_rate(das_data_in, dx=1.0):
    """Function to convert DAS data from velocity to strain-rate. 
    Note: Doesn't apply gauge length effects. This is done separately in apply_gauge_length().
    Arguments:
    Required:
    das_data_in - Array of data to convert from velocity to strain-rate. (np array)
    Optional:
    dx - The spacing between the DAS channels. Can be a float or the same shape as das_data_in (float or np array)
    
    Returns:
    das_data_out - np array of strain rate data.
    """
    das_data_out = np.gradient(das_data_in, dx, axis=0) / dx
    return das_data_out


def strain_rate_to_vel(das_data_in, dx=1.0):
    """TO BE COMPLETED"""


def apply_gauge_length(das_data_in, gauge_length=10.0):
    """TO BE COMPLETED"""


def rotate_synth_Q_T_data_to_das_axis(synth_data_q, synth_data_t, das_azi_from_N, azi_event_to_sta_from_N, aniso_angle_from_N=0.0, aniso_delay_t=0.0, fs=1000.0):
    """Function to rotate synthetic QT data into das axis, assuming vertical arrival angles.
    Arguments:
    Required:
    synth_data_q - Array of synthetic Q component data. (np array of floats)
    synth_data_t - Array of synthetic T component data. (np array of floats)
    das_azi_from_N - DAS positive axis (away from interrogator) from North, in degrees (float)
    azi_event_to_sta_from_N - Epicentral angle of station from event, from North, in degrees (float)
    Optional:
    aniso_angle_from_N - Anisotropy angle from North in degrees. If -1.0 or 0.0, does not apply anisotropy. 
                        (float)
    aniso_delay_t - Delay time between fast and slow shear waves, in seconds. If aniso_angle_from_N <= 0, 
                    then no anisotropy is applied. Default is 0, i.e. no anisotropy applied. (float)
    fs - Sampling rate of data in Hz (float)

    Returns:
    data_out_das_axis - Data out, rotated accordingly (array of floats)
    """
    # Setup input angles in rad:
    gamma = das_azi_from_N*2.*np.pi/360.
    theta = azi_event_to_sta_from_N*2.*np.pi/360.
    if aniso_angle_from_N > 0.:
        phi = aniso_angle_from_N*2.*np.pi/360.
    else:
        phi = 0.0
    # 1. Calculate N and E fast directions:
    N_fast = -synth_data_q * np.cos(theta) * np.cos(phi)
    E_fast = synth_data_t * np.cos(theta) * np.cos(phi)
    # 2. Calculate N and E slow directions:
    N_slow = synth_data_q * np.cos(theta) * np.sin(phi)
    E_slow = -synth_data_t * np.cos(theta) * np.sin(phi)
    N_slow = np.roll(N_slow, int(fs*aniso_delay_t), axis=0)
    E_slow = np.roll(E_slow, int(fs*aniso_delay_t), axis=0)
    # 3. Combine fast and slow data together:
    N_overall = N_fast + N_slow
    E_overall = E_fast + E_slow
    # 4. And convert to DAS orientation:
    data_out_das_axis = N_overall * np.cos(gamma) + E_overall * np.sin(gamma)
    return data_out_das_axis






    
    

