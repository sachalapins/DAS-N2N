import numpy as np 
from obspy import Stream, Trace
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from DASpy.IO import utils
from scipy import ndimage
from scipy import signal
import gc
from scipy.signal.signaltools import wiener as scipy_wiener



def fk_filter(st, wavenumber, max_freq):
    '''
    FK filter for a 2D DAS numpy array. Returns a filtered image.
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    wavenumber - maximum value for the filter  
    max_freq - maximum value for the filter  
    Returns:
    st_fk - FK filtered time series.
    '''
    data=utils.stream2array(st).T
    fs=st[0].stats.sampling_rate
    ch_space=abs(st[1].stats.distance-st[0].stats.distance)

    # Detrend by removing the mean 
    data=data-np.mean(data)
    
    # Apply a 2D fft transform
    fftdata=np.fft.fftshift(np.fft.fft2(data.T))
    
    freqs=np.fft.fftfreq(fftdata.shape[1],d=(1./fs))
    wavenums=2*np.pi*np.fft.fftfreq(fftdata.shape[0],d=ch_space)

    freqs=np.fft.fftshift(freqs) 
    wavenums=np.fft.fftshift(wavenums)

    freqsgrid=np.broadcast_to(freqs,fftdata.shape)   
    wavenumsgrid=np.broadcast_to(wavenums,fftdata.T.shape).T
    
    # Define mask and blur the edges 
    mask=np.logical_and(np.logical_and(wavenumsgrid<=wavenumber,wavenumsgrid>=-wavenumber),abs(freqsgrid)<max_freq)
    x=mask*1.
    blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    # Apply the mask to the data
    ftimagep = fftdata * blurred_mask
    ftimagep = np.fft.ifftshift(ftimagep)
    
    # Finally, take the inverse transform and show the blurred image
    imagep = np.fft.ifft2(ftimagep)

    imagep = imagep.real
    
    # Convert back to a stream    
    
    st_fk=st.copy()
    for channel in range(len(imagep)):
        st_fk[channel].data=imagep[channel]
        
        
    return st_fk

def image_sharpen_demean(st,sigma=3,alpha=30):
    '''
    Sharpens an image.
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    sigma - 
    alpha -
    Returns:
    st_shp - Image sharpened time series.
    '''
    
    image=utils.stream2array(st)
    
    image2=image-np.mean(image)
    blurred_f = ndimage.gaussian_filter(image2, sigma)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, sigma)
    
    imagep = blurred_f + alpha * (blurred_f - filter_blurred_f)

    st_shp=st.copy()
    for channel in range(len(imagep)):
        st_shp[channel].data=imagep[channel]
        
        
    return st_shp

def wiener(st):
    '''
    Wiener filter with wrapper for applying to obspy streams
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    Returns:
    st_wiener - The filtered time series.
    '''
    image=utils.stream2array(st)

    image2=image-np.mean(image)
    filtered_img = scipy_wiener(image2, (5, 5))

    st_wiener=st.copy()
    for channel in range(len(filtered_img)):
        st_wiener[channel].data=filtered_img[channel]
        
    return st_wiener


def notch(st, f_notch, bw):
    """Notch filter to filter out a specific frequency.
    Note: Applies a zero phase filter.
    Arguments:
    st - Stream of DAS data to apply notch filter to (obspy stream)
    f_notch - The frequency to apply a notch filter for in Hz (float)
    bw - The bandwidth of the notch filter in Hz (float)

    Returns:
    data_filt - The filtered time series.
    """
    # Loop over stream, applying notch filter to individual traces:
    st_notch = st.copy()
    for i in range(len(st_notch)):
        # Get data for current trace:
        data = st_notch[i].data
        fs = st_notch[i].stats.sampling_rate

        # Create the notch filter:
        Q = float(f_notch) / float(bw)
        b, a = signal.iirnotch(f_notch, Q, fs)

        # Apply notch filter (zero phase):
        data_filt = signal.filtfilt(b, a, data)
        st_notch[i].data = data_filt

    return st_notch
        
    
    

