# ani = Ambient Noise Interferometry
## functions including deconvolution and cross corrolation 

import numpy as np 
import scipy
import obspy
from scipy import signal

def noise_interferometry(stream, source_trace_num, 
                         sliding_window_length=1, overlap = 0.9,
                         water_level=0.01, alpha = 100, core='deconvolution'):
    #stream.taper(0.1) # python varibles are shared over functions and main. so no modification of main parameter in functions!
    stream.detrend('demean')
    
    data = []
    for tr in stream:
        data.append(tr.data)

    # fft of s and r
    data = np.array(data)
    ns_x = data.shape[0]
    ns_t = data.shape[1]
    dt = stream[0].stats.delta
    # fft length
    #sliding_window_length # second
    ns_fft = round(sliding_window_length / dt)
    ns_shift = round((1-overlap)*sliding_window_length/dt)
    # cross correlation for virtual sources
    nwindows = int((ns_t-ns_fft)/ns_shift)+1
    print('Number of windows', nwindows)
    for j in np.arange(nwindows):
        # define sliding windows
        # print('processing window: ', j)
        idx = np.arange(ns_fft) + j*ns_shift
        datain = data[:, idx]
        cc = np.array([])
        if core == 'autocorrelation':    
                cc, T = autocorrelation_core(datain, dt, ns_fft)
        else:
            k = 0
            for i in source_trace_num:
                if core == 'deconvolution':
                    temp, T = decon_core(datain, datain[i,:], dt, ns_fft, water_level, alpha)
                if core == 'crosscorrelation':
                    temp, T = crosscorrelation_core(datain, datain[i,:], dt, ns_fft, alpha)
                # put all sources into one array [ns_x, ns_fft, n_sources]
                if k==0:
                    cc = temp
                if k>0:
                    cc = np.dstack((cc, temp))
                k = k+1

#         # Normalization and stacking cross correlations
#         if j==0:
#             cc_mean = np.multiply(1/np.max(np.abs(cc), axis=-1), cc.T)
#         if j>0:
#             cc_mean = cc_mean + np.multiply(1/np.max(np.abs(cc), axis=-1), cc.T)
        # Normalization and stacking cross correlations
        if j==0:
            cc_mean = cc.T
        if j>0:
            cc_mean = cc_mean +  cc.T
        #plt.plot(cc_mean[200,:])
        #plt.pause(0.05)
    cc_mean = cc_mean/(j+1)
    print('processed: %d, windows', j)
    #plt.show()

    return cc_mean, T


def decon_core(x, y, dt, npts, water_level=0.001, alpha=20):
    '''
    x(w)/y(w)
    
    1, Fourier transform
    2, deviation
    '''
    #1. Demean
    x = scipy.signal.detrend(x, type='constant')
    y = scipy.signal.detrend(y, type='constant')
    #2. Detrend
    x = scipy.signal.detrend(x, type='linear')
    y = scipy.signal.detrend(y, type='linear')
    ns_t = x.shape[1]    
    ns_x = x.shape[0]
    tukeytaper = scipy.signal.tukey(ns_t, alpha=0.2, sym=True)
    x = x[:,0:ns_t]*tukeytaper
    y = y*tukeytaper
    ########################## FFT of data
    #npts = len(x)
    fx   = np.fft.fft(x, npts, axis=-1)
    fy   = np.fft.fft(y, npts, axis=-1)
    freq = np.fft.fftfreq(npts, dt)
    
    Px   = abs(fx)**2
    Py   = abs(fy)**2
    
    ########################## Gaussian low pass filter G = exp(-f^2/alpha^2)
    #alpha = 20
    G    = np.exp(-freq**2/alpha**2)
    
    # Apply a smoothing operator, improve variance. 
    #nsmooth = 10
    #Px_smooth = np.convolve(Px, np.ones(nsmooth)/nsmooth)
    #Py_smooth = np.convolve(Py, np.ones(nsmooth)/nsmooth)
    
    ########################## The cross spectrum
    Sxy   = np.conj(fy)*fx
    
    ########################## DECONVOLUTION! 
    level = np.ones(npts)*water_level*Py.max()
    pha   = np.vstack((Py[0:], level[0:]))
    pha   = pha.max(axis = 0)
    fxdy    = Sxy/pha * G

    xdy  = np.real(np.fft.ifft(fxdy, axis=-1))
    xdy  = np.fft.ifftshift(xdy,axes=-1)
    
    # Delay time vector, same as above.. only run on the first round
    nf   = int((npts+1)/2)  # window length 
    T    = np.linspace(-nf*dt, nf*dt, npts)

    #
    #xdy  = xdy[:, T>-1]; T = T[T>-1]; xdy  = xdy[:, T<1]; T = T[T<1]; 
    return xdy, T




def crosscorrelation_core(x, y, dt, npts, alpha=200):
    '''
    x(w)/y(w)
    
    1, Fourier transform
    2, deviation
    '''
    ns_t = x.shape[1]    
    ns_x = x.shape[0]
    #1. Demean
    x = scipy.signal.detrend(x, type='constant')
    y = scipy.signal.detrend(y, type='constant')
    #2. Detrend
    x = scipy.signal.detrend(x, type='linear')
    y = scipy.signal.detrend(y, type='linear')

    # one bit normalization
#     x[x>0] =  1; y[y>0] =  1
#     x[x<0] = -1; y[y<0] = -1
    # running-means normalization
#     eqfmin=0.5 
#     eqfmax = 30
#     eqtempx=obspy.signal.filter.bandpass(np.array(x),freqmin=eqfmin,freqmax=eqfmax,
#                                          df=1/dt,corners=4,zerophase=True)
#     eqtempy=obspy.signal.filter.bandpass(np.array(y),freqmin=eqfmin,freqmax=eqfmax,
#                                          df=1/dt,corners=4,zerophase=True)
#     eqsmoothNUM=int(2/dt) #get samples to smooth over..
#     # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
#     eq_smoothx = np.ones(x.shape)
#     for i in range(ns_x):
#         eq_smoothx[i,:] = np.convolve(np.abs(eqtempx[i,:]), np.ones(eqsmoothNUM)/eqsmoothNUM, mode='same')
#     eq_smoothy = np.convolve(np.abs(eqtempy), np.ones(eqsmoothNUM)/eqsmoothNUM,mode='same')
#     eq_smooth=np.sqrt(np.mean(eq_smoothx*eq_smoothy))
#     x= x/(eq_smooth)
#     y= y/(eq_smooth)
    
    # tapering
    
    tukeytaper = scipy.signal.tukey(ns_t, alpha=0.2, sym=True)
    x = x[:,0:ns_t]*tukeytaper
    y = y*tukeytaper
    ########################## FFT of data
    #npts = len(x)
    fx   = np.fft.fft(x, npts, axis=-1)
    fy   = np.fft.fft(y, npts, axis=-1)
    freq = np.fft.fftfreq(npts, dt)
    
    Px   = abs(fx)**2
    Py   = abs(fy)**2
    
    # Apply a smoothing operator, improve variance. 
    Px_smooth = np.ones(Px.shape)
    nsmooth = 21
    for i in range(ns_x):
        Px_smooth[i,:] = np.convolve(Px[i,:], np.ones(nsmooth)/nsmooth)[0:npts]

    Py_smooth = np.convolve(Py, np.ones(nsmooth)/nsmooth)[0:npts]
    
    ########################## The cross spectrum
    Sxy  = np.conj(fy)*fx
    G    = np.exp(-freq**2/alpha**2)
    ########################## Normalized Cross correlation! 
    cohe = Sxy/(np.sqrt(Px_smooth)*np.sqrt(Py_smooth))*G
    #cohe = Sxy#/np.sqrt(sum(Px))/np.sqrt(sum(Py))
    xycorr = np.real(scipy.fft.ifft(cohe, axis=-1))
    xycorr = np.fft.ifftshift(xycorr, axes=-1)
    
    # Delay time vector, same as above.. only run on the first round
    nf   = int((npts+1)/2)  # window length 
    T    = np.linspace(-nf*dt, nf*dt, npts)

    return xycorr, T


def autocorrelation_core(x, dt, npts):
    '''
    x(w)/y(w)
    
    1, Fourier transform
    2, deviation
    '''
    ns_t = x.shape[1]    
    ns_x = x.shape[0]
    #1. Demean
    x = scipy.signal.detrend(x, type='constant')
    #2. Detrend
    x = scipy.signal.detrend(x, type='linear')
    
    # one bit normalization
    x[x>0] =  1
    x[x<0] = -1
    # tapering
    
    tukeytaper = scipy.signal.tukey(ns_t, alpha=0.2, sym=True)
    x = x[:,0:ns_t]*tukeytaper
    ########################## FFT of data
    #npts = len(x)
    fx   = np.fft.fft(x, npts, axis=-1)
    freq = np.fft.fftfreq(npts, dt)
    
    Px   = abs(fx)**2
    
    ########################## The power spectrum (auto correlation)
    Sxx  = np.conj(fx)*fx
    xxcorr = np.real(scipy.fft.ifft(Sxx, axis=-1))
    xxcorr = np.fft.ifftshift(xxcorr, axes=-1)
    
    # Delay time vector, same as above.. only run on the first round
    nf   = int((npts+1)/2)  # window length 
    T    = np.linspace(-nf*dt, nf*dt, npts)

    return xxcorr, T