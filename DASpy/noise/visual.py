# Simple functions to visulation of noise
import numpy as np 
import matplotlib.pyplot as plt

def trace_matplotlib_plot(tr, ax, **kwargs):
    ax.plot(tr.times("matplotlib"), tr.data, **kwargs)
    ax.xaxis_date()

    
def power_spectrum_plot(stream):
    '''
    Plot log-log scale curve of noise spectrum.
    '''
    Px, freq = power_spectrum_calc(stream)
    plt.plot(freq, Px[1,:])
    plt.xscale('log')
    plt.yscale('log')
    return
def power_spectrum_calc(stream):
    '''
    Calculate power spectrum of each trace in an obspy stream object.
    Note: make sure input is a stream and not trace object.
    '''
    data = []
    for tr in stream:
        data.append(tr.data)
    data = np.array(data)
    #ns_x = data.shape[0]
    ns_t = data.shape[1]
    dt = stream[0].stats.delta
    Px, freq = power_spectrum_core(data, dt, ns_t)
    return Px, freq

def power_spectrum_core(x, dt, npts):
    """
    calculate power spectrum for an array x [tr, npts].
    """
    fx   = np.fft.fft(x, npts, axis=-1)
    freq = np.fft.fftfreq(npts, dt)
    freq = freq[np.where(freq>=0)]
    Px   = np.squeeze(2* abs(fx[:,np.where(freq>=0)])**2)

    return Px, freq

def fk_plot(st, flim):
    '''
    Plot fk transform with all traces in the obspy stream st (NOTE: trace distance must be CONSTANT)
    '''
    if str(type(st))== "<class 'obspy.core.stream.Stream'>":
        print('input type: Obspy stream')
        d = []
        for tr in st:
            d.append(tr.data)
        d = np.array(d)
    if str(type(st)) == "<class 'numpy.ndarray'>":
        d = st
    if len(d.shape) ==1:
        print('Only one trace is inputed, no fk defined')
        return
    print('\t: ', d.shape)


    x_fk, f_, kx_ = fk_transform(d, dt = 0.001, dx = 1, pad_x=0)
    x_fk = x_fk[:,f_>flim[0]]; f_ = f_[f_>flim[0]]
    x_fk = x_fk[:,f_<flim[1]]; f_ = f_[f_<flim[1]]
    fig, ax = plt.subplots()
    fig.figsize=(8, 6)
    amp = abs(x_fk)
    maxamp = amp.max()
    im = plt.imshow(amp.T, aspect='auto', extent=[kx_[0], kx_[-1], f_[0], f_[-1]], vmin= -0.5*maxamp, vmax=0.5*maxamp)
    h=plt.colorbar(pad=0.01)
    h.set_label('Amplitude')
    plt.xlabel('Wavenumber (cycle/m)')
    plt.ylabel('Frequency (cycle/s)')
    #plt.ylim((-2, 2))

    return fig, ax
# FK transform
def fk_transform(data, dt=0.001, dx=1, pad_x=0):
    '''
        Wen Zhou 2021-04-13
        This function is partially based on a MATLAB code written by Muhammad Muhammad Fadhillah Akbar as part of his MSc thesis. in Utrecht 2017.
        fk spectrum of a seismic section
        data_fk = output after fk-filter
        f = frequency axis
        kx = wave number axis
        data [n_x, n_t]
        pad_x (0 - )= pad zeros on x (space) domain,  0= no pading, 1 = n_x, 2 = 2*n_x,... 
    '''

    ns_x, ns_t = data.shape
    #print(ns_x, ns_t)
    # no of zero pad colum on one side
    paded_zeros = np.zeros((round(pad_x*ns_x), ns_t) )
    # zero pad on both side IN SPACE
    data = np.concatenate((paded_zeros, data, paded_zeros), axis=0 )
    # f-x data
    # in time
    data = np.fft.fft(data,ns_t,axis=1); 
    # in space
    data = np.fft.fft(data,ns_x,axis=0); 
    # - to + nyquist freq
    f = np.fft.fftfreq(ns_t, d=dt)
    #f = np.linspace(-0.5,0.5,ns_t)/dt; 
    # - to + nyquist k
    kx = np.fft.fftfreq(ns_x, d=dx)
    #kx = np.linspace(-0.5,0.5,ns_x)/dx; 

    f = np.fft.fftshift(f)
    kx = np.fft.fftshift(kx)
    data = np.fft.fftshift(data, axes=0)
    data = np.fft.fftshift(data, axes=1)
    return data, f, kx
