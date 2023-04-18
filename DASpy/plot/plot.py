import numpy as np 
from DASpy.IO import utils
import matplotlib.pyplot as plt
from scipy import ndimage


def image(st,style=1,skip=10,clim=[0],tmin=0,tmax=None,physicalFiberLocations=False,picks=None):
    """
    Simple image plot of DAS Stream, adapted from IRIS DAS workshop function.
    #skip=10 is default to skip every 10 ch for speed
    #style=1 is a raw plot, or 2 is a trace normalized plot
    #clim=[min,max] will clip the colormap to [min,max], deactivated by default
    
    Arguments:
    Required:
    st - The stream containing the DAS data to plot.
    Optional:
    style - Type of plot. Default is raw plot (style=1). style=2 is a trace normalized plot. 
    skip - The decimation in the spatioal domain. Default is 10. (int)
    clim - If specified, it is a list containing the lower and upper 
            limits of the colormap. Default is [], which specifies 
            that python defaults should be used. (list of 2 ints)
    tmin - Plot start time in seconds.
    tmax - Plot end time in seconds.
    physicalFiberLocations - Defines distance from header information.
    picks - DASpy event object. If specified, will plot phase picks 
            on the figure. Default is None, which specifies it is 
            unused. (DASpy detect.detect.event object)

    Returns:
    fig - A python figure object.
    """
    
    fig = plt.figure(figsize=(8,7))
#     fig = plt.figure()
    if style==1:
        img = utils.stream2array(st[::skip]) # raw amplitudes
        clabel = st[0].stats.units
    if style==2:
        img = utils.stream2array(st[::skip].copy().normalize()) # trace normalize
        clabel = st[0].stats.units+' (trace normalized)'

    t_ = st[0].stats.endtime-st[0].stats.starttime
    if physicalFiberLocations==True:
        extent = [st[0].stats.distance/1e3,st[-1].stats.distance/1e3,0,t_]
        xlabel = 'Distance relative to wellhead [km]'
    else:
        dx_ = st[1].stats.distance - st[0].stats.distance
        extent = [0,len(st)*dx_/1e3,0,t_]
        xlabel = 'Linear Fiber Length [km]'
    if len(clim)>1:
        plt.imshow(img.T,aspect='auto',interpolation='None',alpha=0.7,
                   origin='lower',extent=extent,vmin=clim[0],vmax=clim[1],cmap="seismic");
    else:
        plt.imshow(img.T,aspect='auto',interpolation='None',alpha=0.7,
                   origin='lower',extent=extent,vmin=clim[0],vmax=clim[1],cmap="seismic");
        
        
    try: 
        plt.scatter(picks.x/1000,picks.t,marker='_',c='k')
    except:
        pass
        
 


    h=plt.colorbar(pad=0.01);
    h.set_label(clabel)
#     plt.ylim(np.max(extent),0)
    plt.ylabel('Time [s]');
    plt.xlabel(xlabel);
    
    if tmax:
        plt.ylim(tmax,tmin)
    else:
        plt.ylim(np.max(extent),tmin)

    plt.gca().set_title(str(st[0].stats.starttime.datetime)+' - '+str(st[0].stats.endtime.datetime.strftime('%H:%M:%S'))+' (UTC)');

    plt.tight_layout();

    return fig


def time_distance_plot(st, skip=10, channel_spacing_m=1.0, first_channel_offset_m=0.0, clim=[], event=None):
    """
    Function to plot DAS arrivals as simple time vs. distance plot.
    
    Arguments:
    Required:
    st - The stream containing the DAS data to plot.
    Optional:
    skip - The decimation in the spatioal domain. Default is 10. (int)
    channel_spacing_m - The spacing between each DAS channel, in metres.
                        Default is 1.0 m. (float)
    first_channel_offset_m - The spatial offset of the first channel in 
                            the stream <st>, in metres. Default is 0. 
                            (float)
    clim - If specified, it is a list containing the lower and upper 
            limits of the colormap. Default is [], which specifies 
            that python defaults should be used. (list of 2 ints)
    event - DASpy event object. If specified, will plot phase picks 
            on the figure. Default is None, which specifies it is 
            unused. (DASpy detect.detect.event object)

    Returns:
    fig - A python figure object.
    """
    # Setup figure:
    fig, ax = plt.subplots(figsize=(8,8))

    # Convert data to arrays:
    data_arr = utils.stream2array(st[::skip]) # data in (x, t)
    x = ( np.arange(len(st))[::skip] * channel_spacing_m / 1e3 ) + (first_channel_offset_m / 1e3)
    t = np.arange(len(st[0].data)) / st[0].stats.sampling_rate
    T, X = np.meshgrid(t, x)
    # Setup clim if not specified:
    if len(clim) == 0:
        clim = [-np.max(np.abs(data_arr)), np.max(np.abs(data_arr))]
    # And plot data:
    pcmesh = ax.pcolormesh(X, T, data_arr, vmin=clim[0], vmax=clim[1], cmap="seismic")
    # Plot event phase picks, if specified:
    if event:
        for station in list(event.phase_data.keys()):
            for phase in list(event.phase_data[station].keys()):
                rel_arrival_time = event.phase_data[station][phase]['arrival_time'] - st[0].stats.starttime
                distance = float(station[1:]) * channel_spacing_m / 1e3
                ax.scatter(distance, rel_arrival_time, c='k', alpha=0.75, marker='_')
    # Label plot:
    ax.invert_yaxis()
    cb = fig.colorbar(pcmesh, ax=ax, pad=0.01)
    cb.set_label('Strain rate')
    ax.set_title(str(st[0].stats.starttime))
    ax.set_xlabel('Distance along fibre ($km$)')
    ax.set_ylabel('Time ($s$)')
    return fig


def plot_summed(img,xrange=None,dt=None,cmap='seismic'):
    """
    Plot function which displays an image of DAS data alongside amplitudes summed in both time and spatial domain.
    
    Arguments:
    Required:
    img - nparray of DAS data.
    xrange - Spatial range of data
    dt - sampling rate
    cmap - colour map, default is 'seismic'

    Returns:
    fig - A python figure object.
    """
    fig = plt.figure(figsize=(8,8))
    grid = plt.GridSpec(4,4,hspace=0.05,wspace=0.05)
    main_ax = fig.add_subplot(grid[:-1, :-1])
    y_hist = fig.add_subplot(grid[:-1, -1], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, :-1], sharex=main_ax)

    vmn = np.percentile(np.real(img),0.01)
    vmx = np.percentile(np.real(img),99.99)
    vlim=max(abs(vmn),abs(vmx))
    
    tsteps=img.shape[1]
    trange=np.arange(0,dt*tsteps,dt)


    tt,xx = np.meshgrid(trange,xrange)

    mainimg=main_ax.pcolormesh(xx.T,tt.T,img.T,vmin=-1*vlim,vmax=vlim,cmap=cmap)
    mainimg.set_rasterized(True)
    main_ax.invert_yaxis()
    main_ax.xaxis.tick_top()

    main_ax.grid()
    xhistsum=np.sum(abs(img),axis=1)/img.shape[1]

    xhistsum=xhistsum/xhistsum.max()
    yhistsum=np.sum(abs(img),axis=0)/img.shape[0]
    yhistsum=yhistsum/yhistsum.max()

    # x_hist.plot(xrange,np.sum(abs(img),axis=1)/img.shape[1])
    # y_hist.plot(np.sum(abs(img),axis=0)/img.shape[0],trange)

    x_hist.plot(xrange,xhistsum)
    y_hist.plot(yhistsum,trange)

    x_hist.axes.get_yaxis().set_ticks([])
    y_hist.axes.get_xaxis().set_ticks([])
    y_hist.xaxis.set_label_position('top')
    y_hist.yaxis.tick_right()

    plt.setp(y_hist.get_yticklabels(), visible=False)
    # y_hist.axes.yaxis.set_ticklabels([])

    y_hist.grid()
    x_hist.grid()

    y_hist.set_xlabel('Sum over channels')
    x_hist.set_ylabel('Sum over time')

    x_hist.set_ylim(bottom=0)
    y_hist.set_xlim(left=0)

    main_ax.set_ylabel('Time (s)')
    x_hist.set_xlabel('Offset (m)')
    main_ax.xaxis.set_label_position('top')
    main_ax.set_xlabel('Offset (m)')

#     main_ax.set_ylabel('Channel Number')
#     x_hist.set_xlabel('Sample Number')

    main_ax.autoscale(enable=True)

    # fig.tight_layout()

    return fig

def fk_plot(st,wavenumber,max_freq,xrange=200,yrange=0.2,normalise=True):
    """
    FK filter for a 2D DAS numpy array. Returns a filtered image.
    
    Arguments:
    Required:
    st - The stream containing the DAS data to plot.
    wavenumber - maximum wavenumber value
    max_freq - maximum frequency value
    xrange - Spatial limit of data. Default is 200m
    yrange - time limit of data. Default is 0.2s
    normalise - apply fk plot to normalised data. Default it True, ie normalised.

    Returns:
    fig - A python figure object.
    """  
    if normalise==True:
        for tr in st:
            tr.data=tr.data/np.max(tr.data)
    
    data=utils.stream2array(st).T
    fs=st[0].stats.sampling_rate
    ch_space=abs(st[1].stats.distance-st[0].stats.distance)
    
    # Detrend by removing the mean 
    data=data-np.mean(data)
    
    # Apply a 2D fft transform
    fftdata=np.fft.fftshift(np.fft.fft2(data.T))
    
    freqs=np.fft.fftfreq(fftdata.shape[1],d=(1./fs))
    wavenums=np.fft.fftfreq(fftdata.shape[0],d=ch_space)

    freqs=np.fft.fftshift(freqs) 
    wavenums=np.fft.fftshift(wavenums)

    freqsgrid=np.broadcast_to(freqs,fftdata.shape)   
    wavenumsgrid=np.broadcast_to(wavenums,fftdata.T.shape).T

    
    # Define mask and blur the edges 
    mask=np.logical_and(np.logical_and(wavenumsgrid<=wavenumber,wavenumsgrid>=-wavenumber),abs(freqsgrid)<max_freq)
    x=mask*1.
    blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    
    # Plots the filter, with area remove greyed out
    fig = plt.figure(figsize=[6,6])
    img1 = plt.imshow(np.log10(abs(fftdata)), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
    img1.set_clim(-5,5)
    img1 = plt.imshow(abs(blurred_mask-1),cmap='Greys',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],alpha=0.2,aspect='auto')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wavenumber (1/m)')
    plt.xlim(-1*xrange,xrange)
    plt.ylim(-1*yrange,yrange)
    plt.show()
        
    return fig