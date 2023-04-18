import os, sys
from DASpy.IO import utils
from DASpy.filters import filters
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pylops
from obspy import UTCDateTime
import scipy.ndimage as ndimage
import gc 


def radon_slider(st,start_distance=0,winsize=200,overlap=50,slowmin=-0.5e-3,slowmax=0,npx = 101):
    '''    
    Sliding 2D radon transform using pylops
    
    Arguments:
    Required:
    st - Stream of DAS data, containing channels
    start_distance - beginning of the DAS trace, default=0
    winsize - number of channels in the window
    overlap - channel overlap
    
    Optional:

    Returns:
    semblance_out - a dictionary of the semblance and other key parameters
    '''
    # Get data from stream and detrend:
    data=utils.stream2array(st).T
    # Demean/detrend the data:
    data = (data-np.mean(data)).T # (transposed ready for pylops functions)

    channel_spacing=st[2].stats.distance-st[1].stats.distance
    fs=st[0].stats.sampling_rate
    
    par = {'ox':start_distance,'dx':channel_spacing,'nx':data.shape[0], 'ot':0, 'dt':1/fs, 'nt':data.shape[1]}
    # Create axis
    t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)
    
    chidxs=np.arange(winsize//2,data.shape[0]-winsize//2,winsize-overlap)
    
    # Number of sliding windows
    nwins = len(chidxs)
    print(nwins)
    
    # Number of velocity points
    
    px = np.linspace(slowmin, slowmax, npx)
    
    dimsd = data.shape
    dims = (nwins*npx, par['nt'])
    
    # Sliding window transform without taper
    Op = \
        pylops.signalprocessing.Radon2D(t, np.linspace(-par['dx']*winsize//2,
                                                       par['dx']*winsize//2,
                                                       winsize),
                                        px, centeredh=False, kind='linear',
                                        engine='numba')
    
    Slid = pylops.signalprocessing.Sliding2D(Op, dims, dimsd,
                                         winsize, overlap,
                                         tapertype=None)

    radonsum = Slid.H * data.flatten()
    radonsum = radonsum.reshape(nwins,npx,dimsd[1])

    radonsqsum = Slid.H * (data**2).flatten()
    radonsqsum = radonsqsum.reshape(nwins,npx,dimsd[1])

    semblance=radonsum**2/radonsqsum
    
    semblance_out={'semblance':semblance, 'data':data, 'px':px, 'chidxs':chidxs, 'winsize':winsize, 'overlap':overlap, 't':t, 'x':x, 'starttime':st[0].stats.starttime}
    
    # And do some tidying:
    del radonsum, radonsqsum, semblance, data
    gc.collect()

    return semblance_out

def radon_plot(semblance_out,windidx,picks=None):
    '''
    Radon transform plotter
    
    Arguments:
    Required:
    semblance_out - dictionary output from the radon_slider function
    windidx - window panel to plot
    '''    
    data2 = semblance_out['data']
    chidxs=semblance_out['chidxs']
    px= semblance_out['px']
    semblance=semblance_out['semblance']
    t=semblance_out['t']
    x=semblance_out['x']
    winsize=semblance_out['winsize']
    
    picks_df=pd.DataFrame(picks)
    
    if windidx >= len(semblance):
        print("error: only %s panels available"%(len(semblance)))
 
    chidx=chidxs[windidx]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(121)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    tt,pp = np.meshgrid(t,px*1000) # time vs slowness in s/km (converted from s/m)
    
    ax.pcolormesh(pp,tt,semblance[windidx,:,:])
    
    ax.grid()

    ax.set_xlabel('Slowness (s/km)')
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('Time (s)')

    ax2 = fig.add_subplot(122)

    ax2.invert_yaxis()
    ax2.xaxis.tick_top()
    tt,chc = np.meshgrid(t,x[chidx-winsize//2:chidx+winsize//2])

    sampleslice=data2[chidx-winsize//2:chidx+winsize//2,:]

    vmn = np.percentile(np.real(sampleslice),0.001)
    vmx = np.percentile(np.real(sampleslice),99.999)
    vlim=max(abs(vmn),abs(vmx))

    #     main_ax.imshow(img,aspect='auto',vmin=-1*vlim,vmax=vlim,cmap='seismic')
    ax2.pcolormesh(chc,tt,sampleslice,vmin=-1*vlim,vmax=vlim,cmap='seismic')
    ax2.autoscale(enable=True)
    
    ax2.grid()

    ax2.set_xlabel('Offset (m)')
    ax2.xaxis.set_label_position('top') 
    ax2.set_ylabel('Time (s)')
    
    if picks != None:
        ax.scatter(picks_df.px*1000,picks_df.t,c='r')
    
        ax2.scatter(picks_df.x,picks_df.t,c='k')
        
#         try:
#             ax.set_ylim(picks_df.t_max[0]+1,picks_df.t_max[0]-1)
#             ax2.set_ylim(picks_df.t_max[0]+1,picks_df.t_max[0]-1)
#         except:
#             pass

#     for i in range(len(y_max)):
#         ax.plot(px[y_max[i]]*1000,t[x_max[i]], 'ro')
#     x_pick=np.arange(chidx-100,chidx+100)
#     ax2.plot(x_pick,px[y_max[0]]*x_pick+t[x_max[0]],'k',lw=5)


def ss_picker(semblance_out,windidx,neighborhood_size=50,threshold=130):    
    '''
    Picks arrivals from the radon transform data
    
    Arguments:
    Required:
    semblance_out - dictionary output from the radon_slider function
    windidx - window panel to plot
    
    Returns:
    picks - dictionary containing slowness and pick time
    ''' 
    # Get data:
    semblance=semblance_out['semblance']
    t=semblance_out['t']
    x=semblance_out['x']
    px= semblance_out['px']
    chidxs=semblance_out['chidxs']
    chidx=chidxs[windidx]
    
    # Get maxima of senblance space:
    data_max = ndimage.filters.maximum_filter(semblance[windidx,:,:], neighborhood_size)
    maxima = (semblance[windidx,:,:] == data_max)
    data_min = ndimage.filters.minimum_filter(semblance[windidx,:,:], neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    # Find picks:
    picks={'x':[],'px':[],'t':[]}
    for dpx,dt in slices:
        picks['x'].append(chidx)
        px_center = (dpx.start + dpx.stop - 1)/2    
        picks['px'].append(px[int(px_center)])
        t_center = (dt.start + dt.stop - 1)/2
        picks['t'].append(t[int(t_center)])

    return picks


def pick_collate(semblance_out,neighborhood_size=50,threshold=130):
    '''Collated picks from each window'''
    picks_all={'x':[],'px':[],'t':[]}
    picks_all_df=pd.DataFrame(picks_all)
    # Loop over windows, collating picks:
    for windidx in range(0,len(semblance_out['chidxs'])):
        picks=ss_picker(semblance_out,windidx,neighborhood_size,threshold=threshold)
        picks_df=pd.DataFrame(picks)
        picks_all_df=pd.concat([picks_all_df,picks_df])
        
    return picks_all_df

def binning(data,binsize,tmin,tmax):
    """"
    Bins data at user defined intervals
    """
    data=np.sort(data)
    bins=np.arange(tmin,tmax,binsize)
    digitized = np.digitize(data, bins)
    bin_length = [len(data[digitized == i]) for i in range(1, len(bins)+1)]
    return bins,bin_length

def moving_average(a, n=3) :
    """
    Fast moving average 
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def ss_coincidence(semblance_out,event=1,threshold=120,pmin=-0.001,pmax=-0.0001,midpt=500,binsize=0.005,trigger_val=1, plot=True):
    '''
    A coincidence filter and wrapper for ss_picker
    
    Arguments:
    Required:
    semblance_out - dictionary output from the radon_slider function
    threshold - value for the radon picker
    pmin - minimum slowness value in s/km
    pmax - maximum slowness value in s/km
    binsize - bin size in seconds. Size dictates the sensitivity of the coincidence filter.  
    
    Returns:
    picks - dictionary containing slowness and pick time
    '''
    # Collated picks from each window        
    picks_all_df=pick_collate(semblance_out,threshold)
    
    # Filter on slowness
    picks_all_df=picks_all_df[picks_all_df.px<pmax]
    picks_all_df=picks_all_df[picks_all_df.px>pmin]
    
    picks_all_df.reset_index(drop=True,inplace=True)
      
    # Calculate the common midpoint time values
    starttime=UTCDateTime(semblance_out['starttime'])
    mid_ts=[]
    utcpick=[]
    for i in range(len(picks_all_df)):
        t0=picks_all_df.iloc[i].t-(picks_all_df.iloc[i].px)*picks_all_df.iloc[i].x
        mid_ts.append(picks_all_df.iloc[i].px*midpt+t0)
        utcpick.append(starttime+picks_all_df.iloc[i].t)
    picks_all_df['midt']=mid_ts
    picks_all_df['utcpick']=utcpick
    
    # trigger times windowing individual events
    triggers={'event':[],'on':[],'off':[]}
    tmax=np.max(semblance_out['t'])
    tmin=np.min(semblance_out['t'])
    
    bins,bin_length=binning(picks_all_df['midt'],binsize,tmin,tmax)
    bin_length_ma=moving_average(bin_length, n=3)
    bins_ma=moving_average(bins, n=3)
    
    events=[]
    event=event-1
    onoff=0
    for i in range(len(bins_ma)):
        
        if i==len(bins_ma)-1:
            # Adds an off trigger if end of file is reached
            if onoff==1:
                triggers['off'].append(bins_ma[i]+binsize)

        elif bin_length_ma[i]>trigger_val:

            if onoff==0:
                onoff=1
                event=event+1
                triggers['event'].append(event)
                triggers['on'].append(bins_ma[i]-binsize)

            events.append(event)


        else:
            if onoff==1:
                onoff=0  
                triggers['off'].append(bins_ma[i]+binsize)

            events.append(0)
            
    triggers_df=pd.DataFrame(triggers)        
    
    # Assigned an event number to the picks bassed on trigger times 
    event_ids=np.zeros(len(picks_all_df))
    for i in range(len(picks_all_df)):
        tmp=picks_all_df.iloc[i]
        
        for j in range(len(triggers_df)):

            if tmp.midt>triggers_df.iloc[j].on and tmp.midt<triggers_df.iloc[j].off:
                event_ids[i]=triggers_df.iloc[j].event

    picks_all_df['event_id']=event_ids
    
    # Plot picks
    if plot==True:
        x_tmps=np.arange(0,1000,100)
        plt.figure(figsize=[5,10])
        for p in range(len(picks_all_df)):
            t0=picks_all_df.iloc[p].t-(picks_all_df.iloc[p].px)*picks_all_df.iloc[p].x
            
            t_tmps=[]

            for x_tmp in x_tmps:
                t_tmps.append((picks_all_df.iloc[p].px)*x_tmp+t0)

            plt.plot(x_tmps,t_tmps)
            
        plt.ylim(picks_all_df.t.max(),0)
        plt.xlabel('Distance (m)',fontsize=14)
        plt.ylabel('Time (s)',fontsize=14)
        plt.grid()
        plt.show()
              
    
    return picks_all_df,triggers_df,event


def radpicker(files,event=1):
    """
    This is a wrapper function which reads in a tdms file, filters the data, applys a radon transform and picks the events.
    
    Arguments:
    Required:
    files - list of files to process produced using glob
    event - starting event number
    """
    for i in range(len(files)):   
        # Read in data
        file=files[i]
        st = utils.tdms_to_stream(file,'BPL1')
        st=utils.tracezoom(st,0,1000,0,30)
        st.filter('bandpass', freqmin=20,freqmax=120.0)

        # Filter data
        wavenumber=0.035
        max_freq=120
        st=filters.fk_filter(st, wavenumber, max_freq)
        st=filters.image_sharpen_demean(st,sigma=3,alpha=30)

        # Apply radon transform
        semblance_out=radon_slider(st,winsize=200,overlap=150,slowmin=-0.4e-3,slowmax=-0.1e-3)
        
        # Pick and apply coincidence filter
        if i==0:
            picks_all_df,triggers_all_df,event=ss_coincidence(semblance_out,event=event,threshold = 120,binsize=0.005,trigger_val=1.0,plot=False)
        else:
            picks_df,triggers,event=ss_coincidence(semblance_out,event=event,threshold = 120,binsize=0.005,trigger_val=1.0,plot=False)
            picks_all_df=pd.concat([picks_all_df,picks_df])
            triggers_all_df=pd.concat([triggers_all_df,triggers])
        
        event=event+1
        print('File %s completed'%(i))
        print('%s events detected so far'%(event))

    return picks_all_df, triggers_all_df


def convert_radpicks_to_nonlinloc(picks_all_df, nonlinloc_outdir='', phase='S', phase_pick_err=0.02):
    """Function to take radon transform detection picks from radpicker() and convert these picks to individual events and 
    nonlinloc files.
    
    Arguments:
    Required:
    picks_all_df - A pandas dataframe containing a list of event phase picks from radpicker(). (pandas df)
    Optional:
    nonlinloc_outdir - The directory to save the nonlinloc obs files to. Default is the current working directory. (str)
    phase - The phase labels to assign for nonlinloc. Should be P or S. (str)
    phase_pick_err - The time error associated with phase picks, in seconds. Default = 0.02 s (float)

    Returns:
    [events - A list of DASpy event objects] - not yet implemented.
    """
    # Get unique event ids from df:
    event_ids = picks_all_df['event_id'].unique()

    # Loop over events, writing to file:
    for event_id in event_ids:
        # Get current event df:
        df_event_curr = picks_all_df.loc[picks_all_df['event_id'] == event_id]
        # Specify file to write to:
        print("***")
        print(df_event_curr)
        print("***")
        event_uid = UTCDateTime(df_event_curr['utcpick'].iloc[0]).strftime('%Y%m%d%H%M%S%f')
        event_out_fname = os.path.join(nonlinloc_outdir, ''.join((event_uid, ".obs")))
        f_out = open(event_out_fname, 'w')

        # Loop over phase picks, writing to file:
        for index, row in df_event_curr.iterrows():
            station = "".join(("D0", str(int(row['x']))))
            date_tmp=UTCDateTime(row['utcpick'])
            pick_date = date_tmp.strftime('%Y%m%d')
            pick_hrmin = date_tmp.strftime('%H%M')
            pick_secs = date_tmp.strftime('%S.%f')
            pick_err = str(phase_pick_err)
            # Write to file:
            f_out.write(" ".join((station, "? ? ?", phase, "?", pick_date, pick_hrmin, pick_secs, "GAU", pick_err, "0.0 0.0 0.0 1.0", "\n")))
        
        # And close file for current event:
        f_out.close()

    
        
    
    
    