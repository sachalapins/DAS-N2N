import numpy as np 
from scipy import signal
from scipy.signal import butter, lfilter
from scipy import ndimage

from obspy import Stream, Trace
from obspy.signal.trigger import recursive_sta_lta,trigger_onset
import matplotlib.pyplot as plt

import pygimli as pg
from disba import PhaseDispersion

from DASpy.IO import utils
from DASpy.filters import filters
from DASpy.detect import rad_detect

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def shot_time(st,on=5,plot=False):
    '''Applies a bp filter then an sta/lta to detect the shot time 
    on traces when no trigger time has been recorded
    Arguments:
    st - Stream of DAS data   
    on - sta/lta trigger level
    plot - plot results
    Returns:
    trigger - array with triggers times from start of trace
    '''
    
    st.filter('bandpass', freqmin=200.0,freqmax=400.0)
    data=utils.stream2array(st)
    fs=st[0].stats.sampling_rate
    
    triggers=[]
    delta=1/fs
    lowcut = 100
    highcut = 2000

    yhistsum=np.sum(abs(data),axis=0)/data.shape[0]
    yhistsum=yhistsum/yhistsum.max()
    
    window = signal.tukey(len(yhistsum),alpha=0.01)
    y = butter_bandpass_filter(yhistsum*window,lowcut, highcut, fs, order=3)
    
    cft = recursive_sta_lta(y, int(0.01 * fs), int(0.5 * fs))
    on_of=trigger_onset(cft, on, 1)
    
    for i in range(len(on_of)):
        triggers.append(round(on_of[i,0]/fs,3))
        
    if plot==True:
        time=np.arange(0,len(yhistsum)*delta,delta)
        
        plt.plot(time,yhistsum,'0.5')
        plt.plot(time,y,'k')
        plt.ylabel('Amplitude',fontsize=14)
        plt.xlabel('Time (s)',fontsize=14)        
        plt.show()
        
        plt.plot(time,cft)
        plt.axhline(y=on,ls='--')
        plt.ylabel('STA/LTA',fontsize=14)
        plt.xlabel('Time (s)',fontsize=14)
        plt.show()
        
    return triggers

# def dispersion(st,freqmin=5,freqmax=51,npx=51,wavenumber=0.5,plot=True):
#     '''
#     Creates a dispersion array using radon transform
#     Arguments:
#     st - Stream of DAS data   
#     freqmin - minimum frequeency of dispersion curve
#     freqmax - maximum frequeency of dispersion curve
#     nvels - number of velocity values for radon transform
#     plot - plot results?
#     Returns:
#     trigger - array with triggers times from start of trace
#     '''
    
#     freqs=np.arange(freqmin,freqmax,1)
#     array=np.empty((len(freqs),npx),dtype=float) # initialize
#     resamplingrate=4*freqmax
#     st.resample(resamplingrate)
    
#     for f in range(len(freqs)):
#         freq=freqs[f]
#         st_flt=st.copy()
#         st_flt.filter('bandpass', freqmin=freq-0.5,freqmax=freq+0.5)
#         st_fk=filters.fk_filter(st_flt, wavenumber, freq+4)

#         win=int(50/freq)
#         img = utils.stream2array(st_fk)

#         for i in range(len(st_fk)-win):
#             st_fk[i].data=img[i:i+win,:].sum(axis=0)
#         print(win)
#     #     fig=plot.image(st_fk,style=2,skip=1,clim=[-2,2])
#         semblance_out=rad_detect.radon_slider(st_fk,winsize=390,overlap=150,slowmin=-2e-3,slowmax=-0.35e-3,npx=npx)
        
#         if plot == True:
#             for windidx in range(0,len(semblance_out['chidxs'])):
#                 rad_detect.radon_plot(semblance_out,windidx)
#                 plt.show()

#         dispersion_tmp=semblance_out['semblance'][0,:,int(0.8*resamplingrate):int(1.4*resamplingrate)].sum(axis=1)

#         array[f,:]=dispersion_tmp

#         print('Frequency %s complete'%(freq))
        
#     return array,semblance_out['px']


class masw:
    "Class for creating dispersion curves"
    
    def __init__(self,name):
        self.name=name
        self.dispersion_data={}
        
    def dispersion_create(self,st,winsize=390,freqmin=5,freqmax=51,npx=51,wavenumber=0.5,plot=True):
        '''
        Creates a dispersion array using radon transform
        Arguments:
        st - Stream of DAS data   
        freqmin - minimum frequeency of dispersion curve
        freqmax - maximum frequeency of dispersion curve
        nvels - number of velocity values for radon transform
        plot - plot results?
        Returns:
        trigger - array with triggers times from start of trace
        '''
        freqs=np.arange(freqmin,freqmax,1)
        array=np.empty((len(freqs),npx),dtype=float) # initialize
        resamplingrate=4*freqmax
        st.resample(resamplingrate)

        for f in range(len(freqs)):
            freq=freqs[f]
            st_flt=st.copy()
            st_flt.filter('bandpass', freqmin=freq-0.5,freqmax=freq+0.5)
            st_fk=filters.fk_filter(st_flt, wavenumber, freq+4)

            win=int(50/freq)
            img = utils.stream2array(st_fk)

            for i in range(len(st_fk)-win):
                st_fk[i].data=img[i:i+win,:].sum(axis=0)
            print(win)
        #     fig=plot.image(st_fk,style=2,skip=1,clim=[-2,2])
            semblance_out=rad_detect.radon_slider(st_fk,winsize=winsize,overlap=150,slowmin=-2e-3,slowmax=-0.35e-3,npx=npx)

            if plot == True:
                for windidx in range(0,len(semblance_out['chidxs'])):
                    rad_detect.radon_plot(semblance_out,windidx)
                    plt.show()

            dispersion_tmp=semblance_out['semblance'][0,:,int(0.8*resamplingrate):int(1.4*resamplingrate)].sum(axis=1)

            array[f,:]=dispersion_tmp

            print('Frequency %s complete'%(freq))

        self.dispersion_data['raw_array']=array
        self.dispersion_data['freqs']=freqs
        self.dispersion_data['px']=semblance_out['px']
        self.dispersion_data['vels']=abs(1/semblance_out['px'])
#         return array,semblance_out['px']

    def plot(self,dtype=2,picks=None):
        '''
        Plotting function
        dtype - 1=raw data
                2=processed data
        '''
        
        freqs=self.dispersion_data['freqs']
        
        Y,X = np.meshgrid(self.dispersion_data['vels'],self.dispersion_data['freqs'])
        
        fig, ax = plt.subplots(figsize=(10,4))
        
        if dtype==1:
            pcmesh = ax.pcolormesh(X, Y, self.dispersion_data['raw_array'],cmap='jet')
            
        if dtype==2:           
            pcmesh = ax.pcolormesh(X, Y, self.dispersion_data['processed_array'],cmap='jet')
            
        if picks !=None:
            plt.scatter(freqs+0.5,picks)

        plt.xlim(np.min(freqs),np.max(freqs))
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Velocity (m/s)', fontsize=14)
        plt.show()
            
        
    def process(self,normalise=True,sharpen=True,sigma=2,alpha=50,plot=True):
        '''
        Processing function for the dispersion data'''
              
        Y,X = np.meshgrid(self.dispersion_data['vels'],self.dispersion_data['freqs'])
        disp_data=np.copy(self.dispersion_data['raw_array'])
        
        if normalise==True:
                      
            for h in range(len(disp_data)):
                disp_data[h]=disp_data[h]/np.max(disp_data[h])
                
        if sharpen==True:
            blurred_f = ndimage.gaussian_filter(disp_data, sigma)
            filter_blurred_f = ndimage.gaussian_filter(blurred_f, sigma)

            disp_data = blurred_f + alpha * (blurred_f - filter_blurred_f)        
        
        self.dispersion_data['processed_array']=disp_data
        
        if plot==True:
            self.plot()
        
            
    def pick(self,plot=True):
        '''
        Picks the maximum velocity value for each frequency.
        '''
        disp_data=self.dispersion_data['processed_array']
        
        dis_pos=[]
        for i in range(len(disp_data)):
            im_tmp=disp_data[i,:]

            if len(dis_pos) != 0:
                lims=dis_pos[i-1]
                m=np.max(im_tmp[lims-5:lims+2])
            else:
                m=np.max(im_tmp)

            pos = [i for i, j in enumerate(im_tmp) if j == m]
            dis_pos.append(pos[0])         


        vel_max=[]
        for j in range(len(dis_pos)):
            vel_max.append(self.dispersion_data['vels'][dis_pos[j]])
            
        self.dispersion_data['dispersion']=vel_max
        
        if plot==True:
            self.plot(picks=vel_max)
            
        return {'freqs':self.dispersion_data['freqs'],'dispersion':self.dispersion_data['dispersion']}

class DCModelling(pg.Modelling):
    def __init__(self, freq, mat,possion=0, verbose=False):
        super().__init__()
        self.freq=freq
        self.thk=mat[:,0]
        self.vp=mat[:,1]
        self.dens=mat[:,3]
        self.wavetype="rayleigh"
        self.possion=possion
        self.mesh_ = pg.meshtools.createMesh1D(len(self.thk))
        self.setMesh(self.mesh_)
    def vs2vp(self,vs):
        possion=self.possion
        vp=np.sqrt(2*(1-possion)/(1-2*possion))*vs
        return vp
    def response(self,model):
     #Thickness(km),Vp(km/s),Vs(km/s),Rho(g/cm3)
        if self.possion==0:
            vp=self.vp
        else:
            vp=self.vs2vp(model)
            
    
        mat = np.vstack((self.thk,vp, model, self.dens)) / 1e3
        #print(mat)
        pd = PhaseDispersion(*mat,dc=0.0001)
        cp = pd(1./ self.freq, mode=0, wave=self.wavetype)
        vr = cp.velocity*1000  # (km/s->m/s)

        return vr #phase velocity
        
def create_start_model(obs,mat,freqmin,freqmax, plot=True):
    '''Create starting model and plot with observed data.
    
    Arguments:
    Required:
    obs - Observed rayleigh wave velocities.
    freqmin - minimum frequency of rayleigh waves.
    freqmax - maximum frequency of rayleigh waves.
    mat - initial velocity model

    Returns:
    f - starting model.''' 
    
    freq=np.linspace(7,51,len(obs))
    f =DCModelling(freq, mat,possion=0.4) # G(freq)=vr 
    vs=mat[:,2]
    vr=f.response(vs)
    
    if plot == True:
        plt.plot(freq, vr,'b-')

        plt.plot(obs.freqs,obs.dispersion,'rx')
        plt.ylabel('Velocity (m/s)',fontsize=14)
        plt.xlabel('Frequency (Hz)',fontsize=14)
        plt.grid()
        plt.show()
        
    return f
    
