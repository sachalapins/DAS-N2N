# ------------------- Import neccessary modules ------------------
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach

# ------------------- End: Import neccessary modules ------------------

# ------------------- Define generally useful functions/classes -------------------

class e3d_model:
    "Class to store e3d model parameters"
    
    def __init__(self,model_name):
        self.model_name=model_name
        self.velocity_model={}
        self.model_parameters={}
        self.receivers={}
        self.source={}
        
    def import_velocity(self,fname,units='m'):
        """
        Import velocity model from file
        
        Arguments:
        Required:
        fname - velocity file in format [depth vp vs rho]
        Optional:
        units - file units. Needs to be km for e3d. 
        """
        velmod=pd.read_csv(fname,sep=' ',names=['depth','vp','vs','rho'])
        
        if units =='m':
            
            velmod.depth=velmod.depth/1000
            velmod.vp=velmod.vp/1000
            velmod.vs=velmod.vs/1000
            velmod.rho=velmod.rho/1000 
        
        self.velocity_model['depth']=velmod.depth.values
        self.velocity_model['vp']=velmod.vp.values
        self.velocity_model['vs']=velmod.vs.values
        self.velocity_model['rho']=velmod.rho.values

    def plot_velocity(self):
        
        """
        Quick plotting function of velocity profile
        to do: added attenuation to the model
        """
        
        plt.figure(figsize=[2,6])
        plt.plot(self.velocity_model['vp'],self.velocity_model['depth'],'r',label='vp')
        plt.plot(self.velocity_model['vs'],self.velocity_model['depth'],'b',label='vs')

        plt.ylim(np.max(self.velocity_model['depth']),np.min(self.velocity_model['depth']))
        
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Depth (km)')
        plt.legend()
        plt.grid()
        plt.show()
        
    def assign_model_parameters(self,xmax,zmax,dh,duration):
        """
        Define the key model parameters
        Arguments:
        Required:
        xmax - Maximum length of model profile
        zmax - Maximum depth of model profile
        dh - Cell size. This is dependent on the minimum wavelength
        duration - Time duration of the model
        """
        self.model_parameters['xmax']=xmax
        self.model_parameters['zmax']=zmax
        self.model_parameters['dh']=dh
        self.model_parameters['duration']=duration
        
    def position_receivers(self,xstart,xend,dx=0,nrec=0,zstart=0,zend=0):
        """
        Define receiver locations
        Arguments:
        Required:
        xstart - First receiver x location in km
        xend - Last receiver x location in km
        nrec - Number of receivers
        Optional:
        zstart - First receiver z location in km
        zend - Last receiver z location in km        
        """
        
        try:
            if dx !=0:
                recxs=np.arange(xstart,xend+dx,dx)
                reczs=np.linspace(zstart,zend,len(recxs))
            elif nrec!=0:
                recxs=np.linspace(xstart,xend,nrec)
                reczs=np.linspace(zstart,zend,nrec)
                
            self.receivers['recxs']=recxs
            self.receivers['reczs']=reczs
        except:
            print('Define either dx or nrec')
            
    def define_source(self,srcx,srcz,src_type=1,freq=50,amp=1e+16,Mxx=1,Myy=1,Mzz=1,Mxy=0,Mxz=0,Myz=0):
        """
        Define the source location and type
        Arguments:
        Required:
        srcx - x-coordinate of source
        srcz - z-coordinate of source
        src_type - Source types. 1: Explosive (p-wave)
                                 4: Moment tensor

        """
        self.source['srcx']=srcx
        self.source['srcz']=srcz
        self.source['freq']=freq
        self.source['amp']=amp
        self.source['src_type']=src_type
        
        if src_type==4:
            self.source['mt']=[Mxx,Myy,Mzz,Mxy,Mxz,Myz]
            
            
    def plot_model(self):
        
        """
        Quick plotting function of model dimensions.
        To do: add velocity model
        """
        
        plt.figure(figsize=[10,5])
        
        plt.scatter(self.receivers['recxs'],self.receivers['reczs'],marker='v')
        if self.source['src_type']==4:
            beachplt = beach(self.source['mt'], xy=(self.source['srcx'],self.source['srcz']), width=self.model_parameters['xmax']*0.05)
            ax = plt.gca()
            
            ax.add_collection(beachplt) 
            ax.set_aspect("equal")
            
        else:
            plt.scatter(self.source['srcx'],self.source['srcz'],marker='*',color='r',s=200)
        
        plt.axhline(y=0,c='0.5')
        plt.xlim(0,self.model_parameters['xmax'])
        plt.ylim(self.model_parameters['zmax'],-0.1*self.model_parameters['zmax'])
        
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.grid()
        plt.show()
        
    def create_e3d_file(self,path='./'):
        """
        Function to create an e3d input file
        Arguments:
        Optional:
        path - path to save file e.g. 'model/'
        """
        dt=0.606*self.model_parameters['dh']/np.max(self.velocity_model['vp']) # dt needs to satify the courant condition
        t=int(self.model_parameters['duration']/dt)
        
        # Check path exists, if not create one
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Create e3d parameter file
        f=open('%s%s_e3dmodel.txt'%(path,self.model_name),'w')
        f.write("grid x=%s z=%s dh=%s b=2 q=1\ntime dt=%0.5f t=%s\n"%(self.model_parameters['xmax'],self.model_parameters['zmax'],self.model_parameters['dh'],dt,t))
        f.write("block p=%s s=%s r=%s Q=20 Qf=50\n"%(self.velocity_model['vp'][0],self.velocity_model['vs'][0],self.velocity_model['rho'][0]))
        
        for i in range(1,len(self.velocity_model['vp'])-1):
            f.write("block p=%s s=%s r=%s z1=%s z2=%s Q=20 Qf=50\n"%(self.velocity_model['vp'][i],self.velocity_model['vs'][i],self.velocity_model['rho'][i],
                                                                     self.velocity_model['depth'][i],self.velocity_model['depth'][i+1]))
        
        f.write("block p=%s s=%s r=%s z1=%s z2=%s Q=20 Qf=50\n\n"%(self.velocity_model['vp'][i+1],self.velocity_model['vs'][i+1],self.velocity_model['rho'][i+1],
                                                                   self.velocity_model['depth'][i+1],self.model_parameters['zmax'])) # extend to the based of the model  
        
        f.write("visual movie=5\n\n")

        if self.source['src_type']!=4:
            f.write("source type=%s x=%s z=%s freq=%s amp=%s\n\n"%(self.source['src_type'],self.source['srcx'],self.source['srcz'],self.source['freq'],self.source['amp'])) 
        else:
            f.write("source type=%s x=%s z=%s freq=%s amp=%s Mxx=%s Myy=%s Mzz=%s Mxy=%s Mxz=%s Myz=%s\n\n"%(self.source['src_type'],self.source['srcx'],self.source['srcz'],self.source['freq'],self.source['amp'],self.source['mt'][0],self.source['mt'][1],self.source['mt'][2],self.source['mt'][3],self.source['mt'][4],self.source['mt'][5])) 

        for r in range(len(self.receivers['recxs'])):
            f.write('sac x=%0.3f z=%0.3f file=%s\n'%(self.receivers['recxs'][r],self.receivers['reczs'][r],self.model_name))

        f.write("visual sample=0.1 movie=1 scale=10000000000/n")
        f.close()
        
        print('File created: %s%s_e3dmodel.txt'%(path,self.model_name))