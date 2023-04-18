import numpy as np 
import obspy
from obspy import Stream, Trace
from obspy import UTCDateTime
from obspy.io.segy.segy import SEGYTraceHeader
from DASpy.IO import tdms_reader 
from DASpy.filters import filters
import sys, os 
import gc 
import glob 
import pandas as pd 


def array2stream(nparray, network, fs, start, channel_spacing=1,units='Strain rate'):
    """
    Converts numpy array format DAS data into a obspy stream.
    
    Arguments:
    Required:
    np_array - numpy array of DAS data
    network - seismic network name
    fs - sample rate in Hz
    start - data start time
    
    Optional:
    channel_spacing - spacing in metres 
    units - DAS units

    Returns:
    st - obpy stream
    """ 
    st=Stream()
    count=0
    for channel in range(len(nparray.T)):
        tr=Trace(data=nparray.T[channel])
        tr.data = np.require(tr.data, dtype=np.float32)
        tr.stats.network=network
        tr.stats.station = ''.join(('D', '{0:04d}'.format(count)))
        tr.stats.channel='Z'
        tr.stats.starttime=start

        tr.stats.sampling_rate=fs
        tr.stats.distance=channel*channel_spacing
        
        if units!='None':
            tr.stats.units = units
            
        st.append(tr)       
        count+=1

    return st
        
def stream2array(st):
    """
    Populates a 2D np.array that is the traces as rows by the samples as cols.
    
    Arguments:
    Required:
    st - obspy stream

    Returns:
    nparray - numpy array
    """
    
    nparray=np.empty((len(st),len(st[0].data)),dtype=float) # initialize
    for index,trace in enumerate(st):
        nparray[index,:]=trace.data

    return nparray

def tdms_to_stream(data_path,network, **kwargs):
    """
    Wrapper for the array2stream function. Reads in TDMS data and outputs an obspy stream.
    
    Arguments:
    Required:
    data_path - path for TDMS file
    network - seismic network name

    Returns:
    obspy stream
    """
    
    tdms=tdms_reader.TdmsReader(data_path)
        
    props = tdms.get_properties()
    zero_offset = props.get('Zero Offset (m)') 
    channel_spacing = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
    n_channels = tdms.fileinfo['n_channels']
    depth = zero_offset + np.arange(n_channels) * channel_spacing
    fs = props.get('SamplingFrequency[Hz]')
    das_start=UTCDateTime(props['ISO8601 Timestamp'])
    if 'first_ch' in kwargs:
        data=tdms.get_data(**kwargs)
    else:
        data=tdms.get_data(first_ch=abs(int(zero_offset)), **kwargs)
    
    return array2stream(data,network,fs,das_start,channel_spacing=channel_spacing)


def stream2segy(st,path,id):
    """
    SEGY writer for DAS data.
    Note: appears to remove ms from time.  
    
    Arguments:
    Required:
    st - obspy stream
    path - path of segy file 
    id - unique file name

    Returns:
    obspy stream
    """
    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32)
    
        if not hasattr(tr.stats, 'segy.trace_header'):
            tr.stats.segy = {}
            tr.stats.segy.trace_header = SEGYTraceHeader()
            tr.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace=int(tr.stats.distance*1000)

    
    network=st[0].stats.network
    if network=="":
        network="xx"

    if not os.path.exists(path):
        os.makedirs(path)

    file="%s.%s.segy"%(network,id)
    filename="%s%s"%(path,file)
    print(filename)
    st.write(filename,format='segy', data_encoding=5)
    

def segy2stream(path,network,channel_spacing=1,units='Strain rate'):
    """
    Reads in SEGY file and outputs and obspy stream with correct header information.
    
    Arguments:
    Required:
    st - obspy stream
    path - path of segy file 
    id - unique file name

    Returns:
    obspy stream
    """
    
    st=obspy.read(path,format='SEGY')
    
    for i in range(len(st)):
        
#         st[i].stats.distance=(tr.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace)/1000
        st[i].stats.distance=(i*channel_spacing)
        st[i].stats.network=network
        st[i].stats.channel='Z'
        st[i].stats.units = units
    
    return st


def convert_das_tdms_to_mseed(tdms_data_dir, mseed_out_dir, first_last_channels=[0,1000], network_code="AA", station_prefix="D", spatial_down_samp_factor=10, fk_filter_params={}, duplicate_Z_and_E=True, fold=False, apply_notch_filter=False, notch_freqs=[], notch_bw=2.5):
    
    """
    Function to read in tdms files and export them into mseed, in a format supported by QMigrate.
    Note: Works best with DAS data split into less than 1 hour chunks.
    
    """
    
    # Get tdms file list:
    tdms_flist = glob.glob(os.path.join(tdms_data_dir, '*.tdms'))
    tdms_flist.sort()

    # Get start time of all data:
    st_das = tdms_to_stream(tdms_flist[0],network_code)
    das_start = st_das[0].stats.starttime
    start_year = das_start.year
    start_day = das_start.julday
    start_hour = das_start.hour
    os.makedirs(os.path.join(mseed_out_dir, str(start_year).zfill(4), str(start_day).zfill(3)), exist_ok=True)

    # Create station labels:
    das_station_labels = []
    das_station_idxs = []
    # data_tmp = tdms.get_data(first_last_channels[0], first_last_channels[1])
    # if fold:
    #     num_channels = int(data_tmp.shape[1]/2.)
    # else:
    #     num_channels = int(data_tmp.shape[1])
    num_channels = len(st_das)
    # del data_tmp
    # gc.collect()
    for i in np.arange(0, num_channels, int(spatial_down_samp_factor), dtype=int):
        das_station_labels.append(''.join((station_prefix, str(i).zfill(4))))
        das_station_idxs.append(i)

    # Loop over tdms files, importing and writing to mseed:
    for tdms_fname in tdms_flist:
        print("Processing for tdms file:", tdms_fname)
        # Import tdms data as stream:
        st_das = tdms_to_stream(tdms_fname,network_code)
        starttime_curr = st_das[0].stats.starttime
        fs = st_das[0].stats.sampling_rate
        channel_spacing = st_das[1].stats.distance - st_das[0].stats.distance
        endtime_curr = st_das[0].stats.endtime

        # # Fold data, if specified:
        # if fold:
        #     # Perform check:
        #     if not (data.shape[1]) % 2 == 0:
        #         print('First/last channels are not correctly specified for folding data. Exiting.')
        #         sys.exit()
        #     fold_point = int(data.shape[1]/2.)
        #     data_part_1 = data[:,0:fold_point]
        #     data_part_2 = np.flip(data[:, fold_point:], axis=1)
        #     data_folded = (data_part_1 + data_part_2)/2.
        #     data = data_folded
        #     del data_folded, data_part_1, data_part_2
        #     gc.collect()
        #     print('Data has been folded.')
        if fold:
            print("Warning: Folding not currently implemented.")

        # Filter data:

        # Apply fk filter:
        if len(list(fk_filter_params.keys())) > 0:
            print('Applying fk filter')
            st_das = filters.fk_filter(st_das, fk_filter_params['wavenumber'], fk_filter_params['max_freq'])
        # Apply notch filter:
        if apply_notch_filter:
            print('Applying notch filter/s')
            for f_notch in notch_freqs:
                st_das = filters.notch(st_das, f_notch, notch_bw)

        # Make dir for current mseed data, if not made already:
        mseed_curr_day_dir = os.path.join(mseed_out_dir, str(starttime_curr.year).zfill(4), str(starttime_curr.julday).zfill(3))
        os.makedirs(mseed_curr_day_dir, exist_ok=True)
        
        # Loop over das channels to save:
        for i in range(len(das_station_labels)):
            # Get st to write to:
            das_station_curr = das_station_labels[i]
            st_fname = os.path.join(mseed_curr_day_dir, ''.join((str(starttime_curr.year).zfill(4), str(starttime_curr.julday).zfill(3), '_',
                                 str(starttime_curr.hour).zfill(2), '0000_', das_station_labels[i], '_N2.m')))
            try:
                st_working = obspy.read(st_fname)
            except FileNotFoundError:
                st_working = obspy.Stream()

            # Add data to stream:
            # Create trace:
            tr_to_add = st_das.select(station=das_station_curr)[0]
            tr_to_add.stats.channel = "EHN"
            tr_to_add.stats.network = network_code
            # Append trace to stream:
            st_working.append(tr_to_add)
            st_working.merge(method=1, interpolation_samples=0)

            # Check if data is masked and unmask if it is:
            if np.ma.is_masked(st_working[0].data):
                data_tmp = st_working[0].data.copy()
                data_tmp = data_tmp.filled(fill_value=0.0)
                st_working[0].data = data_tmp
                del data_tmp
                gc.collect()

            # And deal with overlapping hours:
            if endtime_curr.hour > starttime_curr.hour:
                st_working_new_hour = st_working.copy()
                st_working.trim(starttime=obspy.UTCDateTime(year=starttime_curr.year, julday=starttime_curr.julday, hour=starttime_curr.hour), 
                                endtime=obspy.UTCDateTime(year=starttime_curr.year, julday=starttime_curr.julday, hour=starttime_curr.hour+1))
                st_working_new_hour.trim(starttime=obspy.UTCDateTime(year=starttime_curr.year, julday=starttime_curr.julday, hour=starttime_curr.hour+1), 
                                endtime=obspy.UTCDateTime(year=starttime_curr.year, julday=starttime_curr.julday, hour=starttime_curr.hour+2))

            # And write data to stream:
            if endtime_curr.hour > starttime_curr.hour:
                st_fname_plus_one_hour = os.path.join(mseed_curr_day_dir, ''.join((str(starttime_curr.year).zfill(4), str(starttime_curr.julday).zfill(3), '_',
                                 str(starttime_curr.hour+1).zfill(2), '0000_', das_station_labels[i], '_N2.m')))
                st_working_new_hour.write(st_fname_plus_one_hour, format="MSEED")
                # del st_working_new_hour
            st_working.write(st_fname, format="MSEED")
            # del st_working
            # gc.collect()

            # Duplicate for Z and E components (arbitarily set equal to N comp):
            if endtime_curr.hour > starttime_curr.hour:
                st_working_new_hour[0].stats.channel = "EHZ"
                st_working_new_hour[0].data = (1.0e-12)*np.random.rand(len(st_working_new_hour[0].data))
                st_working_new_hour.write(st_fname_plus_one_hour[:-4]+'Z2.m', format="MSEED")
                st_working_new_hour[0].stats.channel = "EHE"
                st_working_new_hour.write(st_fname_plus_one_hour[:-4]+'E2.m', format="MSEED")
                del st_working_new_hour
            st_working[0].stats.channel = "EHZ"
            st_working[0].data = (1.0e-12)*np.random.rand(len(st_working[0].data))
            st_working.write(st_fname[:-4]+'Z2.m', format="MSEED")
            st_working[0].stats.channel = "EHE"
            st_working.write(st_fname[:-4]+'E2.m', format="MSEED")
            del st_working
            gc.collect()

        # Tidy:
        del st_das
        gc.collect()

    print("Finished converting tdms files to mseed.")


def create_das_stations_file(fibre_lats, fibre_lons, fibre_elevs, dist_between_samps_m, station_dowsample_factor, out_fname, station_static_disp_offset=0.0):
    """Function to find das station coords and write to file for input to QMigrate.
    Note that elevations must be in km."""
    
    # Get coords of individual sampling point station coords:
    num_stat_between_turning_points = np.zeros(len(fibre_lats) - 1)
    for i in range(len(fibre_lats)-1):
        av_elev = ( fibre_elevs[i] + fibre_elevs[i+1] ) / 2.
        dist_between_turning_points_curr_m, azi_ab, azi_ba = obspy.geodetics.base.gps2dist_azimuth(fibre_lats[i], fibre_lons[i], fibre_lats[i+1], fibre_lons[i+1], a=6378137.0+av_elev)
        num_stat_between_turning_points[i] = int(round(dist_between_turning_points_curr_m / ( float(station_dowsample_factor) * dist_between_samps_m ) ))
    all_station_lats = []
    all_station_lons = []
    all_station_elevs = []
    for i in range(len(num_stat_between_turning_points)):
        for j in range(int(num_stat_between_turning_points[i])):
            curr_lat = float(j) * ( ( fibre_lats[i+1] - fibre_lats[i] ) / float(num_stat_between_turning_points[i]) ) + fibre_lats[i]
            curr_lon = float(j) * ( ( fibre_lons[i+1] - fibre_lons[i] ) / float(num_stat_between_turning_points[i]) ) + fibre_lons[i]
            curr_elev = float(j) * ( ( fibre_elevs[i+1] - fibre_elevs[i] ) / float(num_stat_between_turning_points[i]) ) + fibre_elevs[i]
            all_station_lats.append(curr_lat)
            all_station_lons.append(curr_lon)
            all_station_elevs.append(curr_elev)
     
    # Write coords to station df:
    stations_df = pd.DataFrame(columns=['Latitude','Longitude','Elevation','Name'])
    for i in range(len(all_station_lats)):
        station_label = ''.join(('D',str(int(i*station_dowsample_factor + station_static_disp_offset)).zfill(4)))
        lat = all_station_lats[i]
        lon = all_station_lons[i]
        elev = all_station_elevs[i]
        stations_df = stations_df.append({'Latitude': lat, 'Longitude': lon, 'Elevation': elev, 'Name': station_label}, ignore_index=True)

    # And write data to csv:
    stations_df.to_csv(out_fname, index=False)
    print("Saved stations data to:", out_fname)
    

def tracezoom(st,d1,d2,t1,t2):
    
    """
    Function to zoom in on a specific section of the trace.
    
    Arguments:
    Required:
    st - obspy stream
    d1 - start zoom distance 
    d2 - end zoom distance
    t1 - start zoom time 
    t2 - end zoom distance

    Returns:
    obspy stream
    """
    
    st_tmp=Stream()

    for tr in st:
        if tr.stats.distance >d1 and tr.stats.distance <d2:
            st_tmp.append(tr)

    start=st_tmp[0].stats.starttime

    return st_tmp.slice(start+t1,start+t2)