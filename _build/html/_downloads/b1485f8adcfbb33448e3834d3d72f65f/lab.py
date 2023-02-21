import numpy as np
import matplotlib.pyplot as plt
import cartopy



from scipy import optimize, linalg, stats, signal
import obspy


from obspy import Stream
from obspy.signal.util import _npts2nfft



######################################################################################
def setup_map(circle=None, extent=None, color=None, label=None, scale=1.1):
    """
    \t\n
    Return a Figure object with a geoaxes object from cartopy as axis.
    
    Parameters:
    -----------
    circle:   dictionary with keywords "longitude" and "latitude" used as 
              the centre of the map and "maxradius" as the radius [in degrees] 
              of the circle that must be contained in it.
    scale:    float that is used to scale the map extent [1.1 by default].
    extent:   array-like with min and max longitudes and min and max latitudes
              that is used to set the extent instead of the one derived from circle.
              If neither circle or extent are provided, the extent will be global.
    color:    color used to draw the circle and its centre [by default the circle is not drawn]
    label:    used for annotate the circle centre
    
    Return:
    -------
    fig:      the Figure object
        
    """

    if circle: 
        rad = circle["maxradius"]
        lon = circle["longitude"]
        lat = circle["latitude"]
        rad_m = rad * np.pi/180 * 6371009

    if extent:
        lon0 = np.mean(extent[:2])
        lat0 = np.mean(extent[2:])
    elif circle:
        lon0 = lon
        lat0 = lat
    else: 
        lon0,lat0 = None, None

    if lon0 is None: 
        projection = cartopy.crs.PlateCarree()
    else:
        projection = cartopy.crs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
        
    fig,ax = plt.subplots(tight_layout=True, figsize=(8,8), subplot_kw=dict(projection=projection))

    if extent is not None:
        ax.set_extent(extent)
    elif circle is not None:   
        extent = scale*np.array([-rad_m,rad_m,-rad_m,rad_m])
        ax.set_extent(extent,projection)
    else:
        ax.set_global()
        
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS,color="gray");

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels, gl.left_labels = False, False
    
    if circle and color:   
        rad_km = rad * np.pi/180 * 6371.009
        ax.tissot(rad_m/1e3, lon, lat, alpha=0.2, linewidth=2, edgecolor=color, facecolor="none")
        ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color, zorder=30)
    if label:
        marker = ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color, zorder=30)
        ax.annotate(label,(lon,lat-scale*rad/30), transform=cartopy.crs.PlateCarree(), color=marker._facecolors[0], ha='center',va="top",zorder=30)
         
    return fig
######################################################################################







######################################################################################
def preprocessing(original_stream,order=1,taper=0.05):
    stream = original_stream.copy()
    stream.detrend(type="polynomial",order=order)
    stream.taper(taper,type="cosine")
    for trace in stream:
        n, d = trace.stats.npts, trace.stats.delta
        nn = _npts2nfft(n)
        T = (nn-n)*d
        endtime = trace.stats.endtime + T
        trace.trim(endtime=endtime,pad=True,fill_value=0)
    return stream
######################################################################################



######################################################################################
def get_inverse_response(trace,output="DEF",last_stage=False):
    
    response = trace.stats.response
    end_stage = len(response.response_stages)
    if not last_stage: end_stage -= 1
    
    n, d = trace.stats.npts, trace.stats.delta    
    Rs, fs = response.get_evalresp_response(d,n,output=output,end_stage=end_stage)
    Rs[0] = np.inf
    IRs = 1/Rs
    
    return fs,IRs
######################################################################################



######################################################################################
def remove_response(original_stream,output="DEF"):
    stream = original_stream.copy()
    for trace in stream:
        fs,IRs = get_inverse_response(trace,output=output)
        fs,Zs,n,d = get_fft_trace(trace)
        trace.data = np.fft.irfft(Zs*IRs,n) / d
    return stream
######################################################################################



######################################################################################
def filtering(original_stream,BA_filters):
    stream = original_stream.copy()
    for trace,BA_filter in zip(stream,BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        trace.data = np.fft.irfft(Zs*Hs,n) / d
    return stream 
######################################################################################



######################################################################################
def filtering_zerophase(original_stream,butters):
    stream = original_stream.copy()
    for trace,butter in zip(stream,butters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*butter, ws)
        zs = np.fft.irfft(Zs*Hs,n)
        zs = np.flip(zs)
        Zs = np.fft.rfft(zs)
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        trace.data = zs
    return stream
######################################################################################



######################################################################################
def derivative(stream,order=1):
    """
    \t\n
    Return the time derivative of given order of the Stream object. The time derivative is evaluated in the frequency domain
    
    Parameters:
    -----------
    \tstream:\t Stream object to be derived
    \torder:\t Order of the time derivative
    
    Return:
    -------
    \tder_stream:  Derived Stream object
    
    Examples:
    ---------
    >>> from obspy import read
    >>> stream = read()
    >>> der_stream = derivative(stream)
    """

    der_stream = stream.copy()
    for trace in der_stream:
        fs,Zs,n,d = get_fft_trace(trace)
        ss = 2j*np.pi*fs
        ss = ss**order
        trace.data = np.fft.irfft(Zs*ss,n) / d

    return der_stream
######################################################################################



######################################################################################
def remove_mean(stream,taper=0.05):
    """
    \t\n
    It remove the mean of the first decimal percentage of the data from the whole data time series of all the trace of a Stream object. The Stream object is modified in place.
    
    Parameters:
    -----------
    \tstream:\t Stream object 
    \ttaper:\t  Decimal percentage on which calculate the mean (5% by default)
    
    Examples:
    ---------
    >>> from obspy import read
    >>> stream = read()
    >>> remove_mean(stream)
    """

    for trace in stream:
        k = int(taper*trace.stats.npts)
        y = trace.data.astype(float)
        mean = y[:k].mean()
        trace.data = y - mean
######################################################################################




######################################################################################
def get_fft_trace(trace):
    
    n, d = trace.stats.npts, trace.stats.delta
    fs = np.fft.rfftfreq(n,d)
    Zs = np.fft.rfft(trace.data) * d
    
    return fs,Zs,n,d
######################################################################################



######################################################################################
def get_end_stage(response, end=False):
    end_stage = len(response.response_stages)
    stage = response.response_stages[-1]
    gain = 1
    if not end:
        if stage.input_units == "COUNTS": 
            end_stage -= 1
            gain = stage.stage_gain
    return end_stage,gain
######################################################################################


######################################################################################
def get_fft_response(trace, output="DEF", end=False):
    
    response = trace.stats.response
    sensitivity = response.instrument_sensitivity.value

    end_stage,gain = get_end_stage(response, end=end)
    
    n, d = trace.stats.npts, trace.stats.delta    
    Rs, fs = response.get_evalresp_response(d, n, output=output, end_stage=end_stage)
    Rs *= gain
    Rs[0] = np.inf
    IRs = 1/Rs
    
    return fs,Rs,IRs,sensitivity
######################################################################################


######################################################################################
def remove_response(stream, output="DEF", end=False):
    new = stream.copy()
    for trace in new:
        fs,Rs,IRs,sensitivity = get_fft_response(trace, output=output, end=end)
        fs,Zs,n,d = get_fft_trace(trace)
        trace.data = np.fft.irfft(Zs*IRs, n) / d
    return new
######################################################################################


######################################################################################
def filtering(stream, BA_filters):
    filtered_stream = stream.copy()
    for trace,BA_filter in zip(filtered_stream, BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        trace.data = np.fft.irfft(Zs*Hs,n) / d
    return filtered_stream 
######################################################################################


######################################################################################
def filtering_zerophase(stream,BA_filters):
    filtered_stream = stream.copy()
    for trace,BA_filter in zip(filtered_stream,BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        Zs = np.fft.rfft(zs) * d
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        trace.data = zs
    return filtered_stream
######################################################################################



######################################################################################
def get_channels_with_orientation(iterable):
    channels = []
    for elem in iterable:
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code
        else:
            channel = elem.stats.channel
        if channel not in channels: channels.append(channel)
    return channels
######################################################################################


######################################################################################
def get_channels(iterable, return_indices=False):
    channels = []
    indices = []
    for k, elem in enumerate(iterable):
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code[:2]+"*"
        else:
            channel = elem.stats.channel[:2]+"*"
        if channel not in channels: 
            channels.append(channel)
            indices.append(k)
    if return_indices:
        return (channels, indices)
    else:
        return channels
######################################################################################



######################################################################################
def make_label(trace_id,label_id,extra_label):

    label = None
    
    if label_id:
        label = trace_id
        if extra_label is not None: 
            if extra_label[:2] == "\n":
                label += extra_label
            else:
                label += " "+extra_label
    else:
        if extra_label is not None: label = extra_label
    
    return label
######################################################################################



######################################################################################
def get_gaps(stream,reftime):
    
    gaps = stream.get_gaps()
    channels = get_channels_with_orientation(stream)
    dict_gaps = {}
    for channel in channels:
        dict_gaps[channel] = np.array([ [gap[4]-reftime,gap[5]-reftime] for gap in gaps if gap[3] == channel])
   
    return dict_gaps
######################################################################################


######################################################################################
def plot_time(stream, reftime=None, picks=None,
                      remove_sensitivity=False, fig=None,
                      ncol=1, col=0, sharey=False,
                      label_id=True, extra_label=None, taper=None, title=None, 
                      linewidth=1, color=None, alpha=1):
    
    logy_color = color is None

    if extra_label is not None: label_id = False
    
    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)
    
    if reftime is None: reftime = min(trace.stats.starttime for trace in stream)

    channels = get_channels_with_orientation(stream)

    dict_gaps = get_gaps(stream,reftime)
    dict_picks = get_picks(stream,reftime,picks)
    
    logy = fig is None
    
    nrow = len(channels)

    
    if logy:
        height = 2*nrow+0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(nrow,ncol,tight_layout=True,sharex="col",sharey=sharey,figsize=(10,height))
        if nrow == 1 and ncol == 1: axes = np.array([axes])
        if title is not None: fig.suptitle(title)
        axes = np.array(axes).reshape(nrow,-1)
        for ax in axes[-1]:
            ax.set_xlabel("Time [s] with respect to the origin time")
    else:
        axes = fig.axes
    axes = np.array(axes).reshape(nrow,-1)
    
    dict_axes = {}
    for ax,channel in zip(axes[:,col],channels):
        
        now = stream.select(channel=channel)    
                         
        for k,trace in enumerate(now):
    
            ts = trace.times() 
            ts += trace.stats.starttime - reftime

            if k == 0: label = make_label(trace.stats.channel,label_id,extra_label)
            else: label = None

            zs = trace.data.copy()
            if remove_sensitivity: zs = zs.astype(float) / trace.stats.response.instrument_sensitivity.value
            
            line, = ax.plot(ts,zs, label=label, color=color, linewidth=linewidth, alpha=alpha)
            if logy_color: color = line._color

            if label is not None: ax.legend()

            if trace.data.max() > 0 and trace.data.min() < 0:
                ax.axhline(0, color="gray", linewidth=0.5)
                
            if taper is not None:
                k = int(trace.stats.npts*0.05)
                T = trace.stats.delta * k
                ax.axvspan(ts[0],    ts[0]+T, color="yellow", alpha=0.2)
                ax.axvspan(ts[-1]-T, ts[-1],  color="yellow", alpha=0.2)

        for t0,t1 in dict_gaps[channel]:
            ax.axvspan(t0,t1, color="red", alpha=0.2)
        for t0,dt in dict_picks[channel]:
            ax.axvline(t0, color="green")
            ax.axvspan(t0-dt,t0+dt, color="green", alpha=0.2)
    
    if logy_trace: stream = stream[0]
    
    return fig
######################################################################################



######################################################################################
def get_gaps(stream,reftime):
    
    gaps = stream.get_gaps()
    channels = get_channels_with_orientation(stream)
    dict_gaps = {}
    for channel in channels:
        dict_gaps[channel] = np.array([ [gap[4]-reftime,gap[5]-reftime] for gap in gaps if gap[3] == channel])
   
    return dict_gaps
######################################################################################



######################################################################################
def get_picks(stream,reftime,picks):
    
    channels = get_channels_with_orientation(stream)
    dict_picks = {}
    for channel in channels:
        dict_picks[channel] = []
        if picks is None: continue
        now = stream.select(channel=channel)
        trace = now[0]
        for pick in picks:
            if trace.id == pick.waveform_id.id:
                dict_picks[channel].append([pick.time-reftime,pick.time_errors])
   
    return dict_picks
######################################################################################



######################################################################################
def get_channels(iterable, return_indices=False):
    channels = []
    indices = []
    for k, elem in enumerate(iterable):
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code[:2]+"*"
        else:
            channel = elem.stats.channel[:2]+"*"
        if channel not in channels: 
            channels.append(channel)
            indices.append(k)
    if return_indices:
        return (channels, indices)
    else:
        return channels
######################################################################################




######################################################################################
def make_label(trace_id,label_id,extra_label):

    label = None
    
    if label_id:
        label = trace_id
        if extra_label is not None: 
            if extra_label[:2] == "\n":
                label += extra_label
            else:
                label += " "+extra_label
    else:
        if extra_label is not None: label = extra_label
    
    return label
######################################################################################


######################################################################################
def plot_fft(stream, 
             fig=None, nrow=1, row=0, sharey="row", xscale="log",
             label_id=True, extra_label=None, title=None,
             bands=None, remove_sensitivity=False, 
             linewidth=1, color=None, alpha=1):

    if extra_label is not None: label_id = False


    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)

    logy = fig is None
    
    if bands is None: bands = [None for _ in stream]
    
    logy_xlabel = row+1 == nrow

    ncol = len(stream)
    c = 1
    if ncol == 6: c == 2
    ncol //= c
    nrow *= c
    row  *= c

    if logy:
        height = 3*nrow+0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(nrow, ncol, tight_layout=True, sharex=True, sharey=sharey, figsize=(10,height))
        if nrow == 1 and ncol == 1: axes = np.array([axes])
        axes = axes.reshape((nrow,-1))
        if title is not None: fig.suptitle(title)
    else:
        nrow = len(fig.axes)//ncol
        axes = np.array(fig.axes).reshape((nrow,-1))
    
    axes = axes[row:].flatten()

    for k, (ax, trace, band) in enumerate( zip( axes, stream, bands ) ):
    
        fs,Zs,n,d = get_fft_trace(trace)
        if remove_sensitivity: Zs /= trace.stats.response.instrument_sensitivity.value
            
        label = make_label(trace.stats.channel,label_id,extra_label)
        if k%ncol == 0: ylabel = True
        else:           ylabel = False
        
        xlabel = False
        if logy_xlabel:
            if k//ncol+1 == c: xlabel=True
        
        plot_spectrum(fs, Zs, ax, ylabel=ylabel, xlabel=xlabel, label=label, band=band, linewidth=linewidth, color=color, alpha=alpha, xscale=xscale)
                   
    if logy_trace: stream = stream[0]


    return fig
######################################################################################



######################################################################################
def plot_spectrum(fs, Rs, axes, ylabel=True, xlabel=True, label=None, band=None, linewidth=1, color=None, alpha=1, hline=None, xscale="log"):
    
    logy = type(axes) == np.ndarray or type(axes) == list
    if logy:
        ax = axes[0]
        axphase = axes[1]
        if ylabel:
            ax.set_ylabel("Amplitude")
            axphase.set_ylabel("Phase [deg]")
        if xlabel:
            axphase.set_xlabel("Frequency [Hz]")
    else:
        ax = axes
        if ylabel:
            ax.set_ylabel("Amplitude")
        if xlabel:
            ax.set_xlabel("Frequency [Hz]")
        

    amplitude = abs(Rs)
    ax.plot(fs[1:],amplitude[1:],label=label, linewidth=linewidth, color=color, alpha=alpha)
    ax.set_yscale("log")
    if xscale is not None: ax.set_xscale(xscale)

    if logy:
        angles = np.angle(Rs,deg=True)
        axphase.plot(fs[1:], angles[1:], label=label, linewidth=linewidth, color=color, alpha=alpha)
        axphase.set_ylim(-190,190)
        axphase.set_yscale("linear")
        axphase.set_yticks([-180,-90,0,90,180])
        for val in [-180,0,180]:
            axphase.axhline(val,color="gray",linewidth=0.5)
            
    if band is not None:
        if len(band) > 1:
            ax.axvspan(*band, linewidth=0.5, color="red", alpha=0.2)
            if logy: axphase.axvspan(*band, linewidth=0.5, color="red", alpha=0.2)
        else:
            ax.axvline(band, linewidth=0.5, color="red")
            if logy: axphase.axvline(band, linewidth=0.5, color="red")
    
    if hline: ax.axhline(hline,color="red",linewidth=0.5)
    
    if label is not None: 
        ax.legend()
        if logy: axphase.legend()
######################################################################################



######################################################################################
def plot_response(stream, output="DEF", end=False, fig=None, title=None, extra_label=None, bands=None, sharey=False, xscale="log", linewidth=1, color=None, alpha=1):

    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)

    logy = fig is None
    

    channels, indices = get_channels(stream, return_indices=True)

    if bands is None: bands = np.array([None for _ in stream])
    newbands = bands[indices]

    if logy:
        height = 6 + 0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(2,len(channels),tight_layout=True,sharex=True,sharey=sharey,figsize=(10,height))
        if title is not None: fig.suptitle(title)
        #for ax in axes.flatten():
        #    ax.set_xscale(xscale)
        axes = axes.reshape((2,-1))
        for ax in axes[-1]:
            ax.set_xlabel("Frequency [hz]")
    else:
        axes = np.resphape(fig.axes,(2,-1))
    
    for k, (axe, channel, band) in enumerate( zip( axes.T, channels, newbands ) ):
        
        trace = stream.select(channel=channel)[0]
    
        fs,Rs,IRs,sensitivity = get_fft_response(trace,output=output,end=end)
        
        label = make_label(channel, logy, extra_label)
        if k==0: ylabel = True
        else:    ylabel = False
        
        plot_spectrum(fs, Rs, axes=axe, ylabel=ylabel, label=label, band=band,
                      linewidth=linewidth, color=color, alpha=alpha, xscale=xscale)

    if logy_trace: stream = stream[0]
    return fig
######################################################################################



######################################################################################
def keep_seismic_channels(inventory):

    band_codes = ["F","G","D","C","E","H","B","M","L","V","U","R","P","T","Q","A","O"]
    inst_codes = ["N","L","H"]

    for network in inventory:
        for k,station in enumerate(network):
            codes = np.unique([ channel.code[:2]+"*" for channel in station ])
            channels = []
            for inst in inst_codes:
                bands = np.unique([ code[0] for code in codes if code[1] == inst ])
                channel = None
                for selected_band in band_codes:
                    if selected_band in bands:
                        channel = [band+inst+"*" for band in bands if band != selected_band]
                        break
                if channel is not None:
                    channels += channel
            for channel in channels:
                inventory = inventory.remove(network.code,station.code,"*",channel)
                
    return inventory
######################################################################################










###########################################################s
def plot_data(data,origin=None,modelled_times=None,fig=None):
    
    if origin is None:
        origin = np.zeros(4)
    
    TE,XE,YE,ZE = origin
    
    Ts,Xs,Ys,Zs,Es,Ps = data.T
    
    Ls = np.sqrt((Xs-XE)**2+(Ys-YE)**2)
    Ds = np.sqrt(Ls**2+(Zs-ZE)**2)
    phases = data[-1]
    if modelled_times is not None:
        Rs = Ts - modelled_times
    else:
        Rs = np.zeros(len(Ts))
                
    if not fig:
        fig,axes = plt.subplots(1,2,tight_layout=True,figsize=(12,4),sharey=True)
    else:
        axes = fig.axes
        
    for phase in range(2):
        whe = phases == phase
        if phase == 1: label = "S data"
        elif phase == 0: label = "P data"
        axes[0].errorbar(Ts[whe],Ds[whe],xerr=Es[whe],fmt=".",linewidth=0.5,label=label)
        axes[1].errorbar(Rs[whe],Ds[whe],xerr=Es[whe],fmt=".",linewidth=0.5)
    if modelled_times is not None:
        axes[0].scatter(modelled_times,Ds,marker="*",color="red",label="model")
    axes[0].legend()
    axes[0].set_ylabel("Hypocentral distance [km]")    
    axes[0].set_xlabel("Times [s]")    
    axes[1].set_xlabel("Residues [s]")    
    
    D_max = 1.05*Ds.max()
    T_max = 1.05*Ts.max()
    R_max = 1.05*( abs(Rs).max() + abs(Es).max() )

    for k,ax in enumerate(axes):
        ax.set_ylim(0,D_max)
        if k == 0:
            ax.axvline(T0,linewidth=0.5,color="black")
            ax.text(T0,0.9*D_max,"Origin Time",rotation="vertical",horizontalalignment="center",verticalalignment="top",backgroundcolor="white")
        else:
            ax.set_xlim(-R_max,R_max)
            ax.axvline(0,linewidth=0.5,color="black")
        
    return fig
###########################################################s



###########################################################s
def plot_conditional_probability(keys,qb,qb_std,functional,nsigma=4,qb_std_test=None):

    M1 = len(qb)
    nr = int(np.ceil(M1/2))
    fig,axes = plt.subplots(nr,2,figsize=(12,8),tight_layout=True)
    axes = axes.flatten()

    for k,key in enumerate(keys):
        ax = axes[k]
        ax.set_xlabel(key)
        vals = np.linspace(-nsigma,nsigma,1000)*qb_std[k] + qb[k]
        if k == 3 and vals[-1] >= 0: 
            vals = vals.clip(-np.inf,0)
            ax.axvline(0,color="black",linewidth=0.5)

        Ls = np.empty(len(vals))
        q = qb.copy()
        for i,val in enumerate(vals):
            q[k] = val
            Ls[i] = functional(q)

        Ls -= Ls.min()
        prob = np.exp(-Ls)
        norma = ( (prob[1:]+prob[:-1])*np.diff(vals)/2 ).sum()
        prob /= norma
        ax.plot(vals,prob,label="conditional\nprobability")
        
        gaus = stats.norm(qb[k],qb_std[k])
        ax.plot(vals,gaus.pdf(vals),label="gaussian")

        if qb_std_test is not None:
            gaus = stats.norm(qb[k],qb_std_test[k])
            ax.plot(vals,gaus.pdf(vals),label="gaussian 2",color="red",linestyle="dashed")

        ax.axhline(0,color="black",linewidth=0.5)
        ax.axvline(qb[k],color="red")
        ax.axvspan(qb[k]-qb_std[k],qb[k]+qb_std[k],alpha=0.2,color="red")
        ax.legend()

    for ax in axes[M1:]:
        ax.set_visible(False)

    return fig
###########################################################s


