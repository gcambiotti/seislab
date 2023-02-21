import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn




#########################################################################
J = lambda x: spherical_jn(0,x)
#########################################################################

#########################################################################
def eva_box(ts, Tb=1):
    """
    \t\n
    Return the  the box function of length `Tb` evaluated at times `ts`
    """

    box = np.ones(ts.shape)
    box[ts<0]  = 0
    box[ts>=Tb] = 0
    
    return box
#########################################################################


#########################################################################
def eva_FT_box(fs, Tb=1):
    """
    \t\n
    Return the Fourier transform of the box function of length `Tb` evaluated at frequencies `fs`
    """
    
    ws = 2*np.pi*fs
    xs = ws*Tb/2
    FT_box = Tb * np.exp(-1j*xs) * J(xs)
    
    return FT_box
#########################################################################


#########################################################################
def eva_test_ground_motion(ts,Tg=1):
    """
    \t\n
    Return the test ground motion (displacement, velocity and accelarations) defined in eq. (3.59) evaluated at times `ts`
    """
    wg = 2*np.pi/Tg
    test = np.empty((3,)+ts.shape)
    
    x = wg*ts
    x2 =2*x
    test[0] = (   np.cos(x2)/4 - np.cos(x) + 3/4 ) / wg**2
    test[1] = ( - np.sin(x2)/2 + np.sin(x)       ) / wg
    test[2] = ( - np.cos(x2)   + np.cos(x)       )
    
    test[:,ts<0] = 0
    test[:,ts>Tg] = 0
    
    return test
#########################################################################



#########################################################################
def eva_test_ground_displacement(ts, Tg=1):
    """
    \t\n
    Return the test ground displacement defined in eq. (3.59) evaluated at times `ts`
    """

    test = eva_test_ground_motion(ts, Tg=Tg)[0]
    
    return test
#########################################################################


#########################################################################
def eva_FT_test_ground_displacement(fs, Tg=1):
    """
    \t\n
    Return the Fourier transform of the test ground displacement defined in eq. (5.15) evaluated at frequencies `fs`
    """

    wg = 2*np.pi / Tg
    ws = 2*np.pi*fs
    
    x = ws*Tg/2
    xg = wg*Tg/2
    xg2 = 2*xg
    
    FT_test = np.empty(fs.shape,dtype=complex)
    
    FT_test  = J(xg2-x) * np.exp(1j*(xg2-x) ) / 8    + J(xg2+x) * np.exp(-1j*(xg2+x)) / 8
    FT_test -= J(xg-x)  * np.exp(1j*(xg-x))   / 2    + J(xg+x)  * np.exp(-1j*(xg+x))  / 2
    FT_test += J(-x)    * np.exp(-1j*x)       * 3/4
    
    FT_test *= Tg / wg**2
    
    return FT_test
#########################################################################







#########################################################################
def DFT_to_periodic(cts, fs, DFT, n, d):
    """
    \t\n
    Return the approximating periodic function defined in eq. (5.31) evaluated 
    at times `cts`. It requires the discrete frequencies `fs`, the DFT 
    coefficients `DFT`, the number of samplings `n` and the time step `d`.
    """
    o = n%2
    m = n//2
    ws = 2*np.pi*fs
    T = n*d
    
    DFT_mod = 2*DFT.copy()
    DFT_mod[0] /= 2
    if o == 0: DFT_mod[-1] /= 2 
    fun = (np.exp(np.tensordot(cts,1j*ws,axes=0)) @ DFT_mod ) .real / T
    
    return fun
#########################################################################


#########################################################################
def DFT_to_FT(cfs, fs, DFT, n, d):
    """
    \t\n
    Return the Fourier transform of the approximating function with compact 
    support defined in eq. (5.33) evaluated at frequencies `cfs`. It requires
    the discrete frequencies `fs`, the DFT coefficients `DFT`, the number of 
    samplings `n` and the time step `d`.
    """

    o = n%2
    m = n//2
    ws = 2*np.pi*fs
    cws = 2*np.pi*cfs
    T = n*d
    
    xs = ws * T/2
    cxs = cws * T/2
    
    xs2,cxs2 = np.meshgrid(xs,cxs)
    xk_p = xs2-cxs2
    xk_m = -xs2-cxs2
    
    ck = np.ones(m+1)
    ck[0] /= 2
    if o == 0: ck[-1] /=2
    
    DFT_mod = DFT * ck
    FT = ( J(xk_p) * np.exp(1j*xk_p) ) @ DFT_mod +  ( J(xk_m) * np.exp(1j*xk_m) ) @ DFT_mod.conjugate()

    return FT
#########################################################################


#########################################################################
def plot_comparison(T, d, eva_fun, eva_FT_fun=None, args=(), tmin=-1, tmax=21, fmin=0, fmax=6, marker=True):

    cts = np.linspace(tmin,tmax,100001)
    cfun = eva_fun(cts,*args)

    ts = np.arange(0,T,d)
    n = len(ts)
    fun = eva_fun(ts,*args)

    fs = np.fft.rfftfreq(n,d)
    DFT = np.fft.rfft(fun) * d
    FT = eva_FT_fun(fs,*args)

    cfun_periodic = DFT_to_periodic(cts,fs,DFT,n,d)

    
    if marker: scatter_options = dict(marker=".", color="red", zorder=30)
    hvline_options = dict(color="black", linewidth=0.5)
    
    
    fig1, axes = plt.subplots(2,tight_layout=True,figsize=(10,4))
    fig1.suptitle("Comparison in the time domain")

    ax = axes[0]
    ax.plot(cts,cfun,label="original")
    line = ax.plot(cts,cfun_periodic,label="approximation")[0]
    if marker is not None: ax.scatter(ts,fun,**scatter_options)

    ax = axes[1]
    whe = np.logical_and(cts>=0,cts<=T)
    ax.plot(cts[whe],cfun[whe]-cfun_periodic[whe],label="difference")
    if marker:  ax.scatter(ts,np.zeros(n),**scatter_options)

    for ax in axes:
        ax.legend()
        ax.axhline(0,**hvline_options)
        ax.axvspan(0,T,color="gray",alpha=0.2)
    axes[-1].set_xlabel("Time [s]")

    if eva_FT_fun is None:
        return fig1,None

    cfs = np.linspace(fmin,fmax,10001)
    cFT = eva_FT_fun(cfs,*args)
    
    cFT_periodic = DFT_to_FT(cfs,fs,DFT,n,d)

    fig2, axes = plt.subplots(3,tight_layout=True,figsize=(10,6),sharex=True)
    fig2.suptitle("Comparison in the frequency domain")

    ax = axes[0]
    ax.set_ylabel("Real part")
    ax.plot(cfs,cFT.real,label="original")
    ax.plot(cfs,cFT_periodic.real,label="approximation")
    if marker: ax.scatter(fs, DFT.real, **scatter_options)

    ax = axes[1]
    ax.set_ylabel("Imaginary part")
    ax.plot(cfs,cFT.imag,label="original")
    ax.plot(cfs,cFT_periodic.imag,label="approximation")
    if marker: ax.scatter(fs, DFT.imag, **scatter_options)

    ax = axes[2]
    ax.plot(cfs,abs(cFT-cFT_periodic),label="difference")

    for ax in axes:
        ax.axhline(0,**hvline_options)
        ax.set_xlim(fmin,fmax)

    for ax in axes:
        ax.legend()
    axes[-1].set_xlabel("Frequency [Hz]")
        
    return fig1,fig2
#########################################################################









#########################################################################
def damped_harmonic_oscillator(ts, U, V, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the `transient_motions` (displacement, velocity, and acceleration)
    of a damped harmonic oscillator evaluated  at the times `ts` with  
    displacement `U` and velocity `V` as initial conditions. 

    The natural period and damping ratio can be set as optional arguments.
    """
    
    w0  = 2*np.pi/natural_period
    wd  = w0 * np.sqrt(1-damping_ratio**2)
    lam = w0*damping_ratio
    
    A = - U*w0**2 - 2*V*lam
    
    transient_motions = np.empty((3,)+ts.shape)
    
    transient_motions[0] = ( U *np.cos(wd*ts) + (U*lam + V)/wd       * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    transient_motions[1] = ( V *np.cos(wd*ts) - (U*w0**2 + V*lam)/wd * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    transient_motions[2] = ( A *np.cos(wd*ts) - (V*w0**2 + A*lam)/wd * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    
    return transient_motions
#########################################################################
    
    
#########################################################################    
def setup_harmonic_oscillator(wg, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the amplitude factor and the phase shift of a damped harmonic 
    oscillator for the given anuglar frequency wg. By default, the natural 
    period and dampimng ratio of the harmonic oscillator are 1 s and 0.1.
    """
    
    w0  = 2*np.pi / natural_period
    lam = w0 * damping_ratio
    
    D = np.sqrt((w0**2-wg**2)**2+(2*wg*lam)**2)
    C = (w0**2-wg**2)/D

    phase = - np.arccos(C)
    amplitude_factor = wg**2 / D
    
    return amplitude_factor, phase
#########################################################################


#########################################################################
def steadystate_driven_oscillator(ts, Tg=1, ground_phase=0, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return ground acceleration of given ground period and phase and the
    steady state relative motion (displacement, velocity and acceleration) 
    of an harmonic oscillator evaluated at the times ts.
    """
    
    wg = 2*np.pi / Tg
    amplitude_factor, phase = setup_harmonic_oscillator(wg, natural_period, damping_ratio)
    
    x = wg * ts + ground_phase 
    ground_acceleration = np.sin(x)
    
    x += phase
    steadystate_motions = np.empty((3,)+ts.shape)
    steadystate_motions[0] = - amplitude_factor / wg**2 * np.sin(x)  # relative displacement
    steadystate_motions[1] = - amplitude_factor / wg    * np.cos(x)  # relative velocity
    steadystate_motions[2] =   amplitude_factor         * np.sin(x)  # relative acceleration
    
    return ground_acceleration, steadystate_motions
#########################################################################


#########################################################################
def sinusoidal_driven_oscillator(ts, Tg=1, ground_phase=0, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return sinusoiudal ground acceleration of given ground period and phase, 
    the steady state relative motion and the transient response (displacement, velocity and acceleration)of 
    an harmonic oscillator evaluated at the times ts.
    """

    # setup 
    wg = 2*np.pi / Tg
    amplitude_factor, phase = setup_harmonic_oscillator(wg, natural_period, damping_ratio)
    
    # initial conditions
    U = amplitude_factor/wg**2 * np.sin(ground_phase+phase)
    V = amplitude_factor/wg    * np.cos(ground_phase+phase)
    
    # the homogeneous (or transient) solution
    transient_motions = damped_harmonic_oscillator(ts, U, V, natural_period, damping_ratio)
    transient_motions[:,ts<0] = 0
    
    # the particular solution
    ground_acceleration, steadystate_motions = steadystate_driven_oscillator(ts, Tg, ground_phase, natural_period, damping_ratio)
    ground_acceleration[ts<0] = 0
    steadystate_motions[:,ts<0] = 0

    # implementing eq. ()
    relative_motions = steadystate_motions + transient_motions
    
    return ground_acceleration, relative_motions, transient_motions
#########################################################################


#########################################################################
def test_driven_oscillator(ts, Tg=1, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the relative motion and the transient response (displacement, velocity and acceleration) of 
    an harmonic oscillator evaluated at the times ts driven by the test ground motion.
    """
    
    # setup 
    wg  = 2*np.pi/Tg
    wg2 = 2*wg
    
    A,  beta  = setup_harmonic_oscillator(wg,  natural_period, damping_ratio)
    A2, beta2 = setup_harmonic_oscillator(wg2, natural_period, damping_ratio)

    # initial conditions
    U  =   A  / wg**2  * np.cos(beta)
    V  = - A  / wg     * np.sin(beta)
    U -=   A2 / wg2**2 * np.cos(beta2)
    V -= - A2 / wg2    * np.sin(beta2)
    
    # steadystate motions
    x  = wg  * ts + beta
    x2 = wg2 * ts + beta2

    steadystate_motions = np.empty((3,)+ts.shape)
    steadystate_motions[0] = ( - A*np.cos(x) + A2*np.cos(x2)/4 ) / wg**2
    steadystate_motions[1] = (   A*np.sin(x) - A2*np.sin(x2)/2 ) / wg
    steadystate_motions[2] =     A*np.cos(x) - A2*np.cos(x2)
    
    steadystate_motions[:,ts<0]  = 0
    steadystate_motions[:,ts>=Tg]  = 0

    # transient motions
    transient_motions  = damped_harmonic_oscillator(ts, U, V, natural_period, damping_ratio)
    transient_motions2 = damped_harmonic_oscillator(ts-Tg, U, V, natural_period, damping_ratio)
    transient_motions[:,ts<0] = 0
    transient_motions2[:,ts<Tg] = 0
    transient_motions -= transient_motions2
    
    # relative motion
    relative_motions = steadystate_motions + transient_motions

    return relative_motions, transient_motions
#########################################################################


#########################################################################
def eva_FT_green(f, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the green function in the frequency domain evaluated 
    at the frequency (or frequencies) `f` for the harmonic oscillator 
    with given natural period and damping ratio
    """
    
    w0 = 2*np.pi/natural_period
    w = 2*np.pi*f
    FT_green = -1/(w0**2-w**2+2j*w*w0*damping_ratio)
    
    return FT_green
#########################################################################


#########################################################################
def eva_FT_response(f, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the response function in the frequency domain evaluated 
    at the frequency (or frequencies) `f` for the harmonic oscillator 
    with given natural period and damping ratio
    """
    
    w0 = 2*np.pi/natural_period
    w = 2*np.pi*f
    FT_response = w**2/(w0**2-w**2+2j*w*w0*damping_ratio)
    
    return FT_response
#########################################################################

