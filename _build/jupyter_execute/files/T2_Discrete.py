#!/usr/bin/env python
# coding: utf-8

# (chap-D)=
# # Discrete Fourier transform
# 
# 
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 
# ## Finite time interval and sampling rate
# 
# ```{div} full-width
# 
# In practise, when we observe a time-dependent signal $f(t)$, we can only get its value over a finite time interval and sampling rate. Within the assumpution that we sample the signal over a regular grid $[0,T)$ with time step $\delta t$, we can write
# 
# $$
# t_j = j\,\delta t \qquad j=0,\cdots,n-1 
# $$ (L2:1)
# 
# and so we observe the function from $0$ up to $T-\delta t$ with 
# 
# $$
# T=n\,\delta T
# $$ (L2:2)
# 
# ```
# 
# ### Finite time interval
# 
# ```{div} full-width
# 
# Let us forget for a while the problem issue concerning the finiteness of the sampling rate and focus only on the continuous signal defined over $[0,T)$ and consider the following signal with compact support over the same interval
# 
# $$
# f_c(t) = f(t)\,B_T(t)
# $$ (L2:3)
# 
# This means that we are neglecting the signal outise the sampling time interval $[0,T)$ assuming that it is zero.
# 
# Also, let us consider the periodic function which repeat this signal over all the time interavls $\big[k\,T,(k+1)\,T\big)$
# 
# $$
# f_p(t+k\,T) = f(t)  \qquad \forall t\in[0,T)\quad \forall k\in\mathbb{N}
# $$ (L2:4)
# 
# By definition, we have
# 
# $$
# f(t) = f_p(t) = f_c(t)\qquad \forall \,t\in[0,T)
# $$ (L2:5)
# 
# The periodi function admit the Fourier series representation
# 
# $$
# f_p(t) = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,\omega_k\,t} \qquad \omega_k = k\,\delta\omega \qquad \delta\omega = \frac{2\,\pi}{T}
# $$ (L2:6)
# 
# where the Fourier coefficients $F_k$ can be computed as
# 
# $$
# F_k = \int_0^T f_p(t)\,e^{-i\,\omega_k\,t}\,dt 
# $$ (L2:7)
# 
# We note that the integral in eq. {eq}`L2:7` can be seen as the Fourier transform (evaluated at the discretized frequency $\omega_k$) of signal with compact support 
# 
# $$
# \tilde{f}_c(\omega_k) = \int_0^T f(t)\,e^{-i\,\omega_k\,t}\,dt = \int_0^T f_p(t)\,e^{-i\,\omega_k\,t}\,dt = F_k
# $$ (L2:8)
# 
# Also, we note that the Fourier transform of the signal with compact support takes the following form
# 
# $$
# \begin{align}
# \tilde{f}_c(\omega) &= \int_0^T f_p(t)\, e^{-i\,\omega_\,t}\,dt = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,\int_0^T e^{i\,(\omega_k-\omega)\,t}\,dt =  \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,\frac{e^{i\,(\omega_k-\omega)\,T}-1}{i\,(\omega_k-\omega)} 
# \end{align}
# $$ (L2:9)
# 
# ```
# 
# ```{admonition} Singularities at the discrete frequencies?
# :class: note, full-width
# 
# Despite the denomitator of the addends in eq. {eq}`L2:9`, $\tilde{f}_c(\omega)$ is not singular at the discrete frequencies $\omega_k$. This can be proved by rewriting eq. {eq}`L2:9` in terms of the spherical bessel function of order $0$, similar to what already done in the {numref}`Lecture %s <chap-F>` {ref}`chap-F`, eqs. {eq}`T1:10` and {eq}`T1:11`
# 
# $$
# \begin{align}
# \tilde{f}_c(\omega) &= \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,x_k(\omega)}\,J(x_k(\omega)) \qquad\mathrm{with}\qquad x_k(\omega) = \frac{\omega_k-\omega}{2}\,T
# \end{align}
# $$ (L2:10)
# 
# Then, by considering that
# 
# $$
# x_k(\omega_h) = \pi\,(k-h) \qquad\mathrm{and}\qquad J(\pi\,k) = \delta_{k0}
# $$ (L2:11)
# 
# it is straightforward to show that
# 
# $$
# \tilde{f}_c(\omega_k) = F_k
# $$ (L2:12)
# 
# ```
# 

# ```{admonition} Test ground displacement
# :class: note, full-width
# 
# Let us now considering the test ground displacement discussed in  {numref}`Lecture %s <chap-H>` {ref}`chap-H` and defined in eq. {eq}`H0:62` that we rewrite here for the convenience of the reader as follows
#  
# 
# $$
# \begin{align}
# & u_g(t) = f_p(t)\,B_{T_g}(t)
# \end{align}
# $$(L2:13b)
# 
# with $f_p$ being the following periodic function over the time interval $[0,T_g)$
# 
# $$
# \begin{align}
# & f_p(t) = \frac{A_g}{\omega_g^2}\,\big[\tfrac{1}{4}\,\cos(2\,\omega_g\,t)-\cos(\omega_g\,t) + \tfrac{3}{4}\big]  \qquad\mathrm{with}\qquad \omega_g = \frac{2\,\pi}{T_g}
# \end{align}
# $$ (L2:14bis)
# 
# We note that the test ground displacement is alreay a function with compact support on the time interval $[0,T)$ with $T\geq T_g$. Let us now choose the sampling time interval with the periodic one, i.e., $T=T_g$, and represent eq. {eq}`L2:14bis` with its  Taylor series 
# 
# $$
# \begin{align}
# & f_p(t) = \frac{1}{T_g}\,\sum_{k=-2}^2\,F_k\,e^{i\,\omega_k\,t} \qquad\mathrm{with}\qquad \omega_k = k\,\omega_g 
# \end{align} 
# $$ (L2:15bis)
# 
# where the non-zero Fourier coefficients $F_k$ are given by
# 
# $$
# \begin{align}
# & F_0 = \frac{3}{4}\,\frac{A_g\,T_g}{\omega_g^2} \qquad F_{\pm 1} = \mp \frac{1}{2}\,\frac{A_g\,T_g}{\omega_g^2} \qquad F_{\pm 2} = \pm \frac{1}{8}\,\frac{A_g\,T_g}{\omega_g^2}
# \end{align}
# $$ (L2:16bis)
# 
# In light of this and making use of eg. {eq}`L2:10`, we thus obtain the Fourier transform of test ground displacement
# 
# 
# $$
# \begin{align}
# \tilde{u}_g(\omega) = \frac{1}{T_g}\,\sum_k F_k\,e^{i\,x_k(\omega)}\,J(x_k(\omega)) \qquad\mathrm{with}\qquad x_k(\omega) = \frac{k\,\omega_g-\omega}{2}\,T 
# \end{align}
# $$ (L2:17bis)
# 
# 
# ```

# ### Finite sampling rate
# 
# ```{div} full-width
# 
# Let us now use the Fourier series representation of the periodic function $f_p$ and the fact that 
# 
# $$
# \begin{align}
# f(t)=f_p(t)\qquad\forall t\in[0,T)
# \end{align}
# $$ (L2:18bis)
# 
# to obtain the signal evaluated at the discretized times $t_j$
# 
# $$
# \begin{align}
# f_j = f(t_j) = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,\omega_k\,t_j}
# \end{align}
# $$  (L2:13)
# 
# By noting that
# 
# $$
# \begin{align}
# e^{\omega_{k+p\,n}\,t_j} = e^{\omega_k\,t_j + 2\,\pi\,p} = e^{\omega_k\,t_j}  \qquad \forall\,k\in[0,\cdots,n-1]
# \end{align}
# $$ (L2:14)
# 
# and making the substitution $k\rightarrow k+p\,n$, we can recast eq. {eq}`L2:13` as follows 
# 
# 
# $$
# \begin{align}
# f_j = \frac{1}{\delta t}\,\frac{1}{n}\,\sum_{k=0}^{n-1}\sum_{p=-\infty}^\infty F_{k+p\,n}\,e^{i\,\omega_{k+p\,n}\,t_j} =\frac{1}{\delta t}\,\frac{1}{n}\,\sum_{k=0}^{n-1} \bar{F}_k\,e^{i\,\omega_k\,t_j}
# \qquad
# \bar{F}_k = \sum_p F_{k+p\,n}
# \end{align}
# $$ (L2:15)
# 
# This result shows how the sampled signal can be seen as characterized by a finite number of discrete frequencies $[\omega_0,\cdots,\omega_{n-1}$ and that it can be built using a finite number of combinations of Fourier coefficients $[\bar{F}_0,\cdots,\bar{F}_{n-1}]$.
# 
# Also, it can be shown that these combinations can be obtained from the sampled signal as follows
# 
# $$
# \begin{align}
# \bar{F}_k = \delta t\,\sum_{j=0}^{n-1} f_j\,e^{-i\,\omega_k\,t_j}
# \end{align}
# $$  (L2:16)
# 
# ```
# 
# ```{admonition} Proof 
# :class: tip, full-width
# 
# Eq. {eq}`L2:16` can be proved by making use of the summation of geometric series
# 
# $$
# \begin{align}
# \sum_{j=0}^{n-1} \alpha^j = \begin{cases}
# \frac{1-\alpha^n}{1-\alpha} & \alpha\neq 1
# \\
# n & \alpha=1
# \end{cases}
# \end{align}
# $$  (L2:17)
# 
# ```
# 
# ```{div} full-width
# 
# Eq. {eq}`L2:16` is the discrete Fourier transform (DFT) of the sampled value, but for the factor $\delta t$, and it is implemented by the function `fft` of the `np.fft` module. Eq. {eq}`L2:15`, instead, is the inverse discrete Fourier transform (IDFT), but for the factor $1/\delta t$, and it is implemented by the function `ifft` of the same module. 
# 
# ```

# ### Real DFT
# 
# ```{div} full-width
# 
# For real-valued functions, we have
# 
# $$
# \begin{align}
# & F_{-k} = F_k^*
# \end{align}
# $$  (L2:18)
# 
# with the asterisc $*$ denoting the complex conjugate. In  light of this, the linear combinations $\bar{F}_k$ satisfy a similar identity
# 
# $$
# \begin{align}
# & \bar{F}_{n-k} = \sum_{p=-\infty}^\infty F_{n-k+n\,p} = \sum_{p=-\infty}^\infty F_{-k+n\,p} = \sum_{p=-\infty}^\infty F_{k-n\,p}^* 
# \\
# &= \bar{F}_k^*
# \end{align}
# $$ (L2:19)
# 
# In the case of real-valued functions, rather than using the DFT, it is convinient to use of the real DFT to take advantage of the avove identity. The real DFT, nevertheless, requires to distinguish between the cases in which the number of samplings is odd or even. In this perspective, let us assume that 
# 
# $$
# \begin{align}
# & n = 2\,m+q 
# \end{align} 
# $$ (L2:20)
# 
# with $q$ being $1$ or $0$ when $n$ is odd or even, respectively. Then, let us also write eq. {eq}`L2:15` by spltting the summation over the index $k$ into two parts and making the subsititution $k\rightarrow n-k$ in the second summation
# 
# $$
# \begin{align}
# f(t_j) &= \frac{1}{\delta t}\,\frac{1}{n}\left(\sum_{k=0}^{m} \bar{F}_k\,e^{i\,\omega_k\,t_j} + \sum_{k=m+1}^{n-1} \bar{F}_k\,e^{i\,\omega_k\,t_j}\right) = \frac{1}{T}\,\left( \sum_{k=0}^{m} \bar{F}_k\,e^{i\,\omega_k\,t_j} + \sum_{k=1}^{m-1+q} \bar{F}_{n-k}\,e^{-i\,\omega_{k}\,t_j} \right) &
# \\
# &=  \frac{1}{T}\,\left(\sum_{k=0}^{m} \bar{F}_k\,e^{i\,\omega_k\,t_j} + \sum_{k=1}^{m-1+q} \bar{F}_{k}^*\,e^{-i\,\omega_{k}\,t_j}\right) 
# \end{align}
# $$  (L2:21)
# 
# ```
# 
# ```{admonition} Odd and even cases
# :class: note, full-width
# 
# In particular, when $n$ is odd, eq. {eq}`L2:21` becomes
# 
# $$
# \begin{align}
# f(t_j) &= \frac{1}{T}\,\left(\bar{F}_0 + \sum_{k=0}^{m} \left[\bar{F}_k\,e^{i\,\omega_k\,t_j}+\bar{F}_k^*\,e^{-i\,\omega_k\,t_j}\right] \right)
# \\
# &= \frac{1}{T}\,\left(\bar{F}_0 +2\, \sum_{k=0}^{m} \big[\Re[\bar{F}_k]\,\cos(\omega_k\,t_j)-\Im[\bar{F}_k]\,\sin(\omega_k\,t_j)\big] \right)
# \end{align}
# $$  (L2:22)
# 
# 
# while, when $n$ is even, it becomes
# 
# $$
# \begin{align}
# f(t_j) &= \frac{1}{T}\,\left(\bar{F}_0 + \sum_{k=0}^{m-1} \left[\bar{F}_k\,e^{i\,\omega_k\,t_j}+\bar{F}_k^*\,e^{-i\,\omega_k\,t_j}\right] + \bar{F}_{m}\,e^{i\,\omega_m\,t_j}\right)
# \\
# &= \frac{1}{T}\,\left(\bar{F}_0 +2\, \sum_{k=0}^{m-1} \big[\Re[\bar{F}_k]\,\cos(\omega_k\,t_j)-\Im[\bar{F}_k]\,\sin(\omega_k\,t_j)\big] + \bar{F}_{m}\,\cos(\omega_m\,t_j)\right)
# \end{align}
# $$  (L2:23)
# 
# We note that, when $n$ is even, $\bar{F}_0$ and  $\bar{F}_m$ are real, and $\cos(\omega_m\,t_j) = (-1)^j$. Different, when $n$ is odd, only $\bar{F}_0$ is real.
# 
# ```
# 
# ```{div} full-width
# 
# It is also usefull to rewrite eq.  {eq}`L2:21` as follows
# 
# $$
# \begin{align}
# f_j &= \frac{1}{T} \sum_{k=0}^{m} c_{k}\,\left[\bar{F}_k\,e^{i\,\omega_k\,t_j}+\bar{F}_k^*\,e^{-i\,\omega_k\,t_j}\right] = \frac{2}{T}\,\Re\left[\sum_{k=0}^{m} c_k\,F_k\,e^{i\,\omega_k\,t_j}\right]
# \end{align}
# $$  (L2:24)
# 
# 
# where the factors $c_k$ take into account the distinction between the odd and evan cases and are defined as follows
# 
# $$
# c_0 = \tfrac{1}{2} \qquad c_{k} =  1 \qquad c_m = \tfrac{1+q}{2}
# $$  (L2:25)
# 
# In light of eq. {eq}`L2:24`, real-valued signals require only the first $m+1$ coefficients $[\bar{F}_0,\cdots,\bar{F}_{m}]$ and are characterized by the $2\,m+1$ discrete frequencies $[-\omega_{m},\cdots,0,\cdots,\omega_m]$ . The coefficients and the non-negative discrete frequencies $[\bar{F}_0,\cdots,\bar{F}_{m}]$ and $[0,\cdots,\omega_m]$ are returned by the functions `rfft` and `rfftfreq` of the `np.fft` module. The inverse real DFT, instead, is implemented by the function `irfft` of the same module. The latter, by default, assume that $n$ is even. If this is not the case, it must be specified the odd number of samplings $n$.
# 
# ```
# 

# In[1]:


import numpy as np
help(np.fft.rfft)


# In[2]:


help(np.fft.rfftfreq)


# In[3]:


help(np.fft.irfft)


# ```{admonition} Approximating function of the original signal
# :class: note, full-width
# 
# In order to clarify the physical meaning of the above discussion, let us generalize eq. {eq}`L2:24` to be evaluated at any times. We thus define the approximating periodic function
# 
# $$
# g_p(t) = \frac{1}{T} \sum_{k=0}^{m} c_{k}\,\left[\bar{F}_k\,e^{i\,\omega_k\,t}+\bar{F}_k^*\,e^{-i\,\omega_k\,t}\right] 
# $$  (L2:26)
# 
# This function coincides with the original signal when evaluated at the discrete times $t_j$ and keeps real values also in between the discrete times. In this respect, it can be seen as an approximation of the original signal within the sampling time interval $[0,T)$. 
# 
# By defining also the associated function with compact support on the sampling time interval
# 
# $$
# g_c(t) = g_p(t)\,B_T(t)
# $$  (L2:27)
# 
# we can use it as an approximation of the original function everywhere, even outside the sampling time interval. Moreover, its Fourier transform yields
# 
# $$
# \begin{align}
# \tilde{g}_c(\omega) = \int_0^T\,g_p(t)\,e^{-i\,\omega\,t}\,d t = \frac{1}{T}\,\sum_{k=0}^{m} c_{k}\,\left[\bar{F}_k\,\frac{e^{i\,(\omega_k-\omega)\,T}-1}{i\,(\omega_k-\omega)} +\bar{F}_k^*\,\frac{e^{-i\,(\omega_k+\omega)\,T}-1}{-i\,(\omega_k+\omega)}\right] 
# \\
# =\sum_{k=0}^{m} c_{k}\,\left[\bar{F}_k\,e^{i\,x_k(\omega)}\,J(x_k(\omega)) + \bar{F}_k^*\,e^{i\,x_{-k}(\omega)}\,J(x_{-k}(\omega)\right]
# \end{align}
# $$  (L2:28)
# 
# with $x_k$ given by eq. {eq}`L2:10`.
# 
# 
# ```

# ## Preliminary considerations
# 
# ```{div} full-width
# 
# We have just seen as it is possible to approximate the Fourier transform of a time-dependent real-valued function using the real discrete Fourier transform. The latter, nevertheless, it is based on the Fourier series representation and it can be applied only to periodic functions. This means that when we choose the finite time interval $[0,T)$ where we sample the time-dependent function, we are also choosing the period $T$ of the periodic function that we will actually consider. 
# 
# ```
# 
# ### Box function
# 
# ```{div} full-width
# 
# In order to make practise on the real DFT, let us import the main python libraries and the local seislab module
# 
# ```

# In[4]:


get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from theory import plot_comparison


# ```{div} full-width
# 
# As first example, let us implement the box function and its Fourier transform as already done in {ref}`chap-F`
# 
# ```

# In[7]:


from theory import eva_box, eva_FT_box


# ```{div} full-width
# 
# Then, we make use of the functions implementing the real DFT (`np.fft.rfft`) and its inverse (`np.fft.irfft`), and of the function returning the discretized frequency (`np.fft.rfftfreq`) for the box function $B_{T_b}$, with $T_b=3$ s, sampled over time interval [0,5) s with time step 0.5 s. We also make two figures showing the comparison in the time and frequency domain using the function `plot_comparison` of the `seislab` module.
# 
# ```

# In[8]:


Tb = 3

# Discretized times
T, d = 5, 0.5
ts = np.arange(0,T,d)
n = len(ts)

# Samplings of the box function 
box = eva_box(ts,Tb=Tb)

# Discretized frequencies and the DFT
fs = np.fft.rfftfreq(n,d)
DFT = np.fft.rfft(box) * d
FT = eva_FT_box(fs,Tb=Tb)

# Inverse DFT
box_check = np.fft.irfft(DFT,n) / d

# Print results
print("Samplings:")
print('Index   Time    Sample   Sample from IDFT')
for k,(t, val, val_check) in enumerate(zip(ts, box, box_check)):
    print('{0:<4d}   {1:6.3f}   {2:6.3f}   {3:6.3f}'.format(k, t, val, val_check))

print("\nDFT:")
print('Index   Freq     DFT               FT')
for k,(f, val, val_ft) in enumerate(zip(fs, DFT, FT)):
    print('{0:<4d}   {1:6.3f}   {2.real:6.3f} + {2.imag:6.3f}j  {3.real:6.3f} + {3.imag:6.3f}j'.format(k, f, val, val_ft))
    
# Plotting
fig1,fig2 = plot_comparison(T, d, eva_box, eva_FT_box, args=(Tb,))


# ```{div} full-width
# 
# Here, as it concerns the comparison in the time domain, the blue and orange lines indicates the original signal and the approximating periodic function $g_p$, eq. (). Also, the gray area indicates the sampling time interval. In the bottom pannel is shown the difference among the two functions. As it concerns the comparison in the frequency domain, instead, the blue and orange lines indicates the real and imaginary parts (top and middle pannels) of the Fourier transform of the original signal and of the Fourier transform of the approximating function with compact support $g_c$, eq. (). The bottom pannel shown the difference $|\tilde{f}(\omega)-\tilde{g}_c(\omega)|$.
# 
# We note that the approximating periodic function coincides with the original function at the discretized times. On the other hand, the approximating periodic function is much smoother than the box function just because it is composed of sine and cosine terms with low frequencies and this makes the largest difference in the surrounding of the discontinuity of the box function. Also, we note that the comparison makes sense only in the sampling time interval [0,10) s. Outisde the sampling time interval, indeed, the original signal is zero while the periodic function repeats itself just with period 10 s. In the frequency domain, instead, we note that the approximation underestimate the spectral content at frequencies higher than the Nyquist frequency $\omega_m$.
# 
# ```

# ```{div} full-width
# 
# In order to take into account for higher frequencies, let us now decrease the time step to 0.1 s and make a similar plot
# ```

# In[9]:


# Discretized times
T, d = 5, 0.1
ts = np.arange(0,T,d)
n = len(ts)

# Samplings of the box function 
box = eva_box(ts, Tb=Tb)

# Discretized frequencies and the DFT
fs = np.fft.rfftfreq(n,d)
DFT = np.fft.rfft(box) * d
FT = eva_FT_box(fs, Tb=Tb)

# Inverse DFT
box_check = np.fft.irfft(DFT,n) / d

# Print results
print("Samplings:")
print('Index   Time    Sample   Sample from IDFT')
for k,(t, val, val_check) in enumerate(zip(ts, box, box_check)):
    print('{0:<4d}   {1:6.3f}   {2:6.3f}   {3:6.3f}'.format(k, t, val, val_check))

print("\nDFT:")
print('Index   Freq     DFT               FT')
for k,(f, val, val_ft) in enumerate(zip(fs, DFT, FT)):
    print('{0:<4d}   {1:6.3f}   {2.real:6.3f} + {2.imag:6.3f}j  {3.real:6.3f} + {3.imag:6.3f}j'.format(k, f, val, val_ft))
    
# Plotting
fig1,fig2 = plot_comparison(T, d, eva_box, eva_FT_box, args=(Tb,))


# ```{div} full-width
# 
# Decreasing the sampling time step we improve the approximation in the time domain but we do not decrease the sampling frequency step with which we evaluate the Fourier transform. In this perspective, we need to increase the sampling time interval as shown hereinafter where the sampling time interval is set to [0,10) s keeping the sampling time step to 0.1 s
# 
# ```

# In[10]:


T, d = 10, 0.1
fig = plot_comparison(T, d, eva_box, eva_FT_box, args=(Tb,))


# ```{div} full-width
# 
# Increasing the length of the sampling time interaval makes the periodic function similar to the original box function on a wider time interval. Nevertheless, the periodic function is still periodic, althogh it does on a wider time interval.
# 
# The approximating periodic function still coincides with the original function at the sampling times. Nevertheless, it oscillates between the discretized times. These oscillations are the Gibbs effect that presents when we try to approximated sharp discontiniuities as those of the box function. 
# 
# ```

# 
# ### Test ground displacement 
# 
# ```{div} full-width
# 
# Let us now perform the same analysis on the test ground displacement that we implenent here only for ground displacements (with default ground period of $T_g=3$ s)
# 
# ```

# In[11]:


from theory import eva_test_ground_motion, eva_test_ground_displacement, eva_FT_test_ground_displacement


# ```{div} full-width
# 
# and make the plot for the sampling interval [0,5) with time step 0.5 s
# 
# ```

# In[12]:


Tg = 3
T, d = 5, 0.5
fig = plot_comparison(T, d, eva_test_ground_displacement, eva_FT_test_ground_displacement, args=(Tg,))


# ```{div} full-width
# 
# Here, instead, the same plot but with the sampling time step decrased to 0.1 s
# 
# ```

# In[13]:


T, d = 5, 0.1
fig = plot_comparison(T, d, eva_test_ground_displacement, eva_FT_test_ground_displacement, args=(Tg,))


# ```{div} full-width
# 
# and extending further the sampling time interval to [0,10) s
# 
# ```

# In[14]:


T, d = 10, 0.1
fig = plot_comparison(T, d, eva_test_ground_displacement, eva_FT_test_ground_displacement, args=(Tg,))


# ```{div} full-width
# 
# We can note that in the case of the displacement ground motion, the periodic function, in addition to coincides with the original function at the discretized times, well approximates the function also in between the discretized times, with differences less than about 0.4 and 0.00001 for sampling time step of 1 and 0.2 s, respectively.
# 
# ```

# ## Convolution
# 
# ```{div} full-width
# 
# Having understood the main aspect of considering a finite sampling time interval and step, let us discuss an efficient numerical method for dealing with sampled signal both in the time and frequency domains. In this respect, we do not longer consider the approximating periodic function, $g_p$, and the approximating Fourier transform of the corresponding function with compact support, $g_c$. On the contrary, we will base only on the sampled values $f_j$ and the DFT coefficients $\bar{F}_k$. In order to sample finely the signal both in the time and frequency domain, we shall choose large sampling time intervals $[0,T)$ and smalle sampling time step $\delta t$
# 
# ```

# ```{tip} 
# :class: full-width
# 
# Let us consider the samplings $f_j$ of the time dependent signal $f(t)$ at the discrete times $t_j$ over the sampling time interval $[0,T)$ with time step $\delta t$. Because the approximating function is periodic, afterwards, it will be convenient to define the sampling $f_j$ also for $j<0$ and $j\geq n$
# 
# $$
# f_{j+p\,n} = f_j \qquad\forall\,j=0,\cdots,n-1\qquad\mathrm{and}\qquad\forall p\in\mathbb{N}
# $$ (L2:29)
# 
# so that, for instance, we have $f_{n+j} = f_{j}$ or $f_{-j} = f_{n-j}$.
# 
# ```

# ```{div} full-width
# 
# Let us consider two time dependent signals $f(t)$ and $g(t)$ and their time convolution
# 
# $$
# \begin{align}
# h(t) &= (f\star g)(t) = \int_{-\infty}^\infty f(\tau)\,g(t-\tau)\,d\tau 
# \end{align}
# $$ (L2:30)
# 
# and see how we can obtain the samplings $h_j$ at the discrete times $t_j$ over the sampkung time interval $[0,T)$ starting from the samplings $f_j$ and $g_j$.
# 
# 
# Let us first assume that the two signals have compact support over the time intervals $[0,T_f)$ and $[0,T_g)$ with $T_f<T_g$. In this case, eq. {eq}`L2:30` can be further specified as follows
#  
# $$
# \begin{align}
# h(t) &=  \int_{0}^{T_f} f(\tau)\,g(t-\tau)\,d\tau = \int_{t-T_f}^t f(t-\tau)\,g(\tau)\,d\tau  = B(t,T_g+T_f)\,\int_{\max(t-T_f,0)}^{\min(t,T_g)} f(t-\tau)\,g(\tau)\,d\tau
# \end{align}
# $$ (L2:31)
# 
# Here, we make use of (i) the compact support of $f$ in the first step, (ii) the subsitution $\tau\rightarrow t-\tau$ in the second step and (iii) the compact support of $g$ in the third and last step. Then, we recast eq. as follows
# 
# $$
# \begin{align}
# h(t) 
# &=\begin{cases}
# \int_{0}^{t} f(t-\tau)\,g(\tau)\,d\tau & 0\leq t < T_f \\
# \int_{t-T_f}^{t} f(t-\tau)\,g(\tau)\,d\tau & T_f \leq t < T_g \\
# \int_{t-T_f}^{T_g} f(t-\tau)\,g(\tau)\,d\tau & T_g \leq t \leq T_g+T_f  \\
# 0 & t \notin [0,T_g+T_f) \\
# \end{cases} = \begin{cases}
# \int_{0}^t f(\tau)\,g(t-\tau)\,d\tau & 0\leq t < T_f \\
# \int_{0}^{T_f} f(\tau)\,g(t-\tau)\,d\tau & T_f \leq t < T_g \\
# \int_{t-T_g}^{T_f} f(\tau)\,g(t-\tau)\,d\tau & T_g \leq t \leq T_g+T_f  \\
# 0 & t \notin [0,T_g+T_f) \\
# \end{cases}
# \end{align}
# $$ (L2:32)
# 
# where in the second and last step we make use of the substitutionn $\tau\rightarrow t-\tau$.
# 
# ```
# 

# ```{admonition} Convolution of two box functions
# :class: note, full-width
# 
# For the case of two Heaviside functions 
# 
# $$
# \begin{align}
# &f(t) = B(t,T_f)\qquad g(t) = B(t,T_g)
# \end{align}
# $$ (L2:33)
# 
# eq. {eq}`L2:32` yields
# 
# $$
# \begin{align}
# h(t) &= \begin{cases}
# t & 0\leq t < T_f \\
# T_f & T_f \leq t < T_g \\
# T_g+T_f-t & T_g \leq t \leq T_g+T_f  \\
# 0 & t \notin [0,T_g+T_f) \\
# \end{cases} = R(t,T_f) + R(t-T_g,T_f)
# \end{align}
# $$ (L2:34)
# 
# where $R$ is the ramp box function
# 
# $$
# \begin{align}
# & R(t,T) = t\,H(t) + (T-t)\,H(t,T)
# \end{align}
# $$ (L2:35)
# 
# ```
# 

# ### Fourier series of the convolution
# 
# ```{div} full-width
# 
# Eq. {eq}`L2:32` shows that the convolution has a compact support over the time interval $[0,T_g+T_f)$ and, so, when we have to choose the sampling time interval $[0,T)$ so that $T\geq T_f+T_g$. We note that, thanks to this choise, the time convolution can be expressed in terms of the associated periodic function with compact support on $[0,T)$ and written as follows
# 
# $$
# \begin{align}
# h(t) &= (f_c\star g_c)(t) = \int_0^T f_p(\tau)\,g_p(t-\tau)\,d\tau = \frac{1}{T^2}\,\sum_{k=-\infty}^\infty\sum_{p=-\infty}^\infty F_k\,G_p\,e^{i\,\omega_p\,t}\,\int_0^T e^{i\,(\omega_k-\omega_p)\,\tau}\,d\tau 
# \\
# &= \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,G_k\,e^{i\,\omega_k\,t}
# \end{align}
# $$ (L2:36)
# 
# This result shows that the Fourier coefficients $H_k$ of the convolution $h$ correspond to the product of those of the two signals $f$ and $g$
# 
# $$
# \begin{align}
# & H_k = F_k\,G_k
# \end{align}
# $$ (L2:37)
# 
# 
# Let us now consider the issue concerning the finiteness of the sampling rate. In this case, we can only get the linear combinations $\bar{F}_k$ and $\bar{G}_k$ from the (real) DFT, that we report also here
# 
# $$
# \begin{align}
# & \bar{F}_k = \sum_{p=-\infty}^\infty F_{k+n\,p} \qquad \bar{G}_k = \sum_{p=-\infty}^\infty G_{k+n\,p}
# \end{align}
# $$ (L2:38)
# 
# Nevertheless, we note that their product
# 
# $$
# \begin{align}
# & \bar{F}_k\,\bar{G_k} = \left(\sum_{p=-\infty}^\infty F_{k+n\,p}\right)\,\left(\sum_{q=-\infty}^\infty G_{k+n\,q}\right) 
# \end{align}
# $$ (L2:39)
# 
# differs by the linear combinations $\bar{H}_k$
# 
# $$
# \begin{align}
# & \bar{H}_k = \sum_{p=-\infty}^\infty H_{k+n\,p} = \sum_{p=-\infty}^\infty F_{k+n\,p}\,G_{k+n\,p}  \neq \bar{F}_k\,\bar{G}_k 
# \end{align}
# $$ (L2:40)
# 
# On the other hand, as far as the two signals $f$ and $g$ have been sampled with high sampling rates, the linear combinations approximates the Fourier coefficients, $\bar{F}_k\approx F_k$ and $\bar{G}_k\approx G_k$, and so
# 
# $$
# \begin{align}
# & \bar{H}_k \approx \bar{F}_k\,\bar{G}_k
# \end{align}
# $$ (L2:41)
# 
# 
# Also, the sampling $h_j$ can be approximated by taking the inverse (real) DFT
# 
# $$
# \begin{align}
# & h_j = \frac{1}{\delta t}\,\frac{1}{n} \sum_{k=0}^{n-1} \bar{H}_k\,e^{i\,\omega_k\,t_j} \approx \frac{1}{\delta t}\,\frac{1}{n} \sum_{k=0}^{n-1} \bar{F}_k\,\bar{G}_k\,e^{i\,\omega_k\,t_j}
# \end{align}
# $$ (L2:42)
# 
# ```
# 
# ### Circular convolution
# 
# ```{div} full-width
# 
# In the end, according to the approximation given by eq. {eq}`L2:42` and to the convention defined by eq. {eq}`L2:29`, we can subsitute the DFT for $\hat{F}_k$ and $\bar{G}_{k}$ and recast eq. {eq}`L2:42` as follows
# 
# $$
# \begin{align}
# h_j &\approx \frac{\delta t}{n} \sum_{k=0}^{n-1} \left(\sum_j f_j \,e^{-i\,\omega_k\,t_j}\right)\left(\sum_p g_p \,e^{-i\,\omega_k\,t_p}\right)\,e^{i\,\omega_k\,t_j}
# =  \frac{\delta t}{n}\sum_k  \sum_p \sum_q\,f_p\,g_q\,e^{i\,\omega_k\,(t_j-t_p-t_q)}
# \\ &=\frac{\delta t}{n} \sum_{p=0}^{n-1} \sum_{q=0}^{n-1}\,f_p\,g_q\,\sum_k \left(e^{2\,\pi\,i\,(j-p-q)/n}\right)^k = \delta t \sum_{p=0}^{n-1} f_p\,g_{j-p}
# \end{align}
# $$ (L2:43)
# 
# In the last step we make use of the following identity
# 
# $$
# \begin{align}
# & \sum_{k=0}^{n-1} \left(e^{2\,\pi\,i\,(j-p-q)/n}\right)^k = \begin{cases}
# n & \forall\,j-p-q = 0 \pmod{n}\\
# 0 
# \end{cases}
# \end{align}
# $$ (L2:44)
# 
# that can be proved starting from the expression for the summation of geometric series, eq. {eq}`L2:17`.
# 
# Eq. {eq}`L2:43` shows that $h_j$ can be approximated by taking into account the periodic beahviour of the samples $f_j$ and $g_j$, when they are considered as resulting from the associated periodic functions, as expressed by the concetion defined by eq. {eq}`L2:29`. For instance, let us specify eq. {eq}`L2:43` for $j=0$ and recast it as follows
# 
# 
# $$
# \begin{align}
# h_0 &=  \delta t \sum_{p=0}^{n-1} f_p\,g_{-p} =  \delta t \sum_{p=0}^{n-1} f_p\,g_{n-p} 
# \end{align}
# $$ (L2:45)
# 
# From this example we can understand that the first samples of the function $f$ interact with the last samples of the function $g$, and viceversa. This is obviously not the time convolution should do and, so, we need to choose the sampling time interval $[0,T)$ to avoid this inconvenience, i.e., with $T\geq T_f+T_g$ as discussed above.
# 
# ```

# ### Box functions
# 
# ```{div} full-width
# 
# Let us now implement the convolution approximation just discussed to the case of the convolution between two box functions $B_{T_f}$ and $B_{T_g}$ with $T_f=3$ s and $T_g=7$ s and choosing the sampling time interval $[0,T)$ with T$=20$ s $>T_f+T_g=10$ s
# ```

# In[15]:


from theory import eva_box

def eva_ramp(ts,T):
    ramp = ts.copy()
    ramp[ts<0] = 0
    ramp[ts>T] = T
    return ramp

def eva_heaviside_conv(ts,Tf,Tg):
    conv = eva_ramp(ts,Tf) - eva_ramp(ts-Tg,Tf)
    return conv


# In[16]:


def plot_heaviside_convolution(T,d,Tf,Tg):

    TT = max(T,Tf+Tg)
    
    tts = np.arange(0,TT,d)
    conv = eva_heaviside_conv(tts,Tf,Tg)

    ts = np.arange(0,T,d)
    n = len(ts)
    box_f = eva_box(ts,Tf)
    box_g = eva_box(ts,Tg)

    FT_box_f = np.fft.rfft(box_f) * d
    FT_box_g = np.fft.rfft(box_g) * d
    FT_conv  = FT_box_f * FT_box_g
    conv_approx = np.fft.irfft(FT_conv,n) / d

    fig, axes = plt.subplots(3,tight_layout=True,sharex=True,figsize=(10,6))
    
    ax = axes[0]
    ax.plot(ts,box_f,label="$T_f$ = "+str(Tf))
    ax.plot(ts,box_g,linestyle="dashed",label="$T_g$ = "+str(Tg))
    
    ax = axes[1]
    ax.plot(tts,conv,label="convolution")
    ax.plot(ts,conv_approx,linestyle="dashed",label="approximation")
    
    ax = axes[2]
    ax.plot(ts,conv[tts<T]-conv_approx)
    
    for ax in axes[:2]:
        ax.legend()
        
    if T < TT: 
        for ax in axes:
            ax.axvspan(0,T,color="gray",alpha=0.2)
    
    return fig


# In[17]:


Tf = 3
Tg = 7
T  = 20
d = 0.01

fig = plot_heaviside_convolution(T,d,Tf,Tg)


# ```{div} full-width
# 
# The issue concerning the circular convolution can be seen by choosing a smaller sampling interval as shown heareinafter for [0,8) s
#                                                                                                                              
# ```

# In[18]:


T  = 8

fig = plot_heaviside_convolution(T,d,Tf,Tg)


# ### Relative motion from driven harmonic oscillators

# In[19]:


from theory import eva_FT_green, eva_FT_response, test_driven_oscillator

T,d = 100,0.01
Tg = 3

def plot_green_convolution(T, d, Tg):
    
    ts = np.arange(0,T,d)
    n = len(ts)

    ground_motions = eva_test_ground_motion(ts, Tg)
    ground_acce = ground_motions[2]

    relative_motions, _ = test_driven_oscillator(ts, Tg)

    fs = np.fft.rfftfreq(n,d)
    FT_ground_acce = np.fft.rfft(ground_acce) * d

    FT_green = eva_FT_green(fs)
    green = np.fft.irfft(FT_green) / d

    FT_relative_disp = FT_ground_acce      * FT_green
    ws = 2*np.pi*fs
    relative_motions_approx = np.vstack( (np.fft.irfft(FT_relative_disp,n), np.fft.irfft(1j*ws * FT_relative_disp,n), np.fft.irfft(-ws**2 * FT_relative_disp,n) )) / d

    ylabels = ["Displacement [m]","Velocity [m/s]","Acceleration [m/s$^2$]"]

    fig, axes = plt.subplots(4,tight_layout=True,figsize=(10,8),sharex=True)
    for ax, ylabel, ground_motion, relative_motion, relative_motion_approx in zip(axes, ylabels, ground_motions, relative_motions, relative_motions_approx):
        ax.plot(ts,ground_motion,label="ground")
        ax.plot(ts,relative_motion,label="relative (exact)")
        ax.plot(ts,relative_motion_approx,linestyle="dashed",label="approximation")
        ax.set_ylabel(ylabel)

    ax = axes[-1]
    ax.plot(ts,green,label="Green function")
    ax.set_ylabel("[s$^2$]")

    for ax in axes:
        ax.legend(loc="upper right")
        ax.axhline(0,color="black",linewidth=0.5)
        
    return fig


# In[20]:


fig = plot_green_convolution(T, d, Tg)


# In[21]:


fig = plot_green_convolution(T, d, Tg)
fig.axes[-1].set_xlim(0,5)


# ### Filtering

# ```{div} full-width
# 
# Filters are usually defined and applied in the frequency domanin. For instante, let us consider the low-pass order-2 buttwerworth filter defined as
# 
# $$
# Q_L(\omega) = H\big(i\,\omega/\omega_L\big) = \frac{\omega_L^2}{\omega_L^2-\omega^2+i\,\sqrt{2}\,\omega\,\omega_L}
# $$
# 
# with $\omega_L$ being the cut-off angular frequency and $H$ being the so-called transfer function
# 
# $$
# H(s) = \frac{1}{s^2+\sqrt{2}\,s+1}
# $$
# 
# Starting from the same transferm function, we can also define the high-pass order-2 butterworth filter
# 
# $$
# Q_H(\omega) = H\big(\omega_H/(i\,\omega)\big) = \frac{\omega^2}{\omega^2-\omega_H^2+i\,\sqrt{2}\,\omega_H}
# $$
# 
# with $\omega_H$ being the cut-off angular frequency.
# 
# The band-pass order-2 butterworth filter, instead, is defined by
# 
# $$
# Q_B(\omega) = H_B\big(i\,\omega/\omega_C\big) =
# $$
# 
# $$
# H(s) = \frac{Q\,s}{s^2+Q\,s+1}
# $$
# 
# with $\omega_C$ and $Q$ being the central angular frequency and the inverse of the so called quality factor
# 
# $$
# \omega_C = \sqrt{\omega_L\,\omega_H}  \qquad Q = \frac{\omega_H-\omega_L}{\omega_C}
# $$
# 
# 
# ```

# In[22]:


help(signal.butter)


# In[226]:


help(signal.freqs)


# In[239]:


from scipy import signal

fl = 0.1
wl = 2*np.pi*fl

b, a = signal.butter(2, wl, 'low', analog=True)
ws, H = signal.freqs(b, a)

s = 1j*ws/wl
H_check = 1/(s**2+np.sqrt(2)*s+1)

fs = ws/(2*np.pi)

fig, axes = plt.subplots(2)
fig.suptitle('Butterworth filter  f < '+str(fl)+' [Hz]')

ax = axes[0]
ax.loglog(fs, abs(H))
ax.loglog(fs, abs(H_check), linestyle="dashed")
ax.set_ylabel('Amplitude')

ax = axes[1]
ax.semilogx(fs, np.angle(H,deg=True))
ax.semilogx(fs, np.angle(H_check,deg=True),linestyle="dashed")
ax.set_ylabel("Phase")
ax.set_ylim(-190,0)
ax.set_yticks([0,-90,-180])

ax.set_xlabel('Frequency [Hz]')

for ax in axes:
    ax.axvline(fl, color='green') # cutoff frequency


# In[240]:


from scipy import signal

fh = 10
wh = 2*np.pi*fh

b, a = signal.butter(2, wh, 'high', analog=True)
ws, H = signal.freqs(b, a)

s = wh/(1j*ws)
H_check = 1/(s**2+np.sqrt(2)*s+1)

fs = ws/(2*np.pi)


fig, axes = plt.subplots(2)
fig.suptitle('Butterworth filter  f > '+str(fh)+' [Hz]')

ax = axes[0]
ax.loglog(fs, abs(H))
ax.loglog(fs, abs(H_check), linestyle="dashed")
ax.set_ylabel('Amplitude')

ax = axes[1]
ax.semilogx(fs, np.angle(H,deg=True))
ax.semilogx(fs, np.angle(H_check,deg=True),linestyle="dashed")
ax.set_ylabel("Phase")
ax.set_ylim(0,190)
ax.set_yticks([0,90,180])

ax.set_xlabel('Frequency [Hz]')

for ax in axes:
    ax.axvline(fh, color='green') # cutoff frequency


# In[242]:


from scipy import signal

fl = 0.1
fh = 10

wl = 2*np.pi*fl
wh = 2*np.pi*fh

b, a = signal.butter(1, [wl,wh], 'bandpass', analog=True)
ws, H = signal.freqs(b, a)

fc = np.sqrt(fl*fh)
Q = (fh-fl)/fc
wc = 2*np.pi*fc
s = 1j*ws/wc
H_check = Q*s/(s**2+Q*s+wc**2)

fs = ws/(2*np.pi)

fig, axes = plt.subplots(2)
fig.suptitle('Butterworth filter  f > '+str(fl)+' [Hz] and f < '+str(fh)+' [Hz]')

ax = axes[0]
ax.loglog(fs, abs(H))
ax.loglog(fs, abs(H_check), linestyle="dashed")
ax.set_ylabel('Amplitude')

ax = axes[1]
ax.semilogx(fs, np.angle(H,deg=True))
ax.semilogx(fs, np.angle(H_check,deg=True),linestyle="dashed")
ax.set_ylabel("Phase")
ax.set_ylim(0,190)
ax.set_yticks([0,90,180])

ax.set_xlabel('Frequency [Hz]')

for ax in axes:
    ax.axvline(fh, color='green') # cutoff frequency


# In[208]:


def plot_filter_convolution(T, d, Tg, fl, fh):
    
    ts = np.arange(0,T,d)
    n = len(ts)

    relative_motions, _ = test_driven_oscillator(ts, Tg)

    fs = np.fft.rfftfreq(n,d)
    FT_relative_motions = np.fft.rfft(relative_motions) * d

    wl = 2*np.pi*fl
    wh = 2*np.pi*fh

    #b, a = signal.butter(2, 1, 'low', analog=True)
    b, a = signal.butter(1, [wl,wh], 'bandpass', analog=True)

    ws = 2*np.pi*fs
    ws, FT_filter = signal.freqs(b, a, ws)

    relative_motions_filtered = np.fft.irfft( FT_relative_motions * FT_filter, n ) / d
    
    ylabels = ["Displacement [m]","Velocity [m/s]","Acceleration [m/s$^2$]"]

    nrow = 4
    fig, axes = plt.subplots(nrow,tight_layout=True,figsize=(10,2*nrow))
    for ax, ylabel, relative_motion, relative_motion_filtered in zip(axes, ylabels, relative_motions, relative_motions_filtered):
        ax.plot(ts,relative_motion,label="relative")
        ax.plot(ts,relative_motion_filtered,label="filtered")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0,10)
        
    ax = axes[-1]
    ax.loglog(fs,abs(FT_filter))
    ax.axvline(1/Tg,color="red")
    ax.axvline(2/Tg,color="red")
    ax.axvline(fl,color="green")
    ax.axvline(fh,color="green")
    ax.axhline(1,color="black",linewidth=0.5)    

    for ax in axes[:3]:
        ax.legend(loc="upper right")
        ax.axhline(0,color="black",linewidth=0.5)

    return fig


# In[209]:


T, d = 100, 0.005
Tg = 1
fl, fh = 6e-1, 3e0
fig = plot_filter_convolution(T,d,Tg,fl,fh)


# In[210]:


fl, fh = 6e-2, 3e-1
fig = plot_filter_convolution(T,d,Tg,fl,fh)


# <p style="page-break-after:always;"></p>
