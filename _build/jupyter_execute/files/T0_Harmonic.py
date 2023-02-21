#!/usr/bin/env python
# coding: utf-8

# (chap-H)=
# # Harmonic oscillators
# 
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 
# ## Driven harmonic oscillators
# 
# ```{div} full-width
# 
# The driven harmonic oscillator is an useful tool in seismology because it can be used for describing either a simple mechanical seismometer or  the deformation of a building when they are subjected to ground motion.
# 
# ```
# 
# ### Seismometer
# 
# 
# ```{figure} ../images/Seismometer.png
# ---
# align: center
# figclass: margin-caption
# name: fig-seismometer
# ---
# Cartoon depicting a seismometer at rest and driven by ground motion (left and right pannels) and the absolute and reference systems.
# ```
# 
# 
# ```{div} full-width
# 
# Let us consider a seismometer composed by an probe mass attached to a spring and embedded in a viscous dashpot as depicted in {numref}`Figure {number} <fig-seismometer>`. Let us denote with $h_0$ the height of the probe mass at rest and consider the absolute coordinate of the probe mass $u$ as the sum of the ground motion $u_g$, the height at rest $h_0$ and its relative motion with respect to it $z$
# 
# $$
# u(t) = u_g(t) + h_0 + z(t)
# $$ (H0:1)
# 
# The elastic, $F_e$, and anaelastic, $F_a$, forces acting on the probe mass will be given by 
# 
# $$
# F_e=-k\,z \qquad F_a =-c\,\dot{z}
# $$ (H0:2)
# 
# From the second Newton's law
# 
# $$
# F_e+F_a = m\,\ddot{u}
# $$ (H0:3)
# 
# declined in the case of a machanical seismometer as the one depicted in figure (??), we thus obtain the following differential equation
# 
# $$
# -k\,z(t) - c\,\dot{z}(t) = m\,\big(\ddot{u}_g(t)+\ddot{z}(t)\big)
# $$ (H0:4)
# 
# where $m$ is the probe mass, $k$ is the elastic constant of the spring and $c$ is the anaelastic friction. We note that the left hand side is the sum of the total forces acting on the probe mass while the right-hand side is the inertial force, i.e., the mass by its total acceleration. 
# 
# By dividing eq. {eq}`H0:4` by the mass $m$ and defining the natural angular frequency $\omega_0$ and the damping ratio $\xi$ as follows
# 
# $$
# \omega_0 = \sqrt{\frac{k}{m}} \qquad \xi = \frac{c}{2\,\omega_0}
# $$ (H0:5)
# 
# we obtain
# 
# $$
# \ddot{z}(t) + 2\,\omega_0\,\xi\,\dot{z}(t) + \omega_0^2\,z(t) = -\ddot{u}_g(t)
# $$ (H0:6)
# 
# that is the classical form of a driven harmonic oscillator, where the left-hand side describes a damped harmonic oscilator and the right-hand side is the driving term that, in the present case, corresponds to minus the ground acceleration.
# 
# In the following we will also make use of the natural period of the harmonic oscillator, which is simply given by
# 
# $$
# T_0 =  \frac{2\,\pi}{\omega_0} = 2\,\pi\, \sqrt{\frac{m}{k}}
# $$ (H0:7)
# 
# ```
# 
# ### Building
# 
# ```{figure} ../images/Building.png
# ---
# align: center
# figclass: margin-caption
# name: fig-building
# ---
# Cartoon depicting a building at rest and driven by ground motion (left and right pannels) and the absolute and reference systems.
# ```
# 
# 
# 
# ````{div} full-width
# 
# Let su consider a building and, as depicted in {numref}`Figure {number} <fig-building>`, denote with $u$ the absolute coordinate of its top, which can be seen as the sum of the ground motion (i.e., at the bottom of the building) and the building height $h$
# 
# $$
# u(t) = u_g(t) + h(t) 
# $$ (H0:8)
# 
# Due to deformation of the building $h$ will vary with time and so we also write $h$ as the sum of its height at rest $h_0$ and its relative displacement $z$
# 
# $$
# h(t) = h_0 + z(t)
# $$ (H0:9)
# 
# The ratio $z/h_0$ can be seen as a measure of the internal strain and, so, the elastic and anaelastic forces will be proportional to $z$ and its time derivative $\dot{z}$, respectively
# 
# $$
# F_e = -\frac{\mu\,z}{h_0}
# \qquad
# F_a = -\frac{\eta\,\dot{z}}{h_0}
# $$ (H0:10)
# 
# with $\mu$ and $\eta$ being the elastic modulus and effective viscosity of the building. In thi respect, from the comparison with eq. {eq}`H0:2`, the elastic constant and anaelastic friction of the building are inversely proportional to the building height: $c\propto h_0^{-1}$ and $c\propto h_0^{-1}$.
# 
# Similar to the seismometer, from the second Newton's law, we thus can write 
# 
# $$
# -k\,z(t) - c\,\dot{z}(t) = m\,\big(\ddot{u}_g(t)+\ddot{z}(t)\big)
# $$ (H0:11)
# 
# and recast it in the form of the driven harmonic oscillator
# 
# $$
# \ddot{z}(t) + 2\,\omega_0\,\xi\,\dot{z}(t) + \omega_0^2\,z(t) = -\ddot{u}_g(t)
# $$ (H0:12)
# 
# 
# ```{admonition} The natural period of buildings
# :class: note
# 
# As a rule of thumb, we note that the the natural period of building is roughly propotional to the the number of floors $n$ in the building. Indeed, considering $h_0=n\,\Delta h$ anmd $m=n\,\Delta m$ with $\Delta h$ and $\Delta m$ being the height and the mass of each floors, we obtain
# 
# $$
# T_0 = 2\,\pi\,\sqrt{\frac{m}{k}} = n\,\delta T
# $$ (H0:13)
# 
# with $\delta T$ being the natural period of each floor
# 
# $$
# \delta T=2\,\pi\,\sqrt{\frac{\delta m}{\delta h}}
# $$ (H0:14)
# 
# In contrast, it can checked that the damping ratio $\xi$ does not vary with the number of floors and it usually ranges from $0.05$ to $0.1$ (or, in the woording of the  engineers, from $5\%$ to $10%$).
# 
# ```
# 
# 
# 
# ```{admonition} Relative or total acceleration?
# :class: note
# 
# As shown by eqs. {eq}`H0:6` and {eq}`H0:12`, both the relative motion of the seismometer and of the building are governed by the same differential equation, having the minus ground acceleration as the driven term. When we will discuss the seismometer, we will focus on the link betweem the ground and the relative motions as the former and the latter are what we want retrive and can measure, respectively. When we will discuss the building, instead, we will be more interested in quantifying the total acceleration to which it is subjected, and we will recover it as the sum of the ground and relative acceleration.
# 
# ```
# 
# ````
# 
# ## Damped Harmonic oscillator
# 
# ```{div} full-width
# 
# The damped harmonic oscillator is a classic problem in mechanics. It describes the movement of a mechanical oscillator (e.g. spring pendulum) under the influence of a restoring force and friction. Let us thus set to zero the ground motion acceletarion and solve the homogenous differential equation with specified initial condition on the position and velocity at time = $t=0$
# 
# $$
# \ddot{z}_\mathrm{h}(t) + 2\,\omega_0\,\xi\,\dot{z}_\mathrm{h}(t) + \omega_0^2\,z_\mathrm{h}(t) = 0
# $$
# $$
# z_\mathrm{h}(0) = u_0
# \qquad
# \dot{z}_\mathrm{h}(0) = v_0
# $$ (H0:15)
# 
# 
# The homogeneous solution can be found starting from this trial form
# 
# $$
# z_\mathrm{h}(t) = Z_+\,e^{\gamma_+\,t} + Z_-\,e^{\gamma_-\,t}
# $$ (H0:16)
# 
# and it is straigforwad to show that the two exponents $\gamma_\pm$ are the solution of a degree-2 polynomial 
# 
# $$
# Z_\pm\,\big(\gamma_\pm^2+2\,\omega_0\,\xi\,\gamma_\pm + \omega_0\big)\,e^{\gamma_\pm\,t} = 0
# $$ (H0:17)
# 
# and read
# 
# $$
# \gamma_\pm = -\omega_0\,\xi \pm \omega_0\,\sqrt{\xi^2-1}
# $$ (H0:18)
# 
# ```
# 
# ### Underdamped harmonic oscillator
# 
# 
# ```{div} full-width
# 
# Within the assumption that $\xi<1$, the square root in the right-hand side of eq. {eq}`H0:18` is imaginary and the two exponents can be rewritten as folllows
# 
# $$
# \gamma_\pm = -\lambda \pm i\,\omega_d
# $$ (H0:19)
# 
# with
# 
# $$
# \lambda = \omega_0\,\xi \qquad
# \omega_d = \omega_0\,\sqrt{1-\xi^2}
# $$ (H0:20)
# 
# In view of the fact that the real parts of the two exponents coincide, we can recast eq. {eq}`H0:16` as follows
# 
# $$
# \begin{align}
# z_\mathrm{h}(t) &= \big(Z_+\,e^{i\,\omega_d\,t} + Z_-\,e^{-i\,\omega_d\,t}\big)\,e^{-\lambda\,t} \\
#                 &= \big(C\,\cos(\omega_d\,t) + S\,\sin(\omega_d\,t)\big)\,e^{-\lambda\,t}
# \end{align}
# $$ (H0:21)
# 
# where $C$ and $S$ are two constants of integration related to the former by
# 
# $$
# C = Z_+ + Z_-   \qquad
# S=i\,\big(Z_+ - Z_-\big)
# $$ (H0:22)
# 
# In light of this, the assumption of  $\xi<1$ leads to underdamped oscillations: the system oscillates (with a slightly different frequency than the natural frequency) with the amplitude gradually decreasing to zero. The angular frequency of the underdamped harmonic oscillator is $\omega_d$  and the exponential decay of the underdamped harmonic oscillator is $\lambda$. In light of this, the homogeneous solution of the harmonic oscillator can be seen as a transient response with an exponential decay.
# 
# The cases $\xi=1$ and $\xi>1$, instead, describe critical and overdamped oscillations. They are not relevant in seismology and we do not discuss them.
# 
# ```
# 
# ### Initial conditions
# 
# ```{div} full-width
# 
# Considering the first order derivative of eq. {eq}`H0:21`
# 
# $$
# \dot{z}_\mathrm{h}(t) = \big[-(C\,\omega_d+S\,\lambda)\,\sin(\omega_d\,t) + (S\,\omega_d-C\,\lambda)\,\cos(\omega_d\,t)\big]\,e^{-\lambda\,t} 
# $$ (H0:23)
# 
# from eq. {eq}`H0:15` we have 
# 
# $$
# z_\mathrm{h}(0)  = C  = u_0
# \qquad \dot{z}_\mathrm{h}(0)  = S\,\omega_d - C\,\lambda  = v_0
# $$ (H0:24)
# 
# This system of equations is solved by
# 
# $$
# C = u_0 \qquad 
# S = \frac{u_0\,\lambda + v_0}{\omega_d} 
# $$ (H0:25)
# 
# 
# and the transient response, eq. {eq}`H0:21`, and its derivatives become
# 
# $$
# \begin{align}
# & z_\mathrm{h}(t) = \left(u_0\,\cos(\omega_d\,t) +\frac{u_0\,\lambda  + v_0}{\omega_d}  \,\sin(\omega_d\,t)\right)\,e^{-\lambda\,t}
# \\
# & \dot{z}_\mathrm{h}(t) = \left(v_0  \,\cos(\omega_d\,t) - \frac{u_0\,\omega_0^2  + v_0\,\lambda}{\omega_d}\,\sin(\omega_d\,t)\right)\,e^{-\lambda\,t}
# \\
# &\ddot{z}_\mathrm{h}(t) = \left[a_0\,\,\cos(\omega_d\,t)-\frac{v_0\,\omega_0^2 + a_0\,\lambda }{\omega_d}  \,\sin(\omega_d\,t)\right]\,e^{-\lambda\,t}
# \end{align}
# $$ (H0:26)
# 
# with $a_0$ being the initial acceleration
# 
# $$
# a_0 = -\big(u_0\,\omega_0^2  + 2\,v_0\,\lambda\big)
# $$ (H0:27)
# 
# ```

# ```{div} full-width
# 
# Eq. {eq}`H0:26` is implemented by the function `damped_harmonic_oscillator` of the `theory` module that we can use for compunting the transient response
# 
# ```

# In[1]:


from theory import damped_harmonic_oscillator
help(damped_harmonic_oscillator)


# ```{div} full-width
# 
# and we plot them for different choises of the initial conditions 
# 
# ```

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from myst_nb import glue
np.set_printoptions(precision=3,suppress=True)
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.ioff()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt

ts = np.linspace(0,10,10001)
fig,axes = plt.subplots(3,tight_layout=True,figsize=(10,6))

U, V = 1, 0
relative_motions = damped_harmonic_oscillator(ts, U, V)
for ax,relative_motion in zip(axes,relative_motions):
    ax.plot(ts,relative_motion,label="U = "+str(U)+" m")

U, V = 0,6
relative_motions = damped_harmonic_oscillator(ts, U, V)
for ax,relative_motion in zip(axes,relative_motions):
    ax.plot(ts,relative_motion,label="V = "+str(V)+" m/s")

ylabels= ["Displacement [m]", "Velocity [m/s]", "Acceleration [m/s$^2$]"]

options = dict(color="black",linewidth=0.5)
for ax, ylabel in zip(axes, ylabels):
    ylim = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim(-ylim,ylim)
    ax.axhline(0,**options)
    ax.axvline(0,**options)
    ax.set_ylabel(ylabel)
    ax.legend()


# In[33]:


glue("fig_transient", fig)


# ````{div} full-width
# ```{glue:figure} fig_transient
# :name: fig-transient
# 
# Transient response (displacement, velocity and acceleration) of the damped harmonic oscillator, eq. {eq}`H0:26`, with initial conditions $u_0=1$ m and $v_0=0$ m/s (blue line), and $u_0=0$ m and $v_0=6$ m/s (orange line). The natural period and damping ratio of the harmonic oscillator are $T=1$ s and $\xi=0.1$.
# ```
# ````

# In[34]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## General solution
# 
# ```{div} full-width
# 
# In view of the fact that the differential equation of the driven harmonic oscillator is linear in both $z$ and $u$ and their derivatives, its general solution, for any ground acceleration, can be expressed in the following form of a time convolution 
# 
# $$
# z(t) = \int_{-\infty}^\infty \ddot{u}_g(\tau)\,G(t-\tau)\,d\tau
# $$ (H0:28)
# 
# between the ground acceleration and a Green function $G$ that we are going to determine.
# 
# ```
# 
# ### Green function
# 
# ```{div} full-width
# 
# Substituting eq. {eq}`H0:28` into eq. {eq}`H0:6` or {eq}`H0:12` we obtain
# 
# $$
# \int_{-\infty}^\infty \ddot{u}_g(\tau)\,\big(\ddot{G}(t-\tau)+2\,\lambda\,\dot{G}(t-\tau)+\omega_0^2\,G(t-\tau)\big)\,d\tau = -\ddot{u}_g(t)
# $$ (H0:29)
# 
# From this point of view, we can understand that we are looking for a Green function $G$ such that
# 
# $$
# \ddot{G}(t)+2\,\lambda\,\dot{G}(t)+\omega_0^2\,G(t) = - \delta(t)
# $$ (H0:30)
# 
# In this perspective, we test the following trial solution
# 
# $$
# G(t) = f(t)\,H(t)
# $$ (H0:31)
# 
# with $f$ satisfying the following initial conditions
# 
# $$
# f(0) = 0 \qquad \dot{f}(0) = -1
# $$ (H0:32)
# 
# These initial conditions have been defined in such a way that
# 
# $$
# \begin{align}
# & \dot{G}(t) = \dot{f}(t)\,H(t) + f(0)\,\delta(t) =  \dot{f}(t)\,H(t)
# \\
# & \ddot{G}(t) = \ddot{f}(t)\,H(t) + \dot{f}(0)\,\delta(t) = \ddot{f}(t)\,H(t) -\delta(t)  
# \end{align}
# $$ (H0:33)
# 
# and, so, eq. {eq}`H0:30` simplifies into 
# 
# $$
# \big(\ddot{f}(t)+2\,\lambda\,\dot{f}(t)+\omega_0^2\,f(t)\big)\,H(t) = 0
# $$ (H0:34)
# 
# In this respect, $f$ is the homoegenous solution for the harmonic oscillator with initial conditions given by eq. {eq}`H0:32`. From eq. {eq}`H0:26` we thus obtain
# 
# $$
# f(t) = -\frac{\sin(\omega_d\,t)\,e^{-\lambda\,t}}{\omega_d}
# $$ (H0:35)
# 
# and the Green function becomes
# 
# $$
# G(t) = -\frac{\sin(\omega_d\,t)\,e^{-\lambda\,t}}{\omega_d}\,H(t)
# $$ (H0:35bis)
# ```
# 
# ### Response function
# 
# ```{div} full-width
# Starting from the Green function, eqs. {eq}`H0:31` and {eq}`H0:35`, we can also obtain the response function $R$ relating the ground and relative accelerations
# 
# $$
# \ddot{z}(t) = \int_{-\infty}^\infty \ddot{u}_g(\tau)\,\ddot{G}(t-\tau)\,d\tau = \int_{-\infty}^\infty \ddot{u}_g(\tau)\,R(t-\tau)\,d\tau
# $$ (H0:36)
# 
# that is the second order derivative of $G$
# 
# $$
# \begin{align}
# R(t) &= \ddot{G}(t) = \ddot{f}(t)\,H(t) -\delta(t)
# \\
# &= \left[ \frac{\omega_d^2-\lambda^2}{\omega_d}\,\sin(\omega_d\,t) +2\,\lambda\,\cos(\omega_d\,t)\right]\,e^{-\lambda\,t}\,H(t) - \delta(t)
# \end{align}
# $$ (H0:37)
# 
# It is straightforward to show that the response function also relates the ground and relative displacements, as well as the ground and relative velocities
# 
# 
# $$
# z(t) = \int_{-\infty}^\infty u_g(\tau)\,R(t-\tau)\,d\tau
# \qquad
# \dot{z}(t) = \int_{-\infty}^\infty \dot{u}_g(\tau)\,R(t-\tau)\,d\tau
# $$ (H0:38)
# 
# Although the Green and response functions allow to determine the relative ground motion, their application requires a time convolution that can be tedious to be computed analytically and hide the physical understanding of the problem. We will use them in the frequency domain after {ref}`chap-F` and to make practise on the {ref}`chap-D`.
# 
# ```
# 

# ## Sinusoidal ground acceleration
# 
# ```{div} full-width
# 
# In order to understand how the harmonic oscillator behaves when driven by ground motion, let us consider the following ground acceleration with period $T_g$ 
# 
# $$
# \ddot{u}_g = g(t)\,H(t)
# $$ (H0:39)
# 
# where $g$ is a monochromatic signal with angular frequency $\omega_g$ and phase $\phi_g$
# 
# $$
# g(t) = A_g\,\sin(x) \qquad x=\omega_g\,t+\phi_g \qquad
# \omega_g = \frac{2\,\pi}{T_g}
# $$ (H0:40)
# 
# 
# Let us implement eqs. {eq}`H0:39` and {eq}`H0:40` in python 
# ```

# In[35]:


def eva_sinusoidal_acceleration(ts, Tg=1, ground_phase=0):
    
    wg = 2*np.pi / Tg
    
    x = wg*ts +  ground_phase
    g = np.sin(x)
    ground_acceleration = g.copy()
    ground_acceleration[ts<0] = 0

    return ground_acceleration, g


# ```{div} full-width
# and plot the ground acceleration and the monochromatic signal
# ```

# In[36]:


del fig
plt.ioff()


# In[37]:


Tg = 1
ts = np.linspace(-2*Tg,5*Tg,10001)
ground_acceleration, g = eva_sinusoidal_acceleration(ts, Tg)

fig,ax = plt.subplots(tight_layout=True,figsize=(10,2))
ax.plot(ts,ground_acceleration,label="ground")
ax.plot(ts,g,linestyle="dashed",label="g")
ax.axvline(0,**options)
ax.axhline(0,**options)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [m/s$^2$]")
ax.legend();


# In[38]:


glue("fig_sinusoidal", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_sinusoidal
# :name: "fig-sinusoidal"
# 
# Sinusoidal ground acceleratio $u_g$ (blue line, eq. {eq}`H0:39`) and the associated monochromatic signal $g$ (orange dashed line, eq. {eq}`H0:40`)  for  $T_g=1$ s, $\phi_g=0$ and $A_g=1$ m/s$^2$.
# ```
# ````

# In[39]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# 
# As we are going to verify, the relative motion can be expressed in the following form
# 
# $$
# z(t) = \big(z_\mathrm{p}(t) + z_\mathrm{h}(t)\big)\,H(t) 
# $$ (H0:41)
# 
# where $z_\mathrm{h}$ is an homogeneous solution and $z_\mathrm{p}$ is a particular solution of the harmonic oscillator driven by the monochromatic signal
# 
# $$
# \ddot{z}_p(t)+2\,\lambda\,\dot{z}_p(t)+\omega_0\,z_\mathrm{p}(t) = -g(t)
# $$ (H0:42)
# 
# Here the Heaviside function $H$ has been introduced because there is no motion before the beginning of the ground motion and it affects the first and second order derivatives as follow
# 
# 
# $$
# \begin{align}
# &\dot{z}(t) = (\dot{z}_\mathrm{p}(t)+\dot{z}_\mathrm{h}(t))\,H + (z_\mathrm{p}(t)+z_\mathrm{h}(t))\,\delta(t)
# \\
# &\ddot{z}(t) = (\ddot{z}_\mathrm{p}(t)+\ddot{z}_\mathrm{h}(t))\,H + 2\,(\dot{z}_\mathrm{p}(t)+\dot{z}_\mathrm{h}(t))\,\delta(t) + (z_p+z_h)\,\dot{\delta}(t)
# \end{align}
# $$ (H0:43)
# 
# Once substituded in eq. {eq}`H0:6`, we obtain
# 
# 
# $$
# 2\,(\dot{z}_\mathrm{p}(t)+\dot{z}_\mathrm{h}(t))\,\delta(t) + (z_\mathrm{p}(t)+z_\mathrm{h}(t))\,\dot{\delta}(t) + 2\,\lambda\,(z_\mathrm{p}(t)+z_\mathrm{h}(t))\,\delta(t) = 0
# $$ (H0:44)
# 
# Within this framework, we can understand that eq. {eq}`H0:41` solves eq. {eq}`H0:6` before and after the initial time, within the time intervals $(-\infty,0)$ and $(0,\infty)$.  In order that it solve the differential equation also in the surrounding of the initial time $t=0$, we need to choose the homogeneous solution in such a way that $z$ and its first derivative are continuous at $t=0$, which means that we have to require that 
# 
# $$
# z_\mathrm{p}(0)+z_\mathrm{h}(0) = 0 \qquad \dot{z}_\mathrm{p}(0) +\dot{z}_\mathrm{h}(0) = 0
# $$ (H0:45)
# 
# We can fullfil this requirement by setting the initial displacement and velocity for the homoegenous solution have to be choosen as follows
# 
# $$
# u_0 = -z_\mathrm{p}(0)  \qquad v_0 = -\dot{z}_\mathrm{p}(0) 
# $$ (H0:46)
# 
# ```
# 
# ```{admonition} Transient and steady-state solutions
# :class: note, full-width
# 
# Although we have not yet determined $z_\mathrm{p}$ and $z_\mathrm{h}$, we can already state that the latter describes a transient response of which the amplitude gradually decreases to zero with the exponential decay $\lambda$. The former, instead, describes the steady state solution that persists and will be the only one after the initial transient.
# 
# ```
# 
# 
# 
# ### Particular (or steady state) solution
# 
# ````{div} full-width
# 
# Let us solve eq. {eq}`H0:42` starting from the following trial solution
# 
# $$
# z_p(t) = - A_g\,\frac{\rho\,\sin(x+\phi)}{\omega_g^2}
# $$ (H0:47)
# 
# The relative velocity and acceleration read
# 
# $$
# \dot{z}_p(t) = - A_g\,\frac{\rho\,\cos(x+\phi)}{\omega_g}
# \qquad \ddot{z}(t) = A_g\,\rho\,\sin(x+\phi) 
# $$ (H0:48)
# 
# and, so, $\rho$ and $\phi$ can be considered as the amplitude factor and phase shift of the relative acceleration with respect to the ground acceleration. 
# 
# Substituting eqs. {eq}`H0:47` and {eq}`H0:48` into eq. {eq}`H0:6`, making use of the following trigognometric identities 
# 
# $$
# \begin{align}
# \sin(x+\phi) = \sin x\,\cos\phi + \cos x\,\sin\phi
# \\
# \cos(x+\phi) = \sin x\,\sin\phi - \cos x\,\cos\phi
# \end{align}
# $$ (H0:49)
# 
# and collecting the sine, $\sin x$, and cosine, $\cos x$, terms, we obtain
# 
# $$
# -\frac{\rho}{\omega_g^2}\,\left[\cos x\,\Big(\big(\omega_0^2-\omega_g^2\big)\,\sin\phi + 2\,\omega_g\,\lambda\,\cos\phi\Big)
# +\sin x\,\Big(\big(\omega_0^2-\omega_g^2\big)\,\cos\phi - 2\,\omega_g\,\lambda\,\sin\phi\Big)\right] = - \sin x 
# $$ (H0:50)
# 
# Just because the right-hand side is characterized by the only sine term, the factor of $\cos x$ in left-hand side have to be zero and we can do it choosing
# 
# $$
# \cos\phi = \frac{\omega_0^2-\omega_g^2}{D} \qquad \sin\phi = \frac{-2\,\omega_g\,\lambda}{D}  \qquad \mathrm{with}\qquad
# D = \sqrt{\big(\omega_0^2-\omega_g^2)^2+(2\,\omega_g\,\lambda)^2}
# $$ (H0:51)
# 
# With this choice, eq. {eq}`H0:50` becomes
# 
# $$
# -\frac{\rho}{\omega_g^2}\,\sin x\,D = - \sin x
# $$ (H0:52)
# 
# that is satisfied by
# 
# $$
# \rho=\frac{\omega_g^2}{D}
# $$ (H0:53)
# 
# ```{admonition} Resonance frequency
# :class: note
# 
# Let us investigate the frequency at which the amplitude factor $\rho$ reaches its maximum value. By investigating the stationary points
# 
# $$
# \frac{\partial \rho}{\partial \omega_g} = \frac{2\,\omega_g\,\omega_0^2}{D^3}\,\big[\omega_0^2-\omega_g^2\,\big(1-2\,\xi^2\big)\big]
# $$ (H0:54)
# 
# we found that $\omega_g=0$ is a minimum, while 
# 
# $$
# \omega_g = \omega_0\,\sqrt{1-2\,\xi^2}
# $$ (H0:55)
# 
# is a maximum. The latter, however, there exists only when $\xi<1/\sqrt{2}\approx 0.707$.
# ```
# 
# ````

# ```{div} full-width
# 
# Eqs. {eq}`H0:47` and {eq}`H0:48` are implemented by the functions `setup_harmonic_oscilator` and `driven_sinusoidal_oscillator` of the `theory` module that we can use for compunting the steady state relative motion due to the sinusoidal ground acceleration
# 
# ```

# In[40]:


from theory import setup_harmonic_oscillator, steadystate_driven_oscillator

help(setup_harmonic_oscillator)
help(steadystate_driven_oscillator)


# ```{div} full-width
# 
# and plot the comparison between the ground acceleration and the relative, $\ddot{z}$, and total, $\ddot{u}=\ddot{u}_g+\ddot{z}$, accelerations for different ground periods 
# 
# ```

# In[41]:


del fig
plt.ioff()


# In[48]:


Tgs = [5,1.5,1,0.1]

damping_ratio = 0.1
label = r"$(\xi={0:<5.3})$".format(damping_ratio)

hvline_options = dict(color="black", linewidth=0.5)
fig,axes = plt.subplots(len(Tgs), 2, tight_layout=True, figsize=(10,2*len(Tgs)), sharey="row")

for axe, Tg in zip(axes, Tgs):
    
    tmin = -2*Tg
    tmax =  5*Tg
    ts = np.linspace(tmin, tmax, 10001)
    ground_acceleration, relative_motions = steadystate_driven_oscillator(ts, Tg)

    ax = axe[0]
    ax.plot(ts, ground_acceleration, label="ground")
    ax.plot(ts, relative_motions[2], label="relative")
    ax.set_ylabel("Acceleration [m/s$^2$]")
    ax = axe[1]
    ax.plot(ts, ground_acceleration, label="ground")
    ax.plot(ts, ground_acceleration+relative_motions[2], label="total")

    for ax in axe:
        ax.legend(title=r"T$_g$="+str(Tg)+" s",loc="upper right", framealpha=1)

axes[0,0].set_title(r"RELATIVE "+label+":  $\ddot{z}(t)$")
axes[0,1].set_title(r"TOTAL "+label+":  $\ddot{u}(t) = \ddot{z}(t)+\ddot{u}_g(t)$")

for ax in axes[-1]:
    ax.set_xlabel("Time [s]")
    
for ax in axes.flatten():
    ax.axhline(0, **hvline_options)
    ax.axvline(0, **hvline_options)


# In[49]:


glue("fig_driven_building", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_driven_building
# :name: fig-driven-building
# 
# Comparison between the sinusoidal ground acceleration (blue line) and the relative and total accelerations (left and right columns, orange lines) of the driven harmonic oscillator with ground periods $T_g=5,1.5,1$ and $0.1$ s (from top to bottom) and the damping ratio set to $\xi=0.1$. The natural period of the harmonic oscillator is always set to $T=1$ s.
# ```
# ````

# In[50]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# 
# We note that the relative ground acceleration can be amplified or damped with respect to the ground acceleration, as well as shifted in time. In particular, for ground periods much shorter than the natural period of the harmonic oscillator, we have that the relative acceleration is almost opposite to the ground accelaration, resulting into a negligible total accelaration. On the contrary, for ground periods much larger than the natural period, we have that the relative acceleration is negligible, resulting into a total acceleration almost equal to that of the ground.
# 
# Let us now consider the case of a damping ratio $\xi=1/\sqrt{2}\approx 0.707$ and make a similar plot to that shown in {numref}`Figure {number} <fig-building>`
# ```

# In[108]:


del fig
plt.ioff()


# In[109]:


damping_ratio = 1/np.sqrt(2)
label = r"$(\xi={0:<5.3})$".format(damping_ratio)

hvline_options = dict(color="black", linewidth=0.5)
fig,axes = plt.subplots(len(Tgs), 2, tight_layout=True, figsize=(10,2*len(Tgs)), sharey="row")

for axe, Tg in zip(axes, Tgs):
    
    tmin = -2*Tg
    tmax =  5*Tg
    ts = np.linspace(tmin, tmax, 10001)
    ground_acceleration, relative_motions = steadystate_driven_oscillator(ts, Tg, damping_ratio=damping_ratio)

    ax = axe[0]
    ax.plot(ts, ground_acceleration, label="ground")
    ax.plot(ts, relative_motions[2], label="relative")
    ax.set_ylabel("Acceleration [m/s$^2$]")
    ax = axe[1]
    ax.plot(ts, ground_acceleration, label="ground")
    ax.plot(ts, ground_acceleration+relative_motions[2], label="total")

    for ax in axe:
        ax.legend(title=r"T$_g$="+str(Tg)+" s",loc="upper right", framealpha=1)

axes[0,0].set_title(r"RELATIVE "+label+":  $\ddot{z}(t)$")
axes[0,1].set_title(r"TOTAL "+label+":  $\ddot{u}(t) = \ddot{z}(t)+\ddot{u}_g(t)$")

for ax in axes[-1]:
    ax.set_xlabel("Time [s]")
    
for ax in axes.flatten():
    ax.axhline(0, **hvline_options)
    ax.axvline(0, **hvline_options)


# In[110]:


glue("fig_driven_seismometer", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_driven_seismometer
# :name: fig-driven-seismometer
# 
# Comparison between the sinusoidal ground acceleration (blue line) and the relative and total accelerations (left and right columns, orange lines) of the driven harmonic oscillator with ground periods $T_g=5,1.5,1$ and $0.1$ s (from top to bottom) and the damping ratio set to $\xi=1/\sqrt{2}\approx 0.707$. The natural period of the harmonic oscillator is always set to $T=1$ s.
# ```
# ````

# In[111]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# 
# In order to have the full picture of the different behavior of the relative acceleration as function of the ground period, we plot here the amplitude factor and phase as function of the ground frequency and for different damping ratios
# 
# ```

# In[ ]:





# In[61]:


del fig
plt.ioff()


# In[62]:


fs = 1e1**np.linspace(-2,2,10001)
wg = 2*np.pi*fs

fig,axes = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

for damping_ratio in [0.1,0.2,0.4,1/np.sqrt(2),0.95]:
    
    label = "{0:<5.3}".format(damping_ratio)

    amplitude_factor, phase = setup_harmonic_oscillator(wg, damping_ratio=damping_ratio)
    phase *= 180/np.pi

    axes[0].semilogx(fs, amplitude_factor, label=label)
    axes[1].semilogx(fs, phase, label=label)

axes[1].set_yticks([0,-45,-90,-135,-180])
axes[0].axhline(1,**options)
axes[1].axhline(-180,**options)
axes[0].set_ylabel(r"Amplitude factor $\rho$")
axes[1].set_ylabel(r"Phase $\phi$ [deg]")

for ax in axes:
    ax.axhline(0,**options)
    ax.set_xlabel("Ground frequency $f_g=1/T_g$ [Hz]")
    ax.legend()


# In[63]:


glue("fig_response", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_response
# :name: fig-response
# 
# Amplitude factor $\rho$, eq. {eq}`H0:53`, and phase $\phi$, eq. {eq}`H0:51`, as function of the ground frequency for different damping ratios $\xi=0.1,0.2,0.4,0.707$ and $0.95$. The natural period of the harmonic oscillator is always set to $T=1$ s.
# ```
# ````

# In[64]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# From this plot, we can understand that at high frequency the amplitude of the relative motion is the same of the ground motion, although shifted by a phase of $-180^\circ$, which results thus in $\ddot{z}_\mathrm{p} = - \ddot{u}_g$.
# ```

# ### Transient response
# 
# ```{div} full-width
# 
# Having obtained the particular (steady-state) relative motion, we can now determine the associated transient response with the initial conditions given by eq. {eq}`H0:46` making use of eqs. {eq}`H0:47` and {eq}`H0:48`. In particular, the transient response $z_\mathrm{h}$ can be obtained from eq. {eq}`H0:26` using 
# 
# $$
# u_0 = -z_\mathrm{p}(0) = \frac{\rho\,\sin(\phi_g+\phi)}{\omega_g^2} \qquad v_0 = -\dot{z}_\mathrm{p}(0) = \frac{A\,\cos(\phi_g+\phi)}{\omega_g}
# $$ (H0:56)
# 
# as initial conditions.
# 
# The steady-state and transient solutions due to the sinusoidal ground acceleration are implemented by the function `sinuosidal_driven_oscillator` of the `theory` module
# ```

# In[65]:


from theory import sinusoidal_driven_oscillator

help(sinusoidal_driven_oscillator)


# ```{div} full-width
# and we use it for plotting the comparison between the ground and relative motion, as well as visualize the contribution from the transient motion
# ```

# In[66]:


del fig
plt.ioff()


# In[86]:


Tg = 1
tmin = - 3*Tg
tmax =  20*Tg
ts = np.linspace(tmin,tmax,10001)
ground_acceleration, relative_motions, transient_motions = sinusoidal_driven_oscillator(ts, Tg)


fig,ax = plt.subplots(tight_layout=True,figsize=(10,3),sharey=True)

ax.plot(ts, ground_acceleration, label="ground")
ax.plot(ts, relative_motions[2], label="relative")
ax.plot(ts, transient_motions[2], label="transient", linestyle="dashed")
ax.axhline(0, color="black", linewidth=0.5)

ax.legend()
ax.set_ylabel("Acceleration [m/s$^2$]")
ax.set_xlabel("Time [s]");


# In[87]:


glue("fig_full", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_full
# :name: fig-full
# 
# Ground, relative and transient accelerations (blue, orange and green lines) for the driven harmonic oscillator with ground period and phase set to $T_g=1$ s and $\phi_g=0$. The natural period and the damping ratio of the harmonic oscillator are $T=1$ s and $\xi=0.1$.
# ```
# ````

# In[69]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Test function
# 
# ```{div} full-width
# 
# The sinusoidal ground acceleration, although usefull for a first understanding of the bahaviour of the driven harmonic oscillator, does not describe a typical seismic wave. Considering the ground velocity and displacement, indeed, we have
# 
# $$
# \dot{u}_g(t) = \int_{-\infty}^t \ddot{u}_g(\tau)\,d\tau =  H(t)\,A_g\,\int_{0}^t \sin(\omega_g\,\tau+\phi_g)\,d\tau = A_g\,\left(\frac{\cos \phi_g-\cos(\omega_g\,t+\phi_g)}{\omega_g}\right)\,H(t)
# $$ (H0:57)
# 
# $$
# u_g(t) = \int_{-\infty}^t \dot{u}_g(\tau)\,d\tau =  \frac{1}{\omega_g}\,\left(t\,\cos\phi_g-\frac{\sin(\omega_g\,t)}{\omega_g}\right)\,H(t)
# $$ (H0:58)
# 
# and we can see that the ground displacement is characterized by a linear trend beginning at $t=0$.
# 
# A smooth function that can be used as seismic wavelet packet for describing realistic ground displacement, velocity and acceleration is
# 
# $$
# \begin{align}
# & \ddot{u}_g(t) = A_g\,\big[\cos(\omega_g\,t) - \cos(2\,\omega_g\,t)\big]\,B_{g}(t)
# \\
# & \dot{u}_g(t) = \frac{A_g}{\omega_g}\,\big[\sin(\omega_g\,t)-\tfrac{1}{2}\,\sin(2\,\omega_g\,t))\big]\,B_{g}(t)
# \\
# & u_g(t) = \frac{A_g}{\omega_g^2}\,\big[\tfrac{1}{4}\,\cos(2\,\omega_g\,t)-\cos(\omega_g\,t) + \tfrac{3}{4}\big]\,B_{g}(t)
# \end{align}
# $$ (H0:59)
# 
# where $B_g(t)$ is the box function over the time interal $[0,T_g$]
# 
# $$
# B_{T_g}(t) = H(t) - H(t-T_g)
# $$ (H0:60)
# 
# Eq. {eq}`H0:59` is implemented by the function `eva_test_ground_motion` 
# 
# ```

# In[88]:


from theory import eva_test_ground_motion

help(eva_test_ground_motion)


# ```{div} full-width
# that we use here for visualize the test ground displacement, velocity and acceleration
# ```

# In[89]:


del fig
plt.ioff()


# In[90]:


Tg = 1
ts = np.linspace(-Tg,6*Tg,10001)
ground_motions = eva_test_ground_motion(ts, Tg)

ylabels = ["Displacement [m]","Velocity [m/s]","Acceleration [m/s$^2$]"]

fig,axes = plt.subplots(3,tight_layout=True,figsize=(10,6),sharex=True)
axes[-1].set_xlabel("Time [s]")

for ax, ground_motion,ylabel in zip(axes, ground_motions, ylabels):

    ax.plot(ts,ground_motion)
    ax.set_ylabel(ylabel)
    ax.axhline(0,**options)


# In[91]:


glue("fig_test", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_test
# :name: fig-test
# 
# Test ground motion (displacement, velocity and acceleration) given by eq. {eq}`H0:59` and $T_g=1$ s.
# ```
# ````

# In[92]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# 
# The particular solution $z_p$ solving eq. {eq}`H0:42` with $g=A_g\,\big[\cos(\omega_g\,t) - \cos(2\,\omega_g\,t)\big]$ is
# 
# $$
# \begin{align}
# & \ddot{z}_p(t) = A_g\,\big[\rho(\omega_g)\,\cos\big(\omega_g\,t+\phi(\omega_g)\big) - A(2\,\omega_g)\,\cos\big(2\,\omega_g\,t+\phi(2\,\omega_g)\big)\big]
# \\
# & \dot{z}_p(t) = \frac{A_g}{\omega_g}\,\big[\rho(\omega_g)\,\sin\big(\omega_g\,t+\phi(\omega_g)\big) - \tfrac{1}{2}\,\sin\big(2\,\omega_g\,t+\phi(2\,\omega_g)\big)\big]
# \\
# & z_p(t) = \frac{A_g}{\omega_g^2}\,\big[ \tfrac{1}{4}\,\rho(2\,\omega_g)\,\cos\big(2\,\omega_g\,t+\phi(2\,\omega_g)\big) -\rho(\omega_g)\,\cos\big(\omega_g\,t+\phi(\omega_g)\big)\big]
# \end{align}
# $$ (H0:61)
# 
# that we obtain from eqs. {eq}`H0:47` and  {eq}`H0:48` with $\phi_g=\pi/2$. At the initial, $t=0$, and end, $t=T_g$, times of the wavelet packet, the particular solution for the displacement and velocity yields
# 
# $$
# \begin{align}
# & \dot{z}_p(0) = \dot{z}_p(T_g) =  \frac{A_g}{\omega_g}\,\big[\rho(\omega_g)\,\sin\phi(\omega_g)- \tfrac{1}{2}\,\sin\phi(2\,\omega_g)\big]
# \\
# & z_p(0) = z_p(T_g) = \frac{A_g}{\omega_g^2}\,\big[ \tfrac{1}{4}\,\rho(2\,\omega_g)\,\cos\phi(2\,\omega_g) -\rho(\omega_g)\,\cos\phi(\omega_g)\big]
# \end{align}
# $$ (H0:62)
# 
# In this respect, in order to make the relative motion due to the test ground displacement continuous at $t=0,T_g$, we write
# 
# $$
# z(t) = z_p(t)\,B_g(t) + z_h(t)\,H(t) - z_h(t-T_g)\,H(t-T_g)
# $$ (H0:63)
# 
# where $z_h$ is the homogeneous obtained setting as initial conditions
# 
# $$
# u_0 = -z_p(0) \qquad v_0 = -\dot{z}_p(0)
# $$ (H0:64)
# 
# so that
# 
# $$
# z(0) = z(T_g) = 0 \qquad \dot{z}(0) = \dot{z}(T_g) = 0
# $$ (H0:65)
# 
# The function `test_driven_oscillator` of the `theory` module implements the above equations and return the relative motion and the transient response for an harmonic oscillator driven by test ground motion
# 
# ```

# In[93]:


from theory import test_driven_oscillator

help(test_driven_oscillator)


# ```{div} full-width
# 
# Here below, we plot the test ground and relative motions (left column) and the only steady-state and transient motions (right column)
# ```

# In[94]:


del fig
plt.ioff()


# In[95]:


relative_motions, transient_motions = test_driven_oscillator(ts, Tg)

fig,axes = plt.subplots(3,tight_layout=True,figsize=(10,6),sharex=True,sharey="row")
axes[-1].set_xlabel("Time [s]")
for ax, ground_motion, relative_motion, transient_motion, ylabel in zip(axes, ground_motions, relative_motions, transient_motions, ylabels):
    
    ax.plot(ts,ground_motion,label="ground")
    ax.plot(ts,relative_motion,label="relative")
    ax.plot(ts,transient_motion,label="transient",linestyle="dashed")
    ax.set_ylabel(ylabel)
        
for ax in axes.flatten():
    ax.axhline(0,**options)
    ax.legend()
    


# In[96]:


glue("fig_relative_test", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_relative_test
# :name: fig-relative-test
# 
# Ground, relative and transient accelerations (blue, orange and green lines) for the driven harmonic oscillator with $T_g=1$.
# ```
# ````

# In[ ]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# Let us now change the ground period to $T_g=2$ s and make a similar plot to that shown in  {numref}`Figure {number} <fig-relative-test>`
# ```

# In[97]:


del fig
plt.ioff()


# In[102]:


Tg = 2
ts = np.linspace(-Tg,6*Tg,10001)
ground_motions = eva_test_ground_motion(ts, Tg)
relative_motions, transient_motions = test_driven_oscillator(ts, Tg)

fig,axes = plt.subplots(3,tight_layout=True,figsize=(10,6),sharex=True,sharey="row")
axes[-1].set_xlabel("Time [s]")
for ax, ground_motion, relative_motion, transient_motion, ylabel in zip(axes, ground_motions, relative_motions, transient_motions, ylabels):
    
    ax.plot(ts,ground_motion,label="ground")
    ax.plot(ts,relative_motion,label="relative")
    ax.plot(ts,transient_motion,label="transient",linestyle="dashed")
    ax.set_ylabel(ylabel)
        
for ax in axes.flatten():
    ax.axhline(0,**options)
    ax.legend()
    


# In[103]:


glue("fig_relative_test_2", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_relative_test_2
# :name: fig-relative-test-2
# 
# Ground, relative and transient accelerations (blue, orange and green lines) for the driven harmonic oscillator with $T_g=2$ s.
# ```
# ````

# In[116]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Dislocation
# 
# 
# ```{div} full-width
# 
# By considering that the ground dislacement from P and S waves is proportional to the time derivative of the dislocation $\delta u$, we can obtained it by integrating the test ground displacement 
# 
# $$
# \delta u(t) = \int_{-\infty}^t u_g(\tau)\,d \tau = \delta u_0\,\big[\tfrac{1}{12\,\pi}\,\sin(2\,\omega_g\,t)-\tfrac{2}{3\,\pi}\,\sin(\omega_g\,t) + \frac{t}{T_g}\big]\,B_{g}(t) + \delta u_0\,H(t-T_g)
# $$  (H0:66)
# 
# that we show here in the case of the ground period set to $T_g=3$ s
# 
# ```

# In[117]:


del fig
plt.ioff()


# In[118]:


Tg = 3
ts = np.linspace(-Tg/2,1.5*Tg,10001)

wg = 2*np.pi / Tg
dislocation = ( np.sin(2*wg*ts)/4 - 2*np.sin(wg*ts) )/(3*np.pi) + ts/Tg
dislocation[ts<0] = 0
dislocation[ts>Tg] = 1

fig,ax = plt.subplots(figsize=(10,2))
ax.plot(ts,dislocation)
ax.axhline(0,**options)
ax.axhline(1,**options)
ax.axvspan(0,Tg,color="gray",alpha=0.2)
ax.set_xlabel("Time [s]");


# In[119]:


glue("fig_dislocation", fig)


# ````{div} full-width
# 
# ```{glue:figure} fig_dislocation
# :name: fig-dislocation
# 
# Time evolution of the dislocation over the fault.
# ```
# ````

# In[120]:


plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# <p style="page-break-after:always;"></p>
