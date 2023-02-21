#!/usr/bin/env python
# coding: utf-8

# (chap-I)=
# # Inverse problem theory
# 
# 
# ```{contents} Sections
# :local:
# :depth: 1
# ```
# 
# ```{div} full-width
# 
# In the perspective of locating the origin of an earthquake using the first arrivals of the P and S waves, we first review some basic knowledges of the inverse problem theory by means of which we can exploit these observations for our purpose.
# 
# Most of the content here discussed is based on the book [Inverse Probelm Theory (2005) by A. Tarantola](https://www.ipgp.fr/~tarantola/Files/Professional/Books/InverseProblemTheory.pdf).
# 
# ```
# 

# ## Probability density and volume functions
# 
# 
# ```{div} full-width
# 
# Let us now review how we define the probability over a continuous $N$-dimensional space $\mathcal{Q}\subset \mathbb{R}^N$. 
# 
# In order to fix the ideas in a non trivial case, we consider a sphere of unitary radius and the longitude, $\phi$, and latitude $\theta$ as the coordinates of the points the  $q=(\phi,\theta)\in\mathcal{Q}=[0,2\,\pi]\times[-\pi,\pi]\subset\mathbb{R}^2$. 
# 
# The infinitesimal volume (or solid angle) is $d V = v_q(q)\,dq$, with $v_q(q)=\cos\theta$ being the volume density, and we can evaluate the volume of a subset $\mathcal{A}\subset\mathcal{Q}$ as follows
# 
# $$
# V(\mathcal{A}) = \int_\mathcal{A}\,dV = \int_\mathcal{A} v_q(q)\,dq 
# $$ (LB:1)
# 
# We note that, in the present example, the total volume (or solid angle) is $V(\mathcal{Q}) = 4\,\pi $.
# 
# Within this framework, we can obtain the probability $P(\mathcal{A})$ for any subset $\mathcal{A}\subset\mathcal{Q}$ defining the probability density function (PDF) $p_q(q)$ or the probability volume function (PVF) $h_q(q)$ as follows
# 
# $$
# P(\mathcal{A}) = \int_A p_q(q)\,dq = \int_A h_q(q)\,v_q(q)\,dq 
# $$ (LB:3)
# 
# The PDF and PVF, by definition, are always linked by the volume density as follows
# 
# $$
# p_q(q) = h_q(q)\,v_q(q)
# $$ (LB:4)
# 
# ```
# 
# 
# ```{admonition} Homogeneous probability and the likelihood
# :class: note, full-width
# 
# Let us define the homogeneous probability over the sphere. The most natural definition which is not affected by the specific coordinate system adopted consists in defining
# 
# $$
# P(\mathcal{A}) = \frac{V(\mathcal{A})}{V(\mathcal{Q})}
# $$ (LB:5)
# 
# for all $\mathcal{A}\subseteq\mathcal{Q}$. The PDF and PVF associated to this homogeneous probability are
# 
# $$
# \begin{align}
# & p_q(q) = \frac{v_q(q)}{V(\mathcal{Q})} = \frac{1}{4\,\pi}\,\cos\theta \\
# & h_q(q) =  \frac{1}{V(\mathcal{Q})} = \frac{1}{4\,\pi}
# \end{align}
# $$ (LB:6)
# 
# From this simple example, we can understand that if we want to find the most likelihood point $q^*$,we have to maximize the PVF. In the present case, indeed, we do not find any maxima and, so, we conclude that all the points are equally likely. In contrast, if we had maximazed the PDF, we would have found that the equatorial points, $q=(\phi,0)$ for all $\phi\in[0,2\pi]$, are more likely and, so, we would have reached a wrong conclusion.
# 
# ```
# 
# ```{admonition} The Jacobian rule
# :class: note, full-width
# 
# Let us consider the change of coordinates $q=f(y)$ with $f:\mathcal{Y}\rightarrow\mathcal{Q}$ and write
# 
# $$
# P(\mathcal{A}) = \int_\mathcal{A} p_q(q)\,dq = \int_{f(\mathcal{A})} p_q(f(y))\,J(y)\,dy
# $$ (LB:7)
# 
# where $J$ is the Jacobian of the change of coordinates, i.e., the absolute value of the determinant of the matrix $F$ of the partial derivaties of $f$ with respect to the new variable $y$
# 
# $$
# J(y) = |\det(F)|
# $$ (LB:8)
# 
# $$
# F  = \left(\begin{matrix} \frac{\partial f_1}{\partial y_1} & \cdots & \frac{\partial f_1}{\partial y_N} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_N}{\partial y_1} & \cdots & \frac{\partial f_N}{\partial y_N} \\ \end{matrix}\right)
# $$ (LB:9)
# 
# In light of eq. {eq}`LB:7`, the PDF for the new coordinates $y$ reads
# 
# $$
# p_y(y) = p_q(f(y))\,J(y)
# $$ (LB:10)
# 
# We note also that, starting from the following identity for describing the same infinitesimal volume element in the two coordinate systems
#  
# $$
# d V = v_q(q)\,d q = v_y(y)\,d y
# $$ (LB:11)
# 
# we can also obtain the transformation rule for the volume density
# 
# $$
# v_y(y) = v_q(f(y))\,J(y)
# $$ (LB:12)
# 
# and learn that the PVF is invariant for change of coordinates
# 
# $$
# h_q(q) = \frac{p_q(q)}{v_q(q)} = \frac{p_y(y)}{v_y(y)} = h_y(y)
# $$ (LB:13)
# 
# This is a further reason for which we have to investigate the likelihood of the points $q\in\mathcal{Q}$ by studying the PVF, rather than the PDF.
# 
# ```
# 

# ## Observation equation
# 
# ```{div} full-width
# 
# First, let us consider the N-dimensional array $y\in\mathbb{R}^N$ collecting the observations (i.e., the arrival times) and the M-dimensional array $m\in\mathcal{M}\subseteq\mathbb{R}^M$ collecting the model parameters (i.e., the origin time and spatial coordinates). We note that the observation space is the whole N-dimensional space $\mathbb{R}^N$, while the model parameter space $\mathcal{M}\subseteq\mathbb{R}^M$ can be a subset of the M-dimensional space $\mathbb{R}^M$ in order to implement some physical constraints like the fact that the origin depth must be non-negative (i.e., the earthquake has to be occurred within the solid earth and not in the atmosphere).
# 
# Also, before proceeding with the discussion of probability distributions, we need to define volume densities for the observation and model spaces in order to distinguish between PDFs and PVs and we simply assume that they are constants
# 
# $$
# \begin{align}
# & v(y) \propto 1 \\
# & v(m) \propto 1
# \end{align}
# $$ (LA:1)
# 
# We then write the observation equation
# 
# $$
# y = f(m) + \varepsilon
# $$ (L:1)
# 
# where $f:\mathcal{M}\rightarrow\mathcal{Y}$ is the function expressing the theoretical relationship between the model parameters and the observations, and $\varepsilon\in\mathbb{R}^N$ is the N-dimensional array collecting the random errors. The random errors $\varepsilon=\varepsilon_\mathrm{obs}+\varepsilon_\mathrm{mod}$ consist of observational, $\varepsilon_\mathrm{obs}$, and modelling, $\varepsilon_\mathrm{mod}$, errors. The formers depend on the instrument of measure (i.e., on the seismometers and the seismologists that make the picking), while the latters come from the deficency in the theoretical modelling (i.e., from the adoption of a simplified velocity model which neglects, for instance, lateral heteorenegities of the P and S waves velocities of the crust and the lithospheric mantle).
# 
# ```
# 
# ### Observational, modelling and random errors
# 
# ```{div} full-width
# 
# For the sake of simplicity, we shall assume that observational and modelling errors are indepenendent and follows Gaussian distributions with zero mean and covariances $C_\mathrm{obs}$ and $C_\mathrm{mod}$ (which are $N\times N$-dimensional matrices), respectively.
# 
# From the assumption of independence and the rule of the error propagation, the random erros $\varepsilon=\varepsilon_\mathrm{obs}+\varepsilon_\mathrm{mod}$ also obey to a Gaussian distribution with zero mean and covariance $C$ given by the sum of the two covaraiance matrices $C_\mathrm{obs}$ and $C_\mathrm{mod}$. In this respect, the probability density function (PDF) for the random errors reads
# 
# 
# $$
# \begin{align}
# & p(\varepsilon) = \frac{1}{\sqrt{(2\,\pi)^N\,\mathrm{det}(C)}}\,\exp\left(-\frac{\varepsilon^\mathrm{T}\,C^{-1}\,\varepsilon}{2}\right) 
# \end{align}
# $$ (L:2)
# 
# where $\mathrm{det}(\cdot)$ stands for the determinant operator (i.e., $\mathrm{det}(C)$ is the determinant of $C$) and
# 
# $$
# C = C_\mathrm{obs}+  C_\mathrm{mod}
# $$ (L:3)
# 
# We note that the factor in front of the exponential in the right-hand side of eq. {eq}`L:2` is the normalization factors which guarantes
# 
# $$
# \int_{\mathbb{R}^N} p(\varepsilon)\,d\varepsilon = 1
# $$ (L:4)
# 
# The quantification of the modelling errors and of their covariance $C_\mathrm{mod}$ is a complex task that requires the comparison of the results from the simplified model with those from a more realistic model. A common and effective strategy for describing the modelling errors consists in assuming that their covariance matrix is simply propotional to that of the observational errors, $C_\mathrm{mod} \propto C_\mathrm{obs}$, and, so, eq. {eq}`L:3` can be rewritten as follows
# 
# $$
# C = \alpha\,C_\mathrm{obs}
# $$ (L:5)
# 
# where $\alpha\in\mathbb{R}^+$ is a non-negative factor that we shall consider as an hyper-parameter and estimate from the observations along with the model parameters $x$. Obviously, we expect that $\alpha\in(1,\infty)$ because the modelling errors cannot compensate the observational ones, being independent. The particular case in which $\alpha=1$, instead, implies that there are no modelling errors, while, if $\alpha \in[0,1)$, it would meaning that the adopted theoretical model $f(m)$ has too many degrees of fredom that allow to fit the observational errors as well.
# 
# ```
# 
# 
# ```{admonition} Volume density for the hyper-parameter space
# :class: note, full-width
# 
# 
# As it concerns the hyper-parameter $\alpha$, the most common way to introduce the volume density consists in the adoption of a logarithmic scale
# 
# $$
# \gamma = \ln \alpha
# $$ (L:8)
# 
# where $\gamma\in\mathbb{R}$ is the hyper-paremeter alternative to $\alpha$, and assume that its volume density is constant
# 
# $$
# v(\gamma) \propto 1
# $$ (LA:8)
# 
# This is due to the fact that (i) the hyper-parameters can span several orders of magnitude, (ii) the resulting PDF in the logarithmic scale is more symmetric and closely ressambles a Gaussian distribution. Also, (iii) the logarithmic hyper-parameter $\gamma$ does not require any positivity constraints.
# 
# According to the transformation rule of the volume density, that for the hyper-parameter $\alpha$ reads
# 
# $$
# v(\alpha) = v(\gamma)\,\left|\frac{\partial \gamma}{\partial \alpha}\right| \propto \alpha^{-1}
# $$ (LA:8A)
# 
# ```
# 
# ## Information from observations
# 
# ```{div} full-width
# 
# Within this framework, just because $\varepsilon = y-f(m)$, eq. {eq}`L:2` can be seen as the conditional PDF of the observations $y$ given the model parameters $m$ and the hyper-parameter $\alpha$
# 
# $$
# p(y|m,\alpha) = \frac{\alpha^{-N/2}}{\sqrt{(2\,\pi)^N\,\mathrm{det}(C_\mathrm{obs})}}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)
# $$ (L:6)
# 
# with $W(m)$ being the so called weighted residual square sum (WRSS)
# 
# $$
# W(m) = (y-f(m))^\mathrm{T}\,C_\mathrm{obs}^{-1}\,(y-f(m))
# $$ (L:7)
# 
# Here, we made use of $C^{-1}=\alpha^{-1}\,C_\mathrm{obs}^{-1}$ and $\mathrm{det}(C) = \alpha^{N}\,\mathrm{det}(C_\mathrm{obs})$. This conditional PDF describes the information from the observations $y$.
# 
# In other words, giving the model parameters $m$ and the hyper-paremeter $\alpha$, we set the mean of the Gaussian distribution and its covariance. In particular, observations closer and closer to the mean $f(m)$ will be more and more likely. We note also that, at this stage, we are considering all the possible observations, even if, in practice, we only have only one realization of the observations which is the set of observations made by the instrument of measure, say $y_\mathrm{obs}$. In this respect, in the following, we shall derive the conditional PDF $p(m,\alpha|y)$ for the model parameters $m$ and the hyper-parameter $\alpha$ given $y$ in order to estimate the model parameters and the hyper-parameters $\alpha$ given the actual observations $y=y_\mathrm{obs}$. This conditional PDF is the so called posterior PDF for the model parameters $m$ and the hyper-parameter $\alpha$.
# 
# ```
# 
# ## Prior and posterior informations
# 
# ```{div} full-width
# 
# In addition to the information from the observations which come from the conditional PDF $p(y|m,\alpha)$, we need to describe prior information on the model parameters $m$ and the hyper-parameter $\alpha$ and we do it with a prior PDF $p(m,\alpha)$. Even the case in which we have no prior information, we can set $p(m,\alpha)$ as the homogeneous PDF and write
# 
# $$
# p(m,\alpha) \propto v(m)\,v(\alpha) \propto \alpha^{-1}
# $$ (LC:2)
# 
# In this respect, we recall that we are assuming that the volume density for the model space is constant, eq. {eq}`LA:1`.
# 
# By combining prior information with the informatios from observations by multiplying the conditional PDF $p(y|m,\alpha)$ and the prior PDF $p(m,\alpha)$, we obtain the PDF describing the PDF both of the observations, the model parameters and the hyper-parameter
# 
# $$
# p(y,m,\alpha) = p(y|m,\alpha)\,p(m,\alpha)
# $$ (L:10)
# 
# Within this framework, the prior PDF $p(m,\alpha)$ can be seen as the marginal PDF of $p(y,m,\alpha)$ for $(m,\alpha)$
# 
# $$
# p(m,\alpha) = \int_{\mathbb{R}^N} p(y,m,\alpha)\,d y 
# $$ (L:11)
# 
# Indeed, by substituting eq. {eq}`L:10` into eq. {eq}`L:11` and making use of eq. {eq}`L:4`, we obtain
# 
# $$
# \int_{\mathbb{R}^N} p(y,m,\alpha)\,d y  = \int_{\mathbb{R}^N} p(y|m,\alpha)\,p(m,\alpha)\,d y  = p(m,\alpha)\, \int_{\mathbb{R}^N} p(y|m,\alpha)\,d y = p(m,\alpha)
# $$ (L:12)
# 
# 
# Let us now rewrite eq. {eq}`L:10` with the roles of $y$ and $(m,\alpha)$ reversed
# 
# $$
# p(y,m,\alpha) = p(m,\alpha|y)\,p(y)
# $$ (L:13)
# 
# Here, $p(m,\alpha|y)$ is the conditional PDF that we can identify as the posterior PDF for the model parameters $m$ and the hyper-parameter $\alpha$ given the observations $y$, and $p(y)$ is the marginal PDF of $p(y,m,\alpha)$ for $y$
# 
# $$
# p(y) = \int_{\mathbb{R}^+} \int_\mathcal{M} p(y,m,\alpha)\,d m\,d\alpha
# $$ (L:14)
# 
# By making use eqs {eq}`L:10` and {eq}`L:13`, we thus obtain the posterior PDF for $(m,\alpha)$ given $y$
# 
# $$
# p(m,\alpha|y) = \frac{p(y|m,\alpha)\,p(m,\alpha)}{p(y)}
# $$ (L:15)
# 
# that is the so called Bayes' theorem by means of which we can update the prior information on the model parameters $m$ and the hyper-parameter $\alpha$ with the information from the observations $y$. 
# 
# We note that the denominator $p(y)$ of the right-hand side of eq. {eq}`L:15` is needed only for normalizing the posterior PDF and, so, we can write
# 
# $$
# p(m,\alpha|y) \propto p(y|m,\alpha)\,p(m,\alpha)
# $$ (L:16)
# 
# In the end, combining eqs {eq}`L:6`, {eq}`LC:2` and {eq}`L:16`, we obatin
# 
# $$
# p(m,\alpha|y) = k\,\alpha^{-N/2-1}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)
# $$ (L:17)
# 
# where $k$ is a constant of normalization such that
# 
# $$
# \int_{\mathcal{M}}\int_{\mathbb{R}^+}p(m,\alpha|y)\,d\alpha\,d m = 1
# $$ (L:17a)
# 
# According to the Jacobian rule the PDF for $(m,\gamma)$ reads
# 
# $$
# p(m,\gamma|y) = p(m,\alpha|y)\,\left|\frac{\partial \gamma}{\partial \alpha}\right| = k\,\exp\left(-\frac{1}{2}\big(N\,\gamma + W(m)\,e^{-\gamma}\big)\right)
# $$ (L:17b)
# 
# where $\gamma$ is the logarithmic hyper-parameter, eq. {eq}`L:8` for which we are assuming that the volume density is constant, eq. {eq}`LA:8´.
# 
# ```
# 
# 

# ### Marginal and conditional PDFs and PVs for the model parameters and the logarithmic hyper-parameter $\gamma$
# 
# $$
# \int_{\mathbb{R}^+} \alpha^{-a}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)\,d \alpha = \left(\frac{W(m)}{2}\right)^{-a+1}\,\Gamma(a-1)
# $$
# 
# $$
# \int_{\mathbb{R}^+} \alpha^{-N/2}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)\,d \alpha = \left(\frac{W(m)}{2}\right)^{-(N-2)/2}\,\Gamma((N-2)/2)
# $$
# 
# $$
# \int_{\mathbb{R}^+} \alpha^{-N/2+1}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)\,d \alpha = \int_{\mathbb{R}^+} \alpha^{-(N-2)/2}\,\exp\left(-\frac{W(m)}{2\,\alpha}\right)\,d \alpha = \left(\frac{W(m)}{2}\right)^{-(N-4)/2}\,\Gamma((N-4)/2)
# $$
# 
# $$
# \bar{\alpha} = \frac{W(m)}{2}\,\frac{\Gamma((N-4)/2)}{\Gamma((N-2)/2)}
# $$
# 
# $$
# \int_{\mathbb{R}} \exp\left(-\frac{1}{2}\big(N\,\gamma + W(m)\,e^{-\gamma}\big)\right)\,d \gamma = \left(\frac{W(m)}{2}\right)^{-(N/2)}\,\Gamma(N/2)
# $$ 
# 
# $$
# \int_{\mathbb{R}} f(\gamma)\,d \gamma = F(\infty)-F(-\infty)
# $$ 
# 
# $$
# (\gamma\,F)' = \gamma \,f + F
# $$
# 
# $$
# \gamma\,F| = \int \gamma \,f + \int F
# $$
# 
# $$
# (\gamma^2\,F_N)' = 2\,\gamma \,F_N + \gamma^2\,f_N
# $$
# 
# 
# $$
# \int_{\mathbb{R}} \gamma\,f(\gamma)\,d \gamma = 
# $$ 
# 
# $$
# F_N = \exp\left(-\frac{1}{2}\big(N\,\gamma + W(m)\,e^{-\gamma}\big)\right)
# $$
# 
# $$
# f_N = F_N\,\left(-N/2+W/2\,e^{-\gamma}\right) = -N/2\,F_N + W/2\,F_{N+2}
# $$
# 
# $$
# \gamma^2\,F_N\big| - \int \gamma^2\,f_N = 2\,\gamma \,F_N 
# $$
# 
# 
# ```{div} full-width
# 
# By making use of the following identity
# 
# $$
# \int_{\mathbb{R}} \exp\left(-\frac{1}{2}\big(N\,\gamma + W(m)\,e^{-\gamma}\big)\right)\,d \gamma = \left(\frac{W(m)}{2}\right)^{-(N/2)}\,\Gamma(N/2)
# $$ (L:18)
# 
# we obtain the marginal PDF for the model parameters
# 
# 
# $$
# p(m|y) = \int_{\mathbb{R}^+} p(m,\alpha|y)\,d\alpha  = \int_{\mathbb{R}} p(m,\gamma|y)\,d\gamma = k\,\Gamma(N/2)\,\left(\frac{2}{W(m)}\right)^{N/2}
# $$ (L:19)
# 
# and the conditional PDF for $\gamma$
# 
# $$
# p(\gamma|m,y) = \frac{p(m,\gamma|y)}{p(m|y)} = \frac{1}{\Gamma(N/2)}\,\left(\frac{W(m)}{2}\right)^{(N/2)}\,\exp\left(-\frac{1}{2}\big(N\,\gamma + W(m)\,e^{-\gamma}\big)\right)
# $$ (L:20)
# 
# In view of the fact that the volume densities for the model and logarithmic hyper-parameter spaces are constants, the PVFs of the above marginal and conditional PDFs are simply proportional to them
# 
# $$
# \begin{align}
# h(m|y) &= \frac{p(m|y)}{v(m)} \propto p(m|y) \\
# h(\gamma|m,y) &= \frac{p(\gamma|m,y)}{v(\gamma)} \propto p(\gamma|m,y)
# \end{align}
# $$ (L:21)
# 
# ```
# 
# ## The study of the posterior PDF
# 
# ```{div} full-width
# 
# Let us first determine the most likelihood logarithmic hyper-parameter $\gamma$ given the model paramneters $m$ (as wel as the observations $y$) by maximizing the PVF $p(\gamma|m,y)$. We do it minimizing the following functional
# 
# $$
# L_\gamma(\gamma) = -2\,\ln h(\gamma|m,y) = N\,\gamma + W(m)\,e^{-\gamma}
# $$ (L:22)
# 
# By setting to zero the first derivative of $L_\gamma$
# 
# $$
# \frac{d L_\gamma (\gamma)}{d \gamma} = N-W(m)\,e^{-\gamma} = 0
# $$ (L:23)
# 
# it results that the most likelihood logarithmic hyper-parameter $\gamma$ given the model parameters $m$ is 
# 
# $$
# \gamma^*(m) = \ln\frac{W(m)}{N}
# $$ (L:24)
# 
# This also means that the most likelihood hyper-parameter $\alpha$ given the model parameters $m$ is 
# 
# $$
# \alpha^*(m) = \frac{W(m)}{N}
# $$ (L:25)
# 
# Let us now focus on the most likelhood model parameters $m$ by maximizing the PVF $h(m|y)$. Also in this case, we define the following functional
# 
# $$
# L_m(m) &= -2\,\ln h(m|y) = N\,\ln W(m) + c \\
# $$ (L:26)
# 
# and investigate it by considering its gradient $G_m$ and hessian $H_m$ 
# 
# 
# $$
# G_m(m) = \frac{\partial L_m(m)}{\partial m} = \frac{N}{W(m)}\,\frac{\partial W(m)}{\partial m}
# $$ (L:27)
# 
# $$
# H_m(m) = \frac{\partial^2 L_m(m)}{\partial m\,\partial m} = \frac{N}{W(m)}\,\frac{\partial^2 W(m)}{\partial m\,\partial m} - \frac{N}{W(m)^2}\,\frac{\partial W(m)}{\partial m}\,\frac{\partial W(m)}{\partial m} 
# $$ (L:28)
# 
# At the most likelihood model parameters $m^*$ which minimize the functional $L_m$
# 
# $$
# \begin{align}
# L_m(m^*) &= \min_{\forall\,m\in\mathcal{M}} L_m(m)
# \end{align}
# $$ (L:29)
# 
# the gradient yields zero
# 
# $$
# G_m(m^*) = 0
# $$ (L:30)
# 
# and the Hessian becomes
# 
# $$
# H_m(m^*) = \left.\frac{1}{\alpha^*(m)}\,\frac{\partial^2 W(m)}{\partial m\,\partial m}\right|_{m=m^*}
# $$ (L:31)
# 
# Let us now assume that the marginal PDF for the model parameters $m$ can be approximate by a Gaussian distribution with mean $m^*$ and covariance $C_m$
# 
# $$
# p(m|y) \approx \frac{1}{\sqrt{(2\,\pi)^M\,\det C_m}}\,\exp\left(-\frac{(m-m^*)^\mathrm{T}\,C_m^{-1}\,(m-m^*)}{2}\right)
# $$ (L:32)
# 
# and approximate the functional $L_m$ and its gradient and hessian as follows
# 
# $$
# \begin{align}
# & L_m(m) \approx (m-m^*)^\mathrm{T}\,C_m^{-1}\,(m-m^*) + c
# \\
# &G_m(m)  \approx 2\,C_m^{-1}\,(m-m^*)
# \\
# &H_m(m)  \approx 2\,C_m^{-1}
# \end{align}
# $$ (L:33)
# 
# This result shows how the gradient $G_m(m^*)=0$ is zero at the most likelihood model parameters $m^*$ and that estimate the uncertainity on the model parameters, i.e., their covariance, as two times the inverse of the Hessian evaluated at the most likelihood model parameters $m^*$
# 
# $$
# C_m = 2\,\left[H_m(m^*)\right]^{-1}
# $$ (L:34)
# 
# ```
# 

# <p style="page-break-after:always;"></p>
