#!/usr/bin/env python
# coding: utf-8

# (chap-dirac-delta)=
# # Dirac delta function
# 
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 

# ## Definition

# ```{div} full-width
# 
# The Dirac delta can be loosely thought of as a function on the real line which is zero everywhere except at the origin, where it is infinite,
# 
# $$\begin{align}
# \delta(x) = \begin{cases} \infty & x=0 \\ 0 & x\neq 0 \end{cases}
# \end{align}$$ (A0:1)
# 
# and which is also constrained to satisfy the identity
# 
# $$\begin{align}
# \int_{-\infty}^\infty \delta(x)\,\mathrm{d}x = 1
# \end{align}$$ (A0:2)
# 
# A better characterization consists in consdering the Dirac delta function as a generalized function. In the theory of distributions, a generalized function is considered not a function in itself but only about how it affects other functions when "integrated" against them. In keeping with this philosophy, to define the delta function properly, it is enough to say what the "integral" of the delta function is against a sufficiently "good" test function $f$
# 
# $$\begin{align}
# \int_{\mathcal{I}} f(x)\,\delta(x)\,\mathrm{d}x = 
# \begin{cases} f(0) & 0\in\mathcal{I} \\ 
# 0 & 0\notin\mathcal{I}
# \end{cases}
# \end{align}$$ (A0:3)
# 
# where $\mathcal{I}$ is an open interval of the real axis. In this respect, the Dirac delta can be defined as the distributional derivative of the Heaviside step function
# 
# $$
# \begin{align}
# \delta(x) = \dot{H}(x) 
# \end{align}
# $$  (A0:4)
# 
# where the dot $\cdot$ stands for the first-order derivative and $H$ is the Heaviside step function
# 
# $$
# \begin{align}
# H(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x<0 \end{cases}
# \end{align} 
# $$ (A0:5)
# 
# ```
# 
# ```{admonition} Proof of 
# :class: note, full-width
# 
# Let us consider the integration by parts of two function $f(x)$ and $g(x)$
# 
# $$\begin{align}
# \int_a^b f(x)\,\dot{g}(x)\,\mathrm{d}x = \left. f(x)\,g(x)\,\right|_a^b - \int_a^b \dot{f}(x)\,g(x)\,\mathrm{d}x
# \end{align}$$ (A0:6)
# 
# and apply it to the case in which $g(x)=H(x)$ and $a<0<b$
# 
# $$\begin{align}
# \int_a^b f(x)\,\delta(x)\,\mathrm{d}x &= \left.f(x)\,H(x)\,\right|_a^b - \int_a^b \dot{f}(x)\,H(x)\,\mathrm{d}x \\
# &= f(b) - \int_0^b \dot{f}(x)\,\mathrm{d}x = f(b) - \left. f(x)\,\right|_0^b = f(0)
# \end{align}$$ (A0:7)
# 
# 

# ## Dirac delta derivative

# ```{div} full-width
# 
# The Dirac delta derivative is defined as follows
# 
# $$\begin{align}
# \int_a^b f(x)\,\delta'(x)\,\mathrm{d}x &= \left. f(x)\,\delta(x)\,\right|_a^b - \int_a^b \dot{f}(x)\,\delta(x)\,\mathrm{d}x =  -\dot{f}(0)
# \end{align}$$ (A0:8)
# 
# For higher-order derivatives one obtians
# 
# $$\begin{align}
# \int_a^b f(x)\,\delta^{(n)}(x)\,\mathrm{d}x &= (-1)^n\,f^{(n)}(0)
# \end{align}$$ (A0:9)
# 
# ```

# ## Basic propetries 
# 
# ```{div} full-width
# 
# The Dirac delta is symmetric
# 
# $$\begin{align}
# \delta(-x) = \delta(x)
# \end{align}$$ (A0:10)
# 
# and the product of the Dirac delta with $x$ is equal to zero
# 
# $$\begin{align}
# x\,\delta(x) = 0
# \end{align}$$ (A0:11)
# 
# The Dirac delta derivative satisfies a number of basic properties
# 
# $$\begin{align}
# & \dot{\delta}(-x) = -\dot{\delta}(x) \\
# & x\,\dot{\delta}(x) = -\delta(x)
# \end{align}$$ (A0:12)
# 
# ```

# ## Three-dimensional Dirac delta

# ```{div} full-width
# 
# The three-dimensional Dirac delta can be defined in the Cartesian reference frame as 
# 
# $$
# \delta(\mathbf{x}) = \delta(x_1)\,\delta(x_2)\,\delta(x_3)
# $$ (A0:13)
# 
# where $x_j$ are the Cartesian coordinates.
# 
# Similar to the one-dimensional case, eq. {eq}`A0:3`, it satify the following identity
# 
# $$\begin{align}
# \int_{\mathcal{V}} f(\mathbf{x})\,\delta(\mathbf{x})\,\mathrm{d}V = 
# \begin{cases} f(\mathbf{0}) & \mathbf{0}\in\mathcal{V} \\ 
# 0 & \mathbf{0}\notin\mathcal{V}
# \end{cases}
# \end{align}$$ (A0:14)
# 
# where $\mathcal{V}\subseteq\mathbb{R}^3$ is a open region of the three-dimensional space and $dV=dx_1\,dx_2\,dx_3$ is the infinitesimal volume element.
# 
# ```
# 
# ```{note}
# :class: full-width
# It can be shown that the Laplacian of the inverse of the radial distance from the origin is proportional to the three-dimensional Dirac delta
# 
# $$
# \nabla^2 \frac{1}{r} = -4\,\pi\,\delta(\mathbf{x})
# $$ (A0:15)
# 
# with
# 
# $$
# r = \sqrt{\mathbf{x}\cdot\mathbf{x}}
# $$ (A0:16)
# 
# ```

# <p style="page-break-after:always;"></p>
