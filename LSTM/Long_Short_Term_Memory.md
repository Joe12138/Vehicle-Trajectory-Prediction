# <center>Long Short-Term Memory</center>

## Abstract

Learning to store information over extended time intervals via recurrent backpropagation takes a very long time, mostly due to insufficient, decaying error back flow. We briefly review Hochreiter's 1991 analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called "Long Short-Term Memory" (LSTM). Truncating the gradient where this does not do harm (*在无害的情况下截断梯度*), LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing *constant* error flow through "constant error carrousels" (*LSTM里面的特殊单元，作用就是：learn to bridge minimal time lags in excess of 1000 discrete time steps*) within special units. Multiplicative fate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is $O(1)$.

## Introduction

Recurrent networks can in principle use their feedback connections to store representations of recent input events in form of activations ("short-term memory", as opposed to "long-term memory" embodied by slowly changing weights). This is potentially significant for many applications, including speech processing, non-Markovian control, and music composition. The most widely used algorithms for learning *what* to put in short-term memory, however, take too much timer or do not work well at all, especially when minimal time lags between inputs and corresponding teacher signals are long. Although theoretically fascinating, existing methods do not provide clear *practical* advantages over, say, backprop in feedforward nets with limited time windows.

**The problem.** With conventional "Back-Propagation Through Time" (BPTT) or "Real-Time Recurrent Learning" (RTRL), error signals "flowing backwards in time" tend to either (1) blow up or (2) vanish: the temporal evolution of the backpropagated error exponentially depends on the size of the weight. Case (1) may lead to oscillating weights, while in case (2) learning to bridge long time lags takes a prohibitive amount of time, or does not work at all.

**The remedy.** 强调LSTM的优点

## Constant Error Backprop

### Exponentially Decaying Error

**Conventional BPTT.** 

- Output unit $k$'s target at time $t$ is denoted by $d_k(t)$. 

- Using mean squared error, $k$'s error signal is $\mathcal{v}_{k}(t)=f^{'}_{k}(net_k(t))(d_k(t)-y^k(t))$
- $y^i(t)= f_i(net_i(t))$ is the activation of a non-input unit $i$ with differentiable activation function $f_i$
- $net_i(t)=\sum_{j}w_{i,j}y^{j}(t-1)$ is unit $i$'s current net input
- $w_{ij}$ is the weight on the connection from unit $j$ to $i$ 
- Some non-output unit $j$'s backpropagated error signal is $v_{j}(t)=f'_j(net_j(t))\sum_{i}w_{ij}v_{i}(t+1)$
- 

