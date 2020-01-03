---
layout: post
mathjax: true
title: "Likelihood is not a Good Measure to Evaluate Generative Models"
---

## Generative Models via Maximum Likelihood Estimation (MLE)

Consider we have a set of examples $ \begin{equation}\{ x_i \}_{i=1}^{n} \end{equation}$ drawn independently from the true but unknown data distribution
 $p_{data}(x)$. We have a model $p_{\theta}(x)$ parameterized with $ \theta $,
  and want to approximate $p_\theta(x)$ to $p_{data}(x)$.
    We could minimize the KL-divergence $D_{KL}(p_{data}(x)|| p_\theta(x))$, which gives:
$$
\begin{equation}
\begin{split}
D_{KL}(p_{data}(x)|| p_\theta(x)) &= \int p_{data}(x) \log \frac{p_{data}(x)}{p_\theta(x)} dx
&= -\mathbb{H}_{p_{data}(x)} - \mathbb{E}_{p_{data}(x)} [\log p_\theta(x)]
&= \text{const} -  \mathbb{E}_{p_{data}(x)} [\log p_\theta(x)]
\end{split}
\end{equation}
$$

where $\mathbb{H}_{p_{data}(x)}$ is the data entropy independent of parameters $\theta$.
  So minimizing $D_{KL}(p_{data}|| p_\theta)$ is equivalent to minimizing 
   $-\mathbb{E}_{p_{data}(x)} [\log p_\theta(x)]$, i.e. the cross entropy $\mathbb{H}(p_{data}, p_\theta)$.  

In practice, we maximize the emprical log-likelihood over $ \{x_i\}_{i=1}^n $, i.e. minimizing the average negative log-likelihood:

$$
\begin{equation}
\mathbb{E}_{p_{data}(x)} [\log p_\theta(x)] = \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i)
\end{equation}
$$

Among the deep generative models which usually work on complex high-dimensional data, e.g. natural images, flow-based generative models, *flows* in short, ar e the ones that exactly do MLE.  A flow is usually parameterized by a sequence of specialized neural network layers; each layer performs bijective trasnform on the input variables. The log-likelihood of a sample is evaluated by *change of variables* theorem. See [this post](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#change-of-variable-theorem) for more details of various types of flows.



## Likelihood can be *independent* of the quality of samples

A very simple but important argument for this part is:

**High log-likelihood is neither sufficient nor neccessary to the quality (or plausibility) of produced samples.** 

This counter-intuitive conclusion is first made clear in (this paper)[1] and tells that likelihood and samples' quality (or plausibility) are largely *independent*. More specifically, For a generative model that reports good likelihood on test set, it doesn't imply good quality of generated samples. And further, being able to generate plausible images doesn't take a model that must have competitive average likelihood. 

### Plausible Samples & Poor Likelihood

Consider the following mixture of gaussians as a generative model:

$$
p(x)=\frac{1}{n}\sum_{i=1}^n\mathcal{N}(x;x_i, \sigma^2 I)
$$

where $\{x_i\}_{i=1}^n$ are means of either training samples or other plausible samples. We  keep the variance $\sigma^2$ very small.  We can safely construct a sample $\tilde{x}=x_i + \epsilon$, where $\epsilon$ is an imperceptable noise (Gaussian or non-Gaussian). Then sample $\tilde{x}$ could be fairly plausible with poor likelihood.

### Good likelihood & Poor Samples
Say $p$ is a perfect density model for data $x$ achieving arbitrarily well log-likelihood, while $q$ is a bad model, e.g. Gaussian noise.
The following mixture model:

$$
0.01 p(x) + 0.99 q(x)
$$

So $99\%$ of the time, a sample from it would be plausibly poor.  However, the log-likelihood hardly change if dimension 
of data $d$ is large:
$$
\log (0.01 p(x) + 0.99 q(x)) \geq \log p(x) - \log 100
$$

$\log p(x)$ can be infinitely large proportionally to $d$.  While constant $\log 100 = 4.61$ is negligible, so that it 
won't pull down the whole log-likelihood to tell the sample is poor.


## Take-away
Theoretical analyses by giving extreme counter-examples in [1] show that *likelihood a generative model reports
 can be independent of samples' visual quality.*



[1] [A note on the evaluation of generative models](https://arxiv.org/pdf/1511.01844.pdf) by Theis, Lucas and Oord, Aaron van den and Bethge, Matthias

