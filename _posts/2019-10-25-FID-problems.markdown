---
layout: post
mathjax: true
title:  "FID problems"
---
# Problem of Frechet Inception Distance(FID) score for evaluation of GANs

Frechet Inception Distance (FID) score is measure for the quality of GAN's generated samples. Given a set of generated samples and a set of real images, we first extract the 2048-dimensional activations from the pool3 layer of [Inception-v3](https://tfhub.dev/google/imagenet/inception_v3/classification/4) model; Then use a multi-variate Gaussian distribution to model both sets of activations. Denote $\mu_g, \Sigma_g$ and $\mu_r, \Sigma_r$ are the meas and covariances of generated images and real images. FID is calculated as:
$$
FID = ||\mu_x - \mu_g||_2^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x\Sigma_g)^\frac{1}{2})
$$
Besides relying on the pretrained Inception-v3 model, the biggest problem of FID is using single-modal multi-variate Gaussian to model the activations, which could be complex and multi-modal.

It is very easy to construct an example of FID's failure. Consider the following three $2d$ distributions:

Distribution A, with 4 modes (each a $2d$ Gaussian):

```python
mu_list = [[-1, 1], [1, 1], [-1, -1], [1, -1]] # means of 4 modes
cov = [[0.1, 0.0], [0.0, 0.1]]  # all have the same covariance.
```

Distribution B, with 4 modes (each a $2d$ Gaussian):

```python
sqrt_2 = np.sqrt(2)
mu_list = [[-sqrt_2, 0], [sqrt_2, 0], [0, sqrt_2], [0, -sqrt_2]] # means of 4 modes
cov = [[0.1, 0.0], [0.0, 0.1]]  # all have the same covariance.
```

Distribution C, with only 1 mode (also $2d$ Gaussian):

```python
mean = [0, 0]
cov = [[1, 0.0], [0.0, 1]]
```


![]({{ '/images/fid_dists.png' | absolute_url }})

If we try to capture these three *different* distributions with one single $2d$ Gaussian, they all have the same $\mu, \Sigma$:

```python
mean = [0, 0]
cov = [[1, 0.0], [0.0, 1]]
```

which implies that $FID(A, B) = FID(A, C) = FID(B, C) = 0$. 

