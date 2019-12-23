---
layout: post
mathjax: true
title: Summary of "Adversarial Examples Are a Natural Consequence of Test Error in Noise"
---

Adversarial examples pose a serious threat to machine learning models. However, they are not the only "illegal inputs" that pose a threat. Another type of inputs images with natural and common corruptions, e.g. additive Gaussian noises. 

So far, the researchers seem to work seperately, and keep them two lines of research. This paper, titled
 [Adversarial Examples Are a Natural Consequence of Test Error in Noise\[1\]](https://arxiv.org/abs/1901.10513)
  establishes the connection between these two kinds of robustness: adversarial robustness and corruption robustness.

## Adversarial and Corruption Robustness

Both of them are defined as functions of *error set* of a statistical classifier. We  denote $E$ the error set, which is the set of points in the input space on which the classifier makes incorrect predictions. 

### Adversarial Robustness

Let p represents the clean image distribution, and $d(x, E)$ denote the distance from $x$ to the nearest point in $E$. Then adversarial robustness $\mathcal{P}_{x\sim p}[d(x, E) < \epsilon]$, the probability that some random sample from $p$ is not within distance $\epsilon$ of any point in the error set $E$. 

### Corruption Robustness

Let $q$ represent the corrupted image distribution. Corruption robustness is $\mathcal{P}_{x\sim q}[x \notin E ]$. 



## Adversarial Robustness and Gaussian Noise Corruption Robustness are highly correlated

One big take-away is while adversarial training performs better, Gaussian data augmentation does improve adversarial robustness. And in the oppsite direction, adversarial training  helps model against noise corruptions. One thing to note is Gaussian noise is just one of the many corruptions in ImageNet-C dataset (15 in total).


The authors suggest that researchers should take adversarial robustness and corruption robustness into considertaion at the same time. Reasons are:

* Corruptions may expose failure modes of a model that we might miss. Adversatial training improves adversarial robustness, but degrades performance on the fog and contrast corruptions. 
* Measuring corruption robustness is significantly easier than measuring adversarial robustness. Computing adversarial robustness perfectly requires solving an NP-hard problem for all points in the test set. That's possibly why hundreds of adversarial defense papers published are successfully fooled later. Since correctly evaluate and report adversarial robustness is hard.
* Failed adversarial defense strategies also fail to improve robustness against Gausssian noise. So the claimed $L_p$ adversarial robustness improvement doesn't implies robustness of distribution shift due to various corruptions.

[1] Adversarial examples are a natural consequence of test error in noise, Ford, Nic and Gilmer, Justin and Carlini, Nicolas and Cubuk, Dogus, ICML 2019