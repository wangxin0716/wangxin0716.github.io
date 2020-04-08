---
layout: post
mathjax: true
title:  A Mutual Information Maximization Perspective of Language Representation Learning
---

# Introduction
Deep Learning in some sense is about building correlations between random variables. For *supervised* 
learning, we aim to train a model $f$ build the connection between input $x$ and label of interest $y$, i.e.
$f(x)=y$. For *unsupervised* or *self-supervised* representation learning, we aim to train a model $f$ 
to generate *good* representation $y=f(x)$ on input $x$. 

Starting from *[word2vec](https://en.wikipedia.org/wiki/Word2vec)*, a main line of NLP 
is to build universal language representations, from word-level to sentence-level to even document level, on 
simply free texts *without* labels. The [captioned paper](https://openreview.net/forum?id=Syx79eBKwr) gives 
a great summary about the recent years' progresses in NLP from a unified perspective - Mutual Information Maximization.

# Mutual Information Maximization Recap
Consider two random variables $X, Y$, the mutual information~(MI) is defined as:

$$
I(X, Y) = KL(p(X)p(Y)|| p(X, Y)). 
$$

MI measures the mutual dependence between the two variables. MI is generally intractable, so in practice we 
seek to maximize a lower bound of MI. One typical and simple one of the many lower bounds is 
InfoNCE~(noise contrastive estimation), which is similar to standard cross-entropy loss in its form:

$$
I(X, Y) \geq E_{p(X, Y)} \big [f_\theta(x, y) - \log \sum_{\tilde{y} \sim \mathbb{Y}} \exp f_\theta(x, \tilde{y}) \big].
$$

where $f$ is a function usually parameterized by neural networks, which takes a pair as input and outputs a 
scalar as *score*. 

We say $(x, y)$ is a positive pair where we want to build correlation, and $(x, \tilde{y})$ is a 
negative pair. The loss is quite intuitive and self-explained, we push up the positive score $f_\theta(x, y)$,
 while push down the negative score $f_\theta(x, \tilde{y})$. For training with the above loss, the key is to 
 construct positive and negative pairs from the dataset. Sometimes the construction is direct and obvious, but sometimes
 we need special designs.
  
  In the next section, we will discuss the constructions of language representation learning methods: 
  word2vec with skipgram, the popular and more powerful [BERT](https://arxiv.org/abs/1810.04805), 
  [XLNet](https://arxiv.org/abs/1906.08237?context=cs.LG) proposed recently, and InfoWord proposed in the
  captioned paper.

# Representation Learning by Maximizing MI

To facilitate the discussion on these methods, we define a general form for 
function $f$: $f_\theta(a, b) = g_\phi(b)^T \cdot g_\omega(a)$, where $\theta=\\{\phi, \omega \\}$ and $g$ is a encoder
that transforms an input to a vector. Our focus is 
how the constructions of pairs $a, b$ and the encoder $g$ vary across different methods.

#### Word2Vec by SkipGram
Word2vec  seems to be the starting point for learning word-level representations. Skipgram maximizes:

$$
E_{p(x^i, x^i_j)} [p(x^i_j | x_i)],
$$
 
where $x_i$ is a target word, or token, and $x^i_j$ is a context word of $x_i$. For example, for sentence "hello world", 
if we treat "hello" as the target word, we may treat the adjacent words of "hello" within window size, e.g. 2, as context
words. The assumption behind is clear that words co-occuring in same sentences are highly correlated. Thus we encourage
the probability $p(\text{context_word}| \text{target_word})$ to be higher.

#### BERT
The most important part we should pay attention is the masked language modeling. Ginen a sequence of tokens of length $T$,
$\mathbf{x_i} = \\{x_1, \ldots, x_T\\}$. BERT replaces $15\%$ of the tokens with (1) a mask symbol with probability 0.8; (2) a random
word with probability 0.1; (3) the original word. For each replaced token, we get a new sequence
 $\hat{\mathbf{x_i}} = \\{x_1, \ldots, \hat{x_i}, \ldots, x_T\\}$, $x_i$ is replaced here. We maximize 
 
 $$
 E_{p(x_i, \hat{\mathbf{x_i}})}  \big[ p(x_i | \hat{\mathbf{x_i}}) \big ],
 $$ 
 
 i.e. to predict the replaced token $x_i$ from the masked sequence $\hat{\mathbf{x_i}}$.  Denote $b$ the masked word, and
 $a$ the masked sequence. Then $g_\phi(b)$ is simply a lookup of embedding layer, and $g_\omega(a)$ is the $i$-th token
 of Transformer output hidden-state sequence. 
 
#### XLNet
 The big difference of XLNet from BERT is its fine-grained construction of pairs on one same sequence.
  Denote $z$ a permutation
 on sequence  $\mathbf{x_i} = \\{x_1, \ldots, x_T\\}$, XLNet maximizes:
 
 $$
 E_{p(x)} \Big [ E_{p(z)} \big[\sum^T_{t=1} \log p(x_t^z| x^z_{<t}) \big] \Big],
 $$ i.e. predicting the target word $x_t^z$ from context $x^z_{<t}$. This is performed on all possible sub-sequences of
 length $t$. 
 
 The encoder Transformer is also replaced by Transformer XL, but is not our focus here. The rest parts of XLNet are 
 pretty much the same as BERT.
 
#### InfoWord
The captioned paper also proposed a DIM loss, which consider the n-gram within the sentence as target, rather
than one simple target word in above losses.

The combined DIM loss and InfoNCE loss in masked language modeling above is used for training. This brings
improvements on some downstream tasks.

The following table give the details about the construction of pairs and encoders:
![]({{ '/images/MI_Perspective_NLP.png' | absolute_url }})
# Conclusion 
 
 This is the best of times; this is the worst of times. On one hand, starting from classic word2vec,
  we've seen impressive processes in language representation learning by using more powerful encoder and bigger volume 
  of training corpus. We easily achieve state-of-the-art results, even better than human performance,
   on many downstream tasks by fine-tuning on these
  powerful representations. On the other hand, we can also say that no fundamental progresses 
  have been made since word2vec.
  We can already see that the future progresses based on the same method has come close to its end. Yet we are still
  far away from even a slight of *language understanding*. 
  
 
 
 

