---
title: 'Learning DS / ML Concepts'
date: 2023-12-20
permalink: /posts/2023/12/learning-ds-ml
excerpt_separator: <!--more-->
toc: true
tags:
  - data science
  - machine learning
---

To fully understand a data science or machine learning concept, we need to be exposed to
all three perspectives—the idea, the math, and the code.
<!--more-->

I've read my fair share of papers, taken both lacklustre and intense classes, and encountered countless formulas,
but when a concept actually comes up in practice, I often find myself missing core chunks of knowledge to actully implement it for my use case, and instead have to spend unexpected, additional time filling in the gaps of how the underlying math functions and how the code should be structured.

Therefore, I believe to truly grasp concepts, we need to view it from all three angles—what it accomplishes (idea),
why it works (math), and how it is developed in practice (code). Of course, depending on the content, these components will differ in
complexity. Moreover, it's best to alternate in waves——begin with the overarching idea, math, and code, then loop back as you build up the additional features. I'll walk through a few examples of what this means, which will hopefully illuminate this formulation.

Self-attention
======
*Attention Is All You Need, Vaswani et al. 2017*  
*https://arxiv.org/abs/1706.03762*

For example, what is self-attention? 

  **Conceptually:** Allow inputs (i.e. tokens) to check each other out, and find out which of the other inputs are different to them and which are less important.  
  **Mathematically:** Dot product the input with itself (ignoring q, k, v at the moment), use the notion of similarity as a proxy for relative importance.  
  **Code:** `x @ x.transpose(-2,-1)`  

We've avoided introducing potentially new and confusing terminology such as "causal masking" and even "embedding", no `torch.tril(torch.ones(...))`, and stripped the concept down to its core.

Next, let's revisit concept, math, and code as we build our intuition and fill in the gaps.

Why attend to itself?
------

 **Conceptually:**
 
Machines don't understand text—they understand numbers. As this post assumes, you know that natural language is first tokenized using a discrete vocabulary and then represented via numeric vectors (e.g. "hello world!" -> tokens [5, 15, 4] -> embedding (with dim 3) [(0.531, -0.421, 0.964), (0.141, 0.632, -0.963), (0.005, 0.964, -0.853)]).

This immediately poses a problem. In the sentence: "He was told to bar him from ever entering the bar again", clearly "bar" refers to different entities; in fact, they are not even the same part of speech. If we want our model to understand the nuances of language, it cannot statically represent letters/words/anything-in-between as fixed vectors.

How did we know the first "bar" *(bar1)* was a verb meaning to prohibit and the second "bar" *(bar2)* was a noun referring to an establishment serving alcohol? We read the surrounding words, saw that *bar1* preceded a noun, and saw *bar* preceded by the verb "enter"...you get the point. At this point, self-attention is ready to make its appearance, because it exactly represents a proposal to emulate this process.

  **Mathematically:**

If we want our tokens to communicate with each other (in other words, perform some sort of mixing operation between their embeddings), the naive answer is to simply average them all into a single vector, which surely contains information about all the tokens and how they relate to each other...well, kind of.  

The glaring issue with this approach is that it's too aggressive. Each sentence now just consists of a single numeric vector repeated across every token—we want each specific token to have a unique representation depending on how it relates to the others.

Approach 2: Dot products! As we know, the dot product of vectors $a$ and $b$ is defined as: $a⋅b = &#124;a&#124;&#124;b&#124;cos(θ)$. The higher the dot product, the closer in angle...or larger magnitude. We don't want the magnitude of our embeddings messing with our notion of importance, so we'll be sure to scale the dot products by some fixed dimensionality (more on that later!) With this, then, we are able to capture a rough snapshot of say, given token x3, how similar and thus important x1, x2, x4, x5, etc. are to its meaning.

  **Code:** We'll skip this for now.

Q, K, V?
------

  **Conceptually:** Alas, we've run into another problem of sorts. If we're just dotting x with itself, then each token $x_i$ will always be most similar to itself, drowning out the actually useful similarity outputs being reported with $x_i⋅x_{i-1}$, $x_i⋅x_0$, etc. Conceptually, failing to scale our dot product will also cause a similar issue. To break this symmetry, we introduce learnable *projections* of x—in other words, matrices we call the *query (Q)*, *key (K)*, and *value (V)*, that will actually allow the model to incorporate contextual information surrounding $x_i$.

  Note: In multi-head attention (MHA), this process is decomposed, wherein we have multiple sets of $Q$, $K$, and $V$, and the results are simply concatenated and projected back to our output dimension, which enables trivial parallelization and thus training and inference speeds as well as allowing the model to, in a sense, learn different "representation subspaces". We won't worry about the details in implementing this.

  Great, so we're almost done. But in the case of autoregressive language generation (which is the setting we focus on here), we're given some prefix tokens $x_0, x_1,...,x_i$ and we want to predict the next token $x_{i+1}$. A super clever way of training such models very efficiently is via a *teacher forcing* setup, where we simply give our model a complete tokenized sentence as both the input and target, but *shift* the target sequence by one. What this accomplishes is that now when our model looks at token 1, its target is token 2, and when it sees tokens 1 and 2 together, its target is now token 3, and so on. In this way, we can train in parallel on entire sequences at once rather than computing outputs sequentially at each step.

  original sequence: $[x_1, x_2, x_3, x_4]$  
  input: $[x_1]$           || target: $[x_2]$  
  input: $[x_1, x_2]$      || target: $[x_3]$  
  input: $[x_1, x_2, x_3]$ || target: $[x_4]$

  Now, to do this, obviously, we can't allow the model to cheat and view tokens beyond the context that it's given.

**In progress!**
 
  **Mathematically:**

  In addition, one detail we haven't discussed is this scaling factor $\sqrt{d_k}$ after taking the dot product of $Q$ and $K$. To see what it's doing there, we need to see the statistical motivations. The key idea: We want unit variance throughout the model, because otherwise large outlier values during attention could cause softmax to drown out other signals and consequently also lead to vanishing gradients (which is evident from the exponential nature of softmax). But why $\sqrt{d_k}$ in particular?

  To derive this scaling factor, the authors assume $Q$ and $K$ are both $d_k$ x $d_k$ with each independent entry $\sim N(0, 1)$. Thus, when we take the dot product, for each entry of $Q⋅K$, we obtain a mean of 0 $(E[AB] = E[A] ⋅ E[B])$ but a variance of $d_k$, since we've taken $d_k$ elementwise multiplications, of which each product has variance 1. (To convince yourself of this mathematically, note that for independent $X$ and $Y$, $Var(XY) = Var(X)[E(Y)]^2 + Var(Y)[E(X)]^2 + Var(X)Var(Y)$.) Therefore, to standardize our new values, we simply divide by the standard deviation / square root of the variance.
  
  This assumption is mostly supported, especially given the prevalence of pre-LayerNorm in recent transformer models. Recall that LayerNorm estimates the mini-batch mean and variance statistics over the feature space and normalizes the data. The upshot: our input data to the attention block $x \sim N(0, 1)$, and moreover, assuming PyTorch's nn.Linear default configuration, the projection matrices (typically denoted $W_q$, $W_k$, etc.) use Kaiming uniform initialization such that variance is intentionally preserved in the forward pass.
  
  **Code:**

First, let's see how we form our causal mask. Mathematically speaking, it might appear to you that what we need is some sort of triangular mask, so that at each step/row $i$, the model can only see the first $i$ elements in that row.
```
### TODO
```
With that, we can build our actual attention class.
```
# in __init__():

self.n_embd = n_embd

self.query = nn.Linear(n_embd, n_embd, bias=False)
self.key = nn.Linear(n_embd, n_embd, bias=False)
self.value = nn.Linear(n_embd, n_embd, bias=False)
self.out = nn.Linear(n_embd, n_embd, bias=False)

# in forward(x, mask=None):

q = self.query(x)
k = self.key(x)
v = self.value(x)

wei = q @ k.transpose(-2, -1) / np.sqrt(self.n_embd)
if mask is not None:
  wei = wei.masked_fill(mask, float("-inf"))
wei = F.softmax(wei, dim=-1)
attn = wei @ v
out = self.out(attn)
return out
```

Note: There are many differing implementations of self-attention, some of which down-project their $q, k, v$, some of which accept separate $q, k, v$ as input rather than $x$, etc., this is simply one method.
  
Other attentions (sliding window, block-sparse)
------

  
MQA/GQA
------
*Fast Transformer Decoding: One Write-Head is All You Need, Noam Shazeer 2019*  
*https://arxiv.org/abs/1911.02150*  
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Ainslie et al. 2023*  
*https://arxiv.org/abs/2305.13245*

Finally, let's tackle a useful recent optimization—*multi-query attention (MQA)* and *grouped query attention (GQA)*.

  **Conceptually:** Motivation: Training an auto-regressive model is sunshine and rainbows because of its trivial parallelizability, but we can't really say the same about inference. Typically when we generate text from such a model (i.e. *autoregressive decoding*), we need to perform a full forward pass through the entire model to sample just one token, and then repeat to sample the second, and so on, which is exceedingly slow. One can imagine with large dimensionality, repeatedly loading the keys and values becomes extremely tedious and memory-intensive.

  Proposal: 
  
  **Mathematically:**
  **Code:**


Attention Sinks
======
*Efficient Streaming Language Models w/ Attention Sinks, Xiao et al. 2023*  
*https://arxiv.org/abs/2309.17453*


Quantization
======

Model.half()?
------
  
Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)
------

*Credits: _________*







