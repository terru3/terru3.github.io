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

To fully understand a data science or machine learning concept, you need to be exposed to
all three perspectives—the idea, the math, and the code.
<!--more-->

I've skimmed through a number of papers, took some classes, Googled my fair share of formulas, 
but when the concept actually comes up in discussion or in practice, I often find myself missing core chunks
of knowledge to actully implement and tweak it to my use case. I've had to go back multiple times to read the notes,
but more importantly to fill in the gaps of how the underlying math functions and how the code should be structured.

Therefore, I believe to truly grasp concepts, we need to view it from all three angles—what it accomplishes (idea),
why it works (math), and how it is developed in practice (code). Of course, depending on the content, these components will differ in
complexity. Moreover, it's best to alternate in waves——begin with the overarching idea, math, and code, then loop back as you build up the additional features. I'll walk through a few examples of what this means, which will hopefully illuminate this formulation.

Self-attention
======

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

How did we know the first "bar" *(bar1)* was a verb meaning to prohibit and the second "bar" *(bar2)* was a noun referring to an establishment serving alcohol? We read the surrounding words, saw that *bar1* preceded a noun, and saw *bar* preceded by the verb "enter"...you get the point. At this point, self-attention is ready to burst onto the scene, because it exactly represents a proposal to emulate this process.

  **Mathematically:**

If we want our tokens to communicate with each other (in other words, perform some sort of mixing operation between their embeddings), the naive answer is to simply average them all into a single vector, which surely contains information about all the tokens and how they relate to each other...well, kind of.  

The glaring issue with this approach is that it's too aggressive. Each sentence now just consists of a single numeric vector repeated across every token—we want each specific token to have a unique representation depending on how it relates to the others.

Approach 2: Dot products! As we know, the dot product of vectors $a$ and $b$ is defined as: $a⋅b = |a||b|cos(θ)$. The higher the dot product, the closer in angle...or larger magnitude. We don't want the magnitude of the pretty much arbitrary embeddings messing with our notion of importance, so we'll be sure to scale the dot products by some fixed dimensionality (more on that later!) With this, then, we are able to capture a rough snapshot of say, given token x3, how similar and thus important x1, x2, x4, x5, etc. are to its meaning.

  **Code:** We'll skip this for now.

Q, K, V?
------

In progress!

  **Conceptually:** Alas, we've run into another problem of sorts. If we're just dotting x with itself, then each token $x_i$ will always be most similar to itself, drowning out the actually useful similarity outputs being reported with $x_i⋅x_{i-1}$, $x_i⋅x_0$, etc. Conceptually, failing to scale our dot product will also cause a similar issue. To break this symmetry, we introduce learnable *projections* of x—in other words, matrices we call the *query (Q)*, *key (K)*, and *value (V)*, that will actually allow the model to incorporate contextual information surrounding $x_i$.

  


  Note: In multi-head attention (MHA), this process is broken down and parallelized _______. We won't worry about the details in implementing this.
 
  **Mathematically:**

  Explain QK scale factor $\sqrt{d_k}$. Assuming $Q, K$ are independent with mean 0 and variance 1, then

  $$ w = q_i ⋅ k_j^\top $$
  
  **Code:**

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
  
Other attentions (sliding window, block-sparse, attention sinks)
------
  
MQA/GQA
------
  
Quantization
======

Model.half()?
------
  
Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)
------











