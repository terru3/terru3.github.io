---
title: 'Machine Unlearning and Data Privacy'
date: 2024-03-21
permalink: /posts/2024/03/machine-unlearning
excerpt_separator: <!--more-->
toc: true
tags:
  - machine learning
  - machine unlearning
  - data privacy
---

A question that naturally arises with machine learning is, how do we selectively erase information from a trained model?
<!--more-->

The most common setting in which this request might arise is in data privacy. Let's say we have an LLM that we've spenth months
and hundreds of thousands of dollars training, and now a few customers have requested that their data be removed from the resulting model.
How do we do this without retraining the model on all the remaining data? Enter **machine unlearning**...

TODO:
  -example applications
  -exact vs. approximate unlearning, ε-certified removal
  -notation
  -types of removal requests
  -attacks like MIA, model inversion, etc.
  -metrics like ZF, AIN, JS-divergence, re-learn time, epistemic uncertainty, etc.
  -existing methods: SISA, UNSIR, GKT, SSD and Fisher-based methods, incompetent teacher, JiT unlearning via Lipschitz regularization
  -personal thoughts / recent developments / underexplored areas: unlearning in LLMs, etc.
  -link of resources to check out / sources
