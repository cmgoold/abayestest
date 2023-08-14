{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0, upper=1> y;
{% endblock %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ bernoulli_logit(mu_j);
{% endblock likelihood %}
{% endblock %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector<lower=0, upper=1>[N] mu_prob_j = inv_logit(mu_j);
  vector<lower=0, upper=1>[2] mu_prob = inv_logit(mu);
  real mu_prob_diff = mu_prob[1] - mu_prob[2];
  array[N] int<lower=0, upper=1> y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(n in 1:N)
    y_rep[n] = bernoulli_logit_rng(mu_j[n]);
{% endblock %}
{% endblock %}
  
