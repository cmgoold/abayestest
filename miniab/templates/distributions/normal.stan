{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  vector[N] y;
{% endblock data %}

{% block parameters %}
  {{ super() }}
  vector<lower=0>[2] sigma;
{% endblock parameters %}

{% block transformed_parameters %}
  {{ super() }}
  vector<lower=0>[N] sigma_j = sigma[j];
{% endblock transformed_parameters %}

{% block model %}
{% block priors %}
  {{ super() }}
  sigma ~ {{ priors.sigma }};
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ normal(mu_j, sigma_j);
{% endblock likelihood %}
{% endblock model %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  real sigma_diff;
  vector[N] y_rep;
{% endblock declarations %}
{% block computations %}
  {{ super() }}
  sigma_diff = sigma[1] - sigma[2];
  for(n in 1:N)
    y_rep[n] = normal_rng(mu_j[n], sigma_j[n]);
{% endblock computations %}
{% endblock generated_quantities %}
  
  
  
