{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  vector[N] y;
{% endblock %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ normal(mu_j, sigma_j);
{% endblock likelihood %}
{% endblock %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector[N] y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(n in 1:N)
    y_rep[n] = normal_rng(mu_j[n], sigma_j[n]);
{% endblock %}
{% endblock %}
  
  
  
