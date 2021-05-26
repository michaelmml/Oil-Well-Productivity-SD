# Oil & Gas Productivity Analysis - South Dakota
Analysis of oil &amp; gas productivity from South Dakota Department of Natural Resources through well-by-well production time series.

## Table of Contents
* [Executive Summary](#executive-summary)
* [Production Analysis](#productivity-analysis)
* [Monte Carlo Markov Chain](#monte-carlo-markov-chain)
* [Distribution of Well Productivity](#distribution-of-well-productivity)

## Executive Summary
South Dakota Department of Agriculture and Natural Resources provides highly detailed information on well-by-well production of oil and gas over time; including also with enhanced production units (EPU) or not. The main criterias of interest are as follows: County / Well Name / Reported Well Status / Date Production / Production Days.

The timeframe and days of production provides a useful way to analyse productivity of each well over time, which is the main aim of this exercise. As this is an academic exercise to analyse general productivity, oil and gas production metrics have been combined into barrels of oil equivalent through the assumption 1 BoE = 5,800 cf. Furthermore, whilst practically, the wells will likely undergo maintenance between production days blocks, this data is not available and therefore omitted.

![Overallproduction](https://user-images.githubusercontent.com/84533632/119716306-e5680400-be5c-11eb-807b-76ae6a8d2beb.png)

![wellproduction](https://user-images.githubusercontent.com/84533632/119737412-266d1200-be77-11eb-897f-126c9f2bb922.png)

Source: https://denr.sd.gov/des/og/producti.aspx


## Production Analysis
The bulk of the exploratory data analysis has been to set out the cumulative production of each well and identifying the days taken to reach the tresholds of one-quarter, one-half and three-quarters. See the attached csv file distilling the data on a per-well basis with characteristics such as total production and number of days to reach quarter / half / three-quarter of cumulative production.

```
allprodquarterindex = allprodrate[['Well_Name', 'allprodquarter']].apply(lambda x: abs(data.loc[(data['Well_Name'] ==
                        x['Well_Name']), 'Cumsum_OilProd'] - x['allprodquarter']).idxmin(), axis=1)
allprodhalfindex = allprodrate[['Well_Name', 'allprodhalf']].apply(lambda x: abs(data.loc[(data['Well_Name'] ==
                        x['Well_Name']), 'Cumsum_OilProd'] - x['allprodhalf']).idxmin(), axis=1)
                        
allprodquarterdays = allprodquarterindex.apply(lambda x: data['Cumsum_Days'].loc[x])
allprodhalfdays = allprodhalfindex.apply(lambda x: data['Cumsum_Days'].loc[x])
```

With the long tail of production from a well towards the end of its life, productivity has been defined with the fraction: Three-quarter of total cumulative production / Number of days to reach threshold. Wells with productivity of less than 1 barrels per day have also been removed.


## Monte Carlo Markov Chain
The distribution of well productivity seems to have a Gamma Distribution and so the Metropolis Hastings algorithm is built to analyse this. The MH algorithm as a way to approximate the distribution of a sample is built on the properties of Markov Chains. Here, in the case of a sample with distribution approximate to the Gamma Distribution, we consider the shape parameter and scale parameter with respect to the sample, using a transition matrix to generate the next approximation - hence the Markov property. Using a proposal distribution and the acceptance algorithm of MH, we decide whether to accept the next set of parameters or to keep the same. And so on...

In this instance when the algorithm is run, the final value for the parameters are shape parameter = 1.17702788 and scale parameter = 45.2135939. Taking the average of the last 50% of accepted results gives shape parameter = 1.14055234 and scale parameter = 45.44502724. The reason is that we start from a random set of parameters which the algorithm move from towards the optimal parameter space.

![MCMC_algorithm](https://user-images.githubusercontent.com/84533632/119716490-15afa280-be5d-11eb-85cc-f78b9b37dec9.png)

## Distribution of Well Productivity

```
# Essentially, acceptance probability = min(1, dist(y)proposal(x|y)/dist(x)proposal(y|x))
# = min(1, exp(log(dist(y)proposal(x|y)/dist(x)proposal(y|x))
# = min(1, exp(   log(dist(y)) + log(proposal(x|y)) - log(dist(x)) + log(proposal(y|x))   )
# as given in MH function and Acceptance function
def metropolis_hastings(p, q, transition_dist, initial_parameters, iterations, data, accept_rule):
    x = initial_parameters
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_dist(x)
        if accept_rule(p(x, data) + np.log(q(x)), p(x_new, data) + np.log(q(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)
```

As an illustration, from the average of the accepted parameters, the gamma distribution is then used to generate obversations to layer on top of the observed data.

![MCMC_normal](https://user-images.githubusercontent.com/84533632/119716522-1cd6b080-be5d-11eb-9125-97946125e8e1.png)

Out of interest, the plot of data without including Enhanced Recovery Units is as follows:


