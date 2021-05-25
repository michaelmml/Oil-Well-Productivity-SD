# Oil & Gas Productivity Analysis - South Dakota
Analysis of oil &amp; gas productivity from South Dakota Department of Natural Resources through well-by-well production time series.

## Table of Contents
* [Executive Summary](#executive-summary)
* [Production Analysis](#productivity-analysis)
* [Monte Carlo Markov Chain](#monte-carlo-markov-chain)
* [Distribution of Well Productivity](#distribution-of-well-productivity)

## Executive Summary
South Dakota Department of Agriculture and Natural Resources provides highly detailed information on well-by-well production of oil and gas over time; including also with enhanced production units (EPU) or not. The timeframe and days of production provides a useful way to analyse productivity of each well over time, which is the main aim of this.
![Overallproduction](https://user-images.githubusercontent.com/84533632/119579185-cd897500-bdb5-11eb-8d26-1afa7e826092.png)

Source: https://denr.sd.gov/des/og/producti.aspx

Oil and gas production metrics have been combined into barrels of oil equivalent through the assumption 1 BoE = 5,800 mcf.

## Production Analysis
The bulk of the explroatory data analysis has been to set out the cumulative production of each well and identifying the days taken to reach the tresholds of one-quarter, one-half and three-quarters. 

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

![MCMC_algorithm](https://user-images.githubusercontent.com/84533632/119579037-84d1bc00-bdb5-11eb-9fbb-28069cba4682.png)

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
![MCMC](https://user-images.githubusercontent.com/84533632/119273914-059d8600-bc05-11eb-989f-5df8fd1a0454.png)

Out of interest, the plot of data without including Enhanced Recovery Units is as follows:

![MCMC_normal](https://user-images.githubusercontent.com/84533632/119567384-2b14c600-bda4-11eb-9422-99812886ce0f.png)

