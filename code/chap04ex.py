#%% [markdown]
# # Examples and Exercises from Think Stats, 2nd Edition
# 
# http://thinkstats2.com
# 
# Copyright 2016 Allen B. Downey
# 
# MIT License: https://opensource.org/licenses/MIT
# 

#%%
from __future__ import print_function, division

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import nsfg
import first
import thinkstats2
import thinkplot

#%% [markdown]
# ## Examples
# 
# One more time, I'll load the data from the NSFG.

#%%
live, firsts, others = first.MakeFrames()

#%% [markdown]
# And compute the distribution of birth weight for first babies and others.

#%%
first_wgt = firsts.totalwgt_lb
first_wgt_dropna = first_wgt.dropna()
print('Firsts', len(first_wgt), len(first_wgt_dropna))
 
other_wgt = others.totalwgt_lb
other_wgt_dropna = other_wgt.dropna()
print('Others', len(other_wgt), len(other_wgt_dropna))

first_pmf = thinkstats2.Pmf(first_wgt_dropna, label='first')
other_pmf = thinkstats2.Pmf(other_wgt_dropna, label='other')

#%% [markdown]
# We can plot the PMFs on the same scale, but it is hard to see if there is a difference.

#%%
width = 0.4 / 16

# plot PMFs of birth weights for first babies and others
thinkplot.PrePlot(2)
thinkplot.Hist(first_pmf, align='right', width=width)
thinkplot.Hist(other_pmf, align='left', width=width)
thinkplot.Config(xlabel='Weight (pounds)', ylabel='PMF')

#%% [markdown]
# `PercentileRank` computes the fraction of `scores` less than or equal to `your_score`.

#%%
def PercentileRank(scores, your_score):
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1

    percentile_rank = 100.0 * count / len(scores)
    return percentile_rank

#%% [markdown]
# If this is the list of scores.

#%%
t = [55, 66, 77, 88, 99]

#%% [markdown]
# And you got the 88, your percentile rank is 80.

#%%
PercentileRank(t, 88)

#%% [markdown]
# `Percentile` takes a percentile rank and computes the corresponding percentile. 

#%%
def Percentile(scores, percentile_rank):
    scores.sort()
    for score in scores:
        if PercentileRank(scores, score) >= percentile_rank:
            return score

#%% [markdown]
# The median is the 50th percentile, which is 77.

#%%
Percentile(t, 50)

#%% [markdown]
# Here's a more efficient way to compute percentiles.

#%%
def Percentile2(scores, percentile_rank):
    scores.sort()
    index = percentile_rank * (len(scores)-1) // 100
    return scores[index]

#%% [markdown]
# Let's hope we get the same answer.

#%%
Percentile2(t, 50)

#%% [markdown]
# The Cumulative Distribution Function (CDF) is almost the same as `PercentileRank`.  The only difference is that the result is 0-1 instead of 0-100.

#%%
def EvalCdf(sample, x):
    count = 0.0
    for value in sample:
        if value <= x:
            count += 1

    prob = count / len(sample)
    return prob

#%% [markdown]
# In this list

#%%
t = [1, 2, 2, 3, 5]

#%% [markdown]
# We can evaluate the CDF for various values:

#%%
EvalCdf(t, 0), EvalCdf(t, 1), EvalCdf(t, 2), EvalCdf(t, 3), EvalCdf(t, 4), EvalCdf(t, 5)

#%% [markdown]
# Here's an example using real data, the distribution of pregnancy length for live births.

#%%
cdf = thinkstats2.Cdf(live.prglngth, label='prglngth')
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel='Pregnancy length (weeks)', ylabel='CDF', loc='upper left')

#%% [markdown]
# `Cdf` provides `Prob`, which evaluates the CDF; that is, it computes the fraction of values less than or equal to the given value.  For example, 94% of pregnancy lengths are less than or equal to 41.

#%%
cdf.Prob(41)

#%% [markdown]
# `Value` evaluates the inverse CDF; given a fraction, it computes the corresponding value.  For example, the median is the value that corresponds to 0.5.

#%%
cdf.Value(0.5)

#%% [markdown]
# In general, CDFs are a good way to visualize distributions.  They are not as noisy as PMFs, and if you plot several CDFs on the same axes, any differences between them are apparent.

#%%
first_cdf = thinkstats2.Cdf(firsts.totalwgt_lb, label='first')
other_cdf = thinkstats2.Cdf(others.totalwgt_lb, label='other')

thinkplot.PrePlot(2)
thinkplot.Cdfs([first_cdf, other_cdf])
thinkplot.Config(xlabel='Weight (pounds)', ylabel='CDF')

#%% [markdown]
# In this example, we can see that first babies are slightly, but consistently, lighter than others.
# 
# We can use the CDF of birth weight to compute percentile-based statistics.

#%%
weights = live.totalwgt_lb
live_cdf = thinkstats2.Cdf(weights, label='live')

#%% [markdown]
# Again, the median is the 50th percentile.

#%%
median = live_cdf.Percentile(50)
median

#%% [markdown]
# The interquartile range is the interval from the 25th to 75th percentile.

#%%
iqr = (live_cdf.Percentile(25), live_cdf.Percentile(75))
iqr

#%% [markdown]
# We can use the CDF to look up the percentile rank of a particular value.  For example, my second daughter was 10.2 pounds at birth, which is near the 99th percentile.

#%%
live_cdf.PercentileRank(10.2)

#%% [markdown]
# If we draw a random sample from the observed weights and map each weigh to its percentile rank.

#%%
sample = np.random.choice(weights, 100, replace=True)
ranks = [live_cdf.PercentileRank(x) for x in sample]

#%% [markdown]
# The resulting list of ranks should be approximately uniform from 0-1.

#%%
rank_cdf = thinkstats2.Cdf(ranks)
thinkplot.Cdf(rank_cdf)
thinkplot.Config(xlabel='Percentile rank', ylabel='CDF')

#%% [markdown]
# That observation is the basis of `Cdf.Sample`, which generates a random sample from a Cdf.  Here's an example.

#%%
resample = live_cdf.Sample(1000)
thinkplot.Cdf(live_cdf)
thinkplot.Cdf(thinkstats2.Cdf(resample, label='resample'))
thinkplot.Config(xlabel='Birth weight (pounds)', ylabel='CDF')

#%% [markdown]
# This confirms that the random sample has the same distribution as the original data.
#%% [markdown]
# ## Exercises
#%% [markdown]
# **Exercise:** How much did you weigh at birth? If you don’t know, call your mother or someone else who knows. Using the NSFG data (all live births), compute the distribution of birth weights and use it to find your percentile rank. If you were a first baby, find your percentile rank in the distribution for first babies. Otherwise use the distribution for others. If you are in the 90th percentile or higher, call your mother back and apologize.

#%%
# Solution goes here


#%%
# Solution goes here

#%% [markdown]
# **Exercise:** The numbers generated by `numpy.random.random` are supposed to be uniform between 0 and 1; that is, every value in the range should have the same probability.
# 
# Generate 1000 numbers from `numpy.random.random` and plot their PMF.  What goes wrong?
# 
# Now plot the CDF. Is the distribution uniform?

#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here


#%%



