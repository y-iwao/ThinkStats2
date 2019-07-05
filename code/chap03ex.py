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
# Again, I'll load the NSFG pregnancy file and select live births:

#%%
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]

#%% [markdown]
# Here's the histogram of birth weights:

#%%
hist = thinkstats2.Hist(live.birthwgt_lb, label='birthwgt_lb')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='Birth weight (pounds)', ylabel='Count')

#%% [markdown]
# To normalize the disrtibution, we could divide through by the total count:

#%%
n = hist.Total()
pmf = hist.Copy()
for x, freq in hist.Items():
    pmf[x] = freq / n

#%% [markdown]
# The result is a Probability Mass Function (PMF).

#%%
thinkplot.Hist(pmf)
thinkplot.Config(xlabel='Birth weight (pounds)', ylabel='PMF')

#%% [markdown]
# More directly, we can create a Pmf object.

#%%
pmf = thinkstats2.Pmf([1, 2, 2, 3, 5])
pmf

#%% [markdown]
# `Pmf` provides `Prob`, which looks up a value and returns its probability:

#%%
pmf.Prob(2)

#%% [markdown]
# The bracket operator does the same thing.

#%%
pmf[2]

#%% [markdown]
# The `Incr` method adds to the probability associated with a given values.

#%%
pmf.Incr(2, 0.2)
pmf[2]

#%% [markdown]
# The `Mult` method multiplies the probability associated with a value.

#%%
pmf.Mult(2, 0.5)
pmf[2]

#%% [markdown]
# `Total` returns the total probability (which is no longer 1, because we changed one of the probabilities).

#%%
pmf.Total()

#%% [markdown]
# `Normalize` divides through by the total probability, making it 1 again.

#%%
pmf.Normalize()
pmf.Total()

#%% [markdown]
# Here's the PMF of pregnancy length for live births.

#%%
pmf = thinkstats2.Pmf(live.prglngth, label='prglngth')

#%% [markdown]
# Here's what it looks like plotted with `Hist`, which makes a bar graph.

#%%
thinkplot.Hist(pmf)
thinkplot.Config(xlabel='Pregnancy length (weeks)', ylabel='Pmf')

#%% [markdown]
# Here's what it looks like plotted with `Pmf`, which makes a step function.

#%%
thinkplot.Pmf(pmf)
thinkplot.Config(xlabel='Pregnancy length (weeks)', ylabel='Pmf')

#%% [markdown]
# We can use `MakeFrames` to return DataFrames for all live births, first babies, and others.

#%%
live, firsts, others = first.MakeFrames()

#%% [markdown]
# Here are the distributions of pregnancy length.

#%%
first_pmf = thinkstats2.Pmf(firsts.prglngth, label='firsts')
other_pmf = thinkstats2.Pmf(others.prglngth, label='others')

#%% [markdown]
# And here's the code that replicates one of the figures in the chapter.

#%%
width=0.45
axis = [27, 46, 0, 0.6]
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(first_pmf, align='right', width=width)
thinkplot.Hist(other_pmf, align='left', width=width)
thinkplot.Config(xlabel='Pregnancy length(weeks)', ylabel='PMF', axis=axis)

thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([first_pmf, other_pmf])
thinkplot.Config(xlabel='Pregnancy length(weeks)', axis=axis)

#%% [markdown]
# Here's the code that generates a plot of the difference in probability (in percentage points) between first babies and others, for each week of pregnancy (showing only pregnancies considered "full term"). 

#%%
weeks = range(35, 46)
diffs = []
for week in weeks:
    p1 = first_pmf.Prob(week)
    p2 = other_pmf.Prob(week)
    diff = 100 * (p1 - p2)
    diffs.append(diff)

thinkplot.Bar(weeks, diffs)
thinkplot.Config(xlabel='Pregnancy length(weeks)', ylabel='Difference (percentage points)')

#%% [markdown]
# ### Biasing and unbiasing PMFs
# 
# Here's the example in the book showing operations we can perform with `Pmf` objects.
# 
# Suppose we have the following distribution of class sizes.

#%%
d = { 7: 8, 12: 8, 17: 14, 22: 4, 
     27: 6, 32: 12, 37: 8, 42: 3, 47: 2 }

pmf = thinkstats2.Pmf(d, label='actual')

#%% [markdown]
# This function computes the biased PMF we would get if we surveyed students and asked about the size of the classes they are in.

#%%
def BiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
        
    new_pmf.Normalize()
    return new_pmf

#%% [markdown]
# The following graph shows the difference between the actual and observed distributions.

#%%
biased_pmf = BiasPmf(pmf, label='observed')
thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf, biased_pmf])
thinkplot.Config(xlabel='Class size', ylabel='PMF')

#%% [markdown]
# The observed mean is substantially higher than the actual.

#%%
print('Actual mean', pmf.Mean())
print('Observed mean', biased_pmf.Mean())

#%% [markdown]
# If we were only able to collect the biased sample, we could "unbias" it by applying the inverse operation.

#%%
def UnbiasPmf(pmf, label=None):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf[x] *= 1/x
        
    new_pmf.Normalize()
    return new_pmf

#%% [markdown]
# We can unbias the biased PMF:

#%%
unbiased = UnbiasPmf(biased_pmf, label='unbiased')
print('Unbiased mean', unbiased.Mean())

#%% [markdown]
# And plot the two distributions to confirm they are the same.

#%%
thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf, unbiased])
thinkplot.Config(xlabel='Class size', ylabel='PMF')

#%% [markdown]
# ### Pandas indexing
# 
# Here's an example of a small DataFrame.

#%%
import numpy as np
import pandas
array = np.random.randn(4, 2)
df = pandas.DataFrame(array)
df

#%% [markdown]
# We can specify column names when we create the DataFrame:

#%%
columns = ['A', 'B']
df = pandas.DataFrame(array, columns=columns)
df

#%% [markdown]
# We can also specify an index that contains labels for the rows.

#%%
index = ['a', 'b', 'c', 'd']
df = pandas.DataFrame(array, columns=columns, index=index)
df

#%% [markdown]
# Normal indexing selects columns.

#%%
df['A']

#%% [markdown]
# We can use the `loc` attribute to select rows.

#%%
df.loc['a']

#%% [markdown]
# If you don't want to use the row labels and prefer to access the rows using integer indices, you can use the `iloc` attribute:

#%%
df.iloc[0]

#%% [markdown]
# `loc` can also take a list of labels.

#%%
indices = ['a', 'c']
df.loc[indices]

#%% [markdown]
# If you provide a slice of labels, `DataFrame` uses it to select rows.

#%%
df['a':'c']

#%% [markdown]
# If you provide a slice of integers, `DataFrame` selects rows by integer index.

#%%
df[0:2]

#%% [markdown]
# But notice that one method includes the last elements of the slice and one does not.
# 
# In general, I recommend giving labels to the rows and names to the columns, and using them consistently.
#%% [markdown]
# ## Exercises
#%% [markdown]
# **Exercise:** Something like the class size paradox appears if you survey children and ask how many children are in their family. Families with many children are more likely to appear in your sample, and families with no children have no chance to be in the sample.
# 
# Use the NSFG respondent variable `numkdhh` to construct the actual distribution for the number of children under 18 in the respondents' households.
# 
# Now compute the biased distribution we would see if we surveyed the children and asked them how many children under 18 (including themselves) are in their household.
# 
# Plot the actual and biased distributions, and compute their means.

#%%
resp = nsfg.ReadFemResp()


#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here

#%% [markdown]
# **Exercise:** I started this book with the question, "Are first babies more likely to be late?" To address it, I computed the difference in means between groups of babies, but I ignored the possibility that there might be a difference between first babies and others for the same woman.
# 
# To address this version of the question, select respondents who have at least two live births and compute pairwise differences. Does this formulation of the question yield a different result?
# 
# Hint: use `nsfg.MakePregMap`:

#%%
live, firsts, others = first.MakeFrames()


#%%
preg_map = nsfg.MakePregMap(live)


#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here

#%% [markdown]
# **Exercise:** In most foot races, everyone starts at the same time. If you are a fast runner, you usually pass a lot of people at the beginning of the race, but after a few miles everyone around you is going at the same speed.
# When I ran a long-distance (209 miles) relay race for the first time, I noticed an odd phenomenon: when I overtook another runner, I was usually much faster, and when another runner overtook me, he was usually much faster.
# 
# At first I thought that the distribution of speeds might be bimodal; that is, there were many slow runners and many fast runners, but few at my speed.
# 
# Then I realized that I was the victim of a bias similar to the effect of class size. The race was unusual in two ways: it used a staggered start, so teams started at different times; also, many teams included runners at different levels of ability.
# 
# As a result, runners were spread out along the course with little relationship between speed and location. When I joined the race, the runners near me were (pretty much) a random sample of the runners in the race.
# 
# So where does the bias come from? During my time on the course, the chance of overtaking a runner, or being overtaken, is proportional to the difference in our speeds. I am more likely to catch a slow runner, and more likely to be caught by a fast runner. But runners at the same speed are unlikely to see each other.
# 
# Write a function called `ObservedPmf` that takes a `Pmf` representing the actual distribution of runners’ speeds, and the speed of a running observer, and returns a new `Pmf` representing the distribution of runners’ speeds as seen by the observer.
# 
# To test your function, you can use `relay.py`, which reads the results from the James Joyce Ramble 10K in Dedham MA and converts the pace of each runner to mph.
# 
# Compute the distribution of speeds you would observe if you ran a relay race at 7 mph with this group of runners.

#%%
import relay

results = relay.ReadResults()
speeds = relay.GetSpeeds(results)
speeds = relay.BinData(speeds, 3, 12, 100)


#%%
pmf = thinkstats2.Pmf(speeds, 'actual speeds')
thinkplot.Pmf(pmf)
thinkplot.Config(xlabel='Speed (mph)', ylabel='PMF')


#%%
# Solution goes here


#%%
# Solution goes here


#%%



