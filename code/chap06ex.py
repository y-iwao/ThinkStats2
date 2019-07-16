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

import brfss

import thinkstats2
import thinkplot

#%% [markdown]
# I'll start with the data from the BRFSS again.

#%%
df = brfss.ReadBrfss(nrows=None)

#%% [markdown]
# Here are the mean and standard deviation of female height in cm.

#%%
female = df[df.sex==2]
female_heights = female.htm3.dropna()
mean, std = female_heights.mean(), female_heights.std()
mean, std

#%% [markdown]
# `NormalPdf` returns a Pdf object that represents the normal distribution with the given parameters.
# 
# `Density` returns a probability density, which doesn't mean much by itself.

#%%
pdf = thinkstats2.NormalPdf(mean, std)
pdf.Density(mean + std)

#%% [markdown]
# `thinkplot` provides `Pdf`, which plots the probability density with a smooth curve.

#%%
thinkplot.Pdf(pdf, label='normal')
thinkplot.Config(xlabel='x', ylabel='PDF', xlim=[140, 186])

#%% [markdown]
# `Pdf` provides `MakePmf`, which returns a `Pmf` object that approximates the `Pdf`. 

#%%
pmf = pdf.MakePmf()
thinkplot.Pmf(pmf, label='normal')
thinkplot.Config(xlabel='x', ylabel='PDF', xlim=[140, 186])

#%% [markdown]
# If you have a `Pmf`, you can also plot it using `Pdf`, if you have reason to think it should be represented as a smooth curve.

#%%
thinkplot.Pdf(pmf, label='normal')
thinkplot.Config(xlabel='x', ylabel='PDF', xlim=[140, 186])

#%% [markdown]
# Using a sample from the actual distribution, we can estimate the PDF using Kernel Density Estimation (KDE).
# 
# If you run this a few times, you'll see how much variation there is in the estimate.

#%%
thinkplot.Pdf(pdf, label='normal')

sample = np.random.normal(mean, std, 500)
sample_pdf = thinkstats2.EstimatedPdf(sample, label='sample')
thinkplot.Pdf(sample_pdf, label='sample KDE')
thinkplot.Config(xlabel='x', ylabel='PDF', xlim=[140, 186])

#%% [markdown]
# ## Moments
# 
# Raw moments are just sums of powers.

#%%
def RawMoment(xs, k):
    return sum(x**k for x in xs) / len(xs)

#%% [markdown]
# The first raw moment is the mean.  The other raw moments don't mean much.

#%%
RawMoment(female_heights, 1), RawMoment(female_heights, 2), RawMoment(female_heights, 3)


#%%
def Mean(xs):
    return RawMoment(xs, 1)

Mean(female_heights)

#%% [markdown]
# The central moments are powers of distances from the mean.

#%%
def CentralMoment(xs, k):
    mean = RawMoment(xs, 1)
    return sum((x - mean)**k for x in xs) / len(xs)

#%% [markdown]
# The first central moment is approximately 0.  The second central moment is the variance.

#%%
CentralMoment(female_heights, 1), CentralMoment(female_heights, 2), CentralMoment(female_heights, 3)


#%%
def Var(xs):
    return CentralMoment(xs, 2)

Var(female_heights)

#%% [markdown]
# The standardized moments are ratios of central moments, with powers chosen to make the dimensions cancel.

#%%
def StandardizedMoment(xs, k):
    var = CentralMoment(xs, 2)
    std = np.sqrt(var)
    return CentralMoment(xs, k) / std**k

#%% [markdown]
# The third standardized moment is skewness.

#%%
StandardizedMoment(female_heights, 1), StandardizedMoment(female_heights, 2), StandardizedMoment(female_heights, 3)


#%%
def Skewness(xs):
    return StandardizedMoment(xs, 3)

Skewness(female_heights)

#%% [markdown]
# Normally a negative skewness indicates that the distribution has a longer tail on the left.  In that case, the mean is usually less than the median.

#%%
def Median(xs):
    cdf = thinkstats2.Cdf(xs)
    return cdf.Value(0.5)

#%% [markdown]
# But in this case the mean is greater than the median, which indicates skew to the right.

#%%
Mean(female_heights), Median(female_heights)

#%% [markdown]
# Because the skewness is based on the third moment, it is not robust; that is, it depends strongly on a few outliers.  Pearson's median skewness is more robust.

#%%
def PearsonMedianSkewness(xs):
    median = Median(xs)
    mean = RawMoment(xs, 1)
    var = CentralMoment(xs, 2)
    std = np.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp

#%% [markdown]
# Pearson's skewness is positive, indicating that the distribution of female heights is slightly skewed to the right.

#%%
PearsonMedianSkewness(female_heights)

#%% [markdown]
# ## Birth weights
# 
# Let's look at the distribution of birth weights again.

#%%
import first

live, firsts, others = first.MakeFrames()

#%% [markdown]
# Based on KDE, it looks like the distribution is skewed to the left.

#%%
birth_weights = live.totalwgt_lb.dropna()
pdf = thinkstats2.EstimatedPdf(birth_weights)
thinkplot.Pdf(pdf, label='birth weight')
thinkplot.Config(xlabel='Birth weight (pounds)', ylabel='PDF')

#%% [markdown]
# The mean is less than the median, which is consistent with left skew.

#%%
Mean(birth_weights), Median(birth_weights)

#%% [markdown]
# And both ways of computing skew are negative, which is consistent with left skew.

#%%
Skewness(birth_weights), PearsonMedianSkewness(birth_weights)

#%% [markdown]
# ## Adult weights
# 
# Now let's look at adult weights from the BRFSS.  The distribution looks skewed to the right.

#%%
adult_weights = df.wtkg2.dropna()
pdf = thinkstats2.EstimatedPdf(adult_weights)
thinkplot.Pdf(pdf, label='Adult weight')
thinkplot.Config(xlabel='Adult weight (kg)', ylabel='PDF')

#%% [markdown]
# The mean is greater than the median, which is consistent with skew to the right.

#%%
Mean(adult_weights), Median(adult_weights)

#%% [markdown]
# And both ways of computing skewness are positive.

#%%
Skewness(adult_weights), PearsonMedianSkewness(adult_weights)

#%% [markdown]
# ## Exercises
#%% [markdown]
# The distribution of income is famously skewed to the right. In this exercise, we’ll measure how strong that skew is.
# The Current Population Survey (CPS) is a joint effort of the Bureau of Labor Statistics and the Census Bureau to study income and related variables. Data collected in 2013 is available from http://www.census.gov/hhes/www/cpstables/032013/hhinc/toc.htm. I downloaded `hinc06.xls`, which is an Excel spreadsheet with information about household income, and converted it to `hinc06.csv`, a CSV file you will find in the repository for this book. You will also find `hinc2.py`, which reads this file and transforms the data.
# 
# The dataset is in the form of a series of income ranges and the number of respondents who fell in each range. The lowest range includes respondents who reported annual household income “Under \$5000.” The highest range includes respondents who made “\$250,000 or more.”
# 
# To estimate mean and other statistics from these data, we have to make some assumptions about the lower and upper bounds, and how the values are distributed in each range. `hinc2.py` provides `InterpolateSample`, which shows one way to model this data. It takes a `DataFrame` with a column, `income`, that contains the upper bound of each range, and `freq`, which contains the number of respondents in each frame.
# 
# It also takes `log_upper`, which is an assumed upper bound on the highest range, expressed in `log10` dollars. The default value, `log_upper=6.0` represents the assumption that the largest income among the respondents is $10^6$, or one million dollars.
# 
# `InterpolateSample` generates a pseudo-sample; that is, a sample of household incomes that yields the same number of respondents in each range as the actual data. It assumes that incomes in each range are equally spaced on a `log10` scale.

#%%
def InterpolateSample(df, log_upper=6.0):
    """Makes a sample of log10 household income.

    Assumes that log10 income is uniform in each range.

    df: DataFrame with columns income and freq
    log_upper: log10 of the assumed upper bound for the highest range

    returns: NumPy array of log10 household income
    """
    # compute the log10 of the upper bound for each range
    df['log_upper'] = np.log10(df.income)

    # get the lower bounds by shifting the upper bound and filling in
    # the first element
    df['log_lower'] = df.log_upper.shift(1)
    df.loc[0, 'log_lower'] = 3.0

    # plug in a value for the unknown upper bound of the highest range
    df.loc[41, 'log_upper'] = log_upper
    
    # use the freq column to generate the right number of values in
    # each range
    arrays = []
    for _, row in df.iterrows():
        vals = np.linspace(row.log_lower, row.log_upper, row.freq)
        arrays.append(vals)

    # collect the arrays into a single sample
    log_sample = np.concatenate(arrays)
    return log_sample


#%%
import hinc
income_df = hinc.ReadData()


#%%
log_sample = InterpolateSample(income_df, log_upper=6.0)


#%%
log_cdf = thinkstats2.Cdf(log_sample)
thinkplot.Cdf(log_cdf)
thinkplot.Config(xlabel='Household income (log $)',
               ylabel='CDF')


#%%
sample = np.power(10, log_sample)


#%%
cdf = thinkstats2.Cdf(sample)
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel='Household income ($)',
               ylabel='CDF')

#%% [markdown]
# Compute the median, mean, skewness and Pearson’s skewness of the resulting sample. What fraction of households report a taxable income below the mean? How do the results depend on the assumed upper bound?

#%%
# Solution goes here


#%%
# Solution goes here


#%%
# Solution goes here

#%% [markdown]
# All of this is based on an assumption that the highest income is one million dollars, but that's certainly not correct.  What happens to the skew if the upper bound is 10 million?
# 
# Without better information about the top of this distribution, we can't say much about the skewness of the distribution.

#%%



