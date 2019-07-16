#%% [markdown]
# ## 6.1 PDF
#
# - probability dencity function
#
# $PDF_{normal}(x) = \frac{1}{\sigma \sqrt{2\pi}}\exp[-\frac{1}{2}(\frac{x-\mu}{\sigma})^2]$

#%%
import thinkstats2
import math 
mean, var = 163, 52.8
std = math.sqrt(var)
pdf = thinkstats2.NormalPdf(mean, std)
pdf.Density(mean + std)

#%%
import thinkplot
thinkplot.Pdf(pdf, label='normal')
thinkplot.Show(xlabel='height (cm)', ylabel='dencity')


#%%
pmf = pdf.MakePmf()

#%% [markdown]
# ## 6.2 KDE
#
# - Kernel density estimation

#%%
import random
sample = [random.gauss(mean, std) for _ in range(500) ]
sample_pdf = thinkstats2.EstimatedPdf(sample)
thinkplot.Pdf(sample_pdf, label='sample KDE')
thinkplot.Pdf(pdf, label='normal')
thinkplot.Show(xlabel='height (cm)', ylabel='dencity')

#%%
import numpy as np
hist = thinkstats2.Hist(np.floor(sample))
thinkplot.Hist(hist)

#%%
cdf = thinkstats2.Cdf(np.floor(sample))
thinkplot.Cdf(cdf)

#%% [markdown]
# 
# ### Raw moment 
# $ m_k = \frac{1}{n} \sum_{i}{{x_i}^k} $    

#%%
def RawMoment(xs, k):
    return sum(x**k for x in xs) / len(xs)

#%% [markdown]
#
# ### Cenral moment
# $ m_k = \frac{1}{n} \sum_{i}{(x_i - \overline{x})^k} $

#%%
def CentralMoment(xs, k):
    mena = RawMoment(xs, 1)
    return sum((x - xs)**k for x in xs) / len(xs)

#%%
def StandardizedMoment(xs, k):
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    return CentralMoment(xs, k) / std**k

#%%
def Skewness(xs):
    return StandardizedMoment(xs, 3)

#%% [markdown]
# ### Pearson's median skewness coefficient
#
# $ g_p = 3(\overline{x} - m)/S $
#

#%%
def Medinan(xs):
    cdf = thinkstats2.MakeCdfFromList(xs)
    return cdf.Value(0.5)

#%% 
def PearsonMedianSkewness(xs):
    mean = RawMoment(xs, 1)
    medi = Medinan(xs)
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    return 3 * (mean - medi) / std

#%%
import first
live, first, others = first.MakeFrames()
data = live.totalwgt_lb.dropna()
pdf = thinkstats2.EstimatedPdf(data)
thinkplot.Pdf(pdf, label='birth weight')
thinkplot.Config(xlable='birth weight (pounds)', ylable='PDF')

#%%
mean = RawMoment(data, 1)
print("mean: {:.2f} pounds".format(mean))
medi = Medinan(data)
print("median: {:.2f} pounds".format(medi))
skewness = Skewness(data)
print("skewness: {:.2f} ".format(skewness))
pearson = PearsonMedianSkewness(data)
print("pearson's median skewness: {:.2f}".format(pearson))

#%%
# BRFSS
import brfss
df = brfss.ReadBrfss(nrows=None)
data = df.wtkg2.dropna()
pdf = thinkstats2.EstimatedPdf(data)
thinkplot.Pdf(pdf, label="adult weight")
thinkplot.Config(xlable='weight (kg)', ylable='PDF')


#%%
pdf = thinkstats2.EstimatedPdf(data)
thinkplot.Pdf(pdf, label="adult weight")
thinkplot.Config(xlable='weight (kg)', ylable='PDF', xlim=[0, 200])

#%%
mean = RawMoment(data, 1)
print("mean: {:.1f} kg".format(mean))
medi = Medinan(data)
print("median: {:.1f} kg".format(medi))
skewness = Skewness(data)
print("skewness: {:.2f} ".format(skewness))
pearson = PearsonMedianSkewness(data)
print("pearson's median skewness: {:.2f}".format(pearson))

#%%
