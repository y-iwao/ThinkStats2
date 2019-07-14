# %% [markdown]
# # 5. 分布をモデル化する
# ## 5.1 指数分布
# $$ CDF(x) = 1 - e^{-\lambda x} $$
#  
#%%
import numpy as np
import thinkstats2
 
def exponential_cdf(lam, x):
    """compute exponential cdf 
    
    Arguments:
        lam : float
            parameter lambda         
        x : array
            variables
    Returns:
        y : array
            CDF
    """
    y = 1.0 - np.exp(- lam * x)
    return y

#%%
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 100)
plt.plot(x, exponential_cdf(2.0, x), label=r"$\lambda$ = 2.0")
plt.plot(x, exponential_cdf(1.0, x), label=r"$\lambda$ = 1.0")
plt.plot(x, exponential_cdf(0.5, x), label=r"$\lambda$ = 0.5")
plt.xlim(0, 3)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()

#%%
# the distribution of interarrival times from a dataset of birth times.
import analytic
import thinkplot 
df =  analytic.ReadBabyBoom()
diffs = df.minutes.diff()
cdf = thinkstats2.Cdf(diffs, label='actual')

thinkplot.Cdf(cdf)
thinkplot.Show(xlabel='minutes', ylabel='CDF')

#%% [markdown]
# ### CCDF
# - complementary CDF
# 1- CDF(x) 
#
# CCDF of exponential distribution
# $$ y = e^{-\lambda x} $$
# take the log of both side
# $$ \log y = -\lambda x $$ 

#%%
# CCDF of birth interarrival times
thinkplot.Cdf(cdf, complement=True)
thinkplot.Show(xlabel='minutes', ylabel='CCDF', yscale='log')

#%% 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
xs = []
ys = []
for x in cdf.Values():
    if(60 < x):
         break
    p = cdf.Prob(x)
    y = np.log((1 - p))
    xs.append(x)
    ys.append(y) 

X = np.array(xs).reshape(-1, 1)
lr = LinearRegression().fit(X, ys)

plt.plot(X, ys, label="actual")
plt.plot(X, lr.coef_ * X + lr.intercept_, label="Linear regression")
plt.xlabel("minutes")
plt.ylabel("log CCDF")
plt.legend()
print(r"probability of birth per unit time = $\lambda$: {:.3f} babies/minute".format(-lr.coef_[0]))
print(r"mean interarrival time = 1/$\lambda$: {:.1f} minutes".format(-1.0 / lr.coef_[0]))
print("score: {:.3f} ".format(lr.score(X, ys)))

#%% [markdown]
# ## 5.2 Normal distribution 

#%%
import scipy.stats
scipy.stats.norm.cdf(0)

#%%
# plot normal cdf
x = np.linspace(-1, 4, 100)
for mu, sigma in zip([1, 2, 3], [0.5, 0.4, 0.3]):
    p = thinkstats2.EvalNormalCdf(x, mu=mu, sigma = sigma)
    thinkplot.plot(x, p, label=(r"$\mu$={} $\sigma$={}".format(mu, sigma)))

thinkplot.Config(xlabel='x', ylabel='CDF')

#%%
import nsfg
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]

#%%
# without trimming
totalwgt_lb = live.totalwgt_lb.dropna()
cdf = thinkstats2.Cdf(totalwgt_lb)
mu = totalwgt_lb.mean()
sigma = totalwgt_lb.std() 
x = cdf.Values()
y = thinkstats2.EvalNormalCdf(x, mu=mu, sigma=sigma)
thinkplot.plot(x, cdf.Probs(x), label='Data')
thinkplot.plot(x, y, label=r'Model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
thinkplot.Config(xlabel="weight (pounds)", ylabel="CDF")


#%%
# with trimming
totalwgt_lb = live.totalwgt_lb.dropna()
cdf = thinkstats2.Cdf(totalwgt_lb)
mu, var = thinkstats2.TrimmedMeanVar(totalwgt_lb, p=0.01)
sigma = np.sqrt(var)
x = cdf.Values()
y = thinkstats2.EvalNormalCdf(x, mu=mu, sigma=sigma)
thinkplot.plot(x, cdf.Probs(x), label='Data')
thinkplot.plot(x, y, label=r'Model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
thinkplot.Config(xlabel="weight (pounds)", ylabel="CDF")

#%% [markdown]
# ## 5.3 Normal probability plt

#%%
n = 1000
thinkplot.PrePlot(3)
for mu, sigma in zip([0, 1, 5], [1, 1, 2]):
    sample = np.random.normal(mu, sigma, n)
    xs, ys = thinkstats2.NormalProbability(sample)
    thinkplot.plot(xs, ys, label=r"$\mu$={} $\sigma$={}".format(mu, sigma))
thinkplot.Config(title="Normal probability plot", xlabel="standard normal sample", ylabel="sample value")   

#%%
mu, var = thinkstats2.TrimmedMeanVar(totalwgt_lb, p=0.01)
maturity = live[live.prglngth >= 37].totalwgt_lb.dropna()
sigma = np.sqrt(var)
xs = [-4, 4]
fxs, fys =thinkstats2.FitLine(xs, inter=mu, slope=sigma)
thinkplot.plot(fxs, fys, color='gray',
 label=r'model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
xs, ys = thinkstats2.NormalProbability(totalwgt_lb)
thinkplot.Plot(xs, ys, label="all")
xs, ys = thinkstats2.NormalProbability(maturity)
thinkplot.Plot(xs, ys, label="maturity")
thinkplot.Config()

#%% [markdown]
# ## Lognormal distribution

#%%
import brfss
df = brfss.ReadBrfss()
weights = df.wtkg2.dropna()

#%%
def PlotNormalModel(sample, title="", xlabel=""):
    cdf = thinkstats2.Cdf(sample, label="actual")
    mu, var = thinkstats2.TrimmedMeanVar(sample, p=0.01)
    sigma = np.sqrt(var)
    xmin = mu - 4.0 * sigma
    xmax = mu + 4.0 * sigma
    xs, ys = thinkstats2.RenderNormalCdf(mu, sigma, xmin, xmax)
    thinkplot.Cdf(cdf)
    thinkplot.plot(xs, ys,
     label=r'model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
    thinkplot.Config(title=title, xlabel=xlabel, ylabel="CDF")

#%%
PlotNormalModel(weights, title="Normal model", xlabel="weights (kg)")

#%%
log_weights = np.log10(weights)
PlotNormalModel(log_weights, title="Log normal model", xlabel="log weights (log10 kg)")

#%%
def PlotNormalProbability(sample, title="", ylabel=""):
    mu, var = thinkstats2.TrimmedMeanVar(sample, p=0.01)
    sigma = np.sqrt(var)
    xs = [-5, 5]
    fxs, fys =thinkstats2.FitLine(xs, inter=mu, slope=sigma)
    thinkplot.plot(fxs, fys, color='gray',
     label=r'model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
    xs, ys = thinkstats2.NormalProbability(sample)
    thinkplot.Plot(xs, ys, label="actual")
    thinkplot.Config(title=title, xlabel="z", ylabel=ylabel)

#%%
PlotNormalProbability(weights, title="Normal model", ylabel="weights (kg)")

#%%
PlotNormalProbability(log_weights, title="Log Normal model", ylabel="log weights (log10 kg")


#%% [markdown]
# ## 5.5 Pareto distribution
# $$ CDF(x) = 1 - (\frac{x}{x_m})^{-\alpha}$$
# $$ y = CCDF(x) = (\frac{x}{x_m})^{-\alpha} $$
# $$ \log{y} = -\alpha (\log{x} -\log{x_m}) $$
#%%
def pareto_distribution(x, xmin, alpha):
    return 1.0 - (x/xmin)**(-alpha)

#%%
thinkplot.preplot(3)
xmin = 0.5
x = np.linspace(xmin, 10, 100)
for alpha in [2, 1, 0.5]:
    thinkplot.plot(x, pareto_distribution(x, xmin, alpha),
     label=r"$\alpha = {}$".format(alpha))
thinkplot.Config(title='Pareto distribution', xlabel='x', ylabel='CDF')

#%%
import populations

pops = populations.ReadData()
print('Number of cities/towns', len(pops))

#%%
cdf = thinkstats2.Cdf(pops)
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel="population", ylabel="CDF")

#%%
log_pops = np.log10(pops)
log_cdf = thinkstats2.Cdf(log_pops, label='data')
thinkplot.Cdf(log_cdf, complement=True)
xmin = 5000
alpha = 1.4
xs, ys = thinkstats2.RenderParetoCdf(xmin=xmin, alpha=alpha, low=0, high=1.0e7)
thinkplot.Plot(np.log10(xs), 1-ys,
 label=r'model $x_m={}$  $\alpha={}$'.format(xmin, alpha))
thinkplot.Config(yscale='log', xlabel='log10 pupulation', ylabel='CCDF')

#%%
thinkplot.Cdf(log_cdf)
mu, var = thinkstats2.TrimmedMeanVar(log_pops, p=0.01)
sigma = np.sqrt(var)
xmin = mu - 4.0 * sigma
xmax = mu + 4.0 * sigma
xs, ys = thinkstats2.RenderNormalCdf(mu, sigma, xmin, xmax)
thinkplot.plot(xs, ys,
 label=r'model $\mu$={:.2f} $\sigma$={:.2f}'.format(mu, sigma))
thinkplot.Config(xlabel='log10 pupulation', ylabel='CDF')

#%%
PlotNormalProbability(log_pops, ylabel="log10 population")

#%% [markdown]
# ## 5.6 random

#%% 
import math
def expovariate(lam):
    p = np.random.random()
    x = -math.log(1 - p) / lam
    return x


#%%
