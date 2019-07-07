#%% [markdown]

# # 4 累積分布関数


#%% [markdown]

# ## 4.1 PMF の限界

# - PMF は値の個数が増えるとランダムノイズの影響が大きくなる
# - PMF では２つの分布の比較が難しい

# - CDF 累積分布関数 cumulative distribution function 

# ## 4.2 パーセンタイル

#%%
def PercentileRank(scores, your_score):
    """Compute the percentile rank of a given score
    
    Arguments:
        scores {list} 
        your_score {float} 
    
    Returns:
        [float] -- the percentile rank
    """
    count =0
    for score in scores:
        if score <= your_score:     
            count += 1
    
    percentile_rank = (100.0 * count) / len(scores)
    return percentile_rank

#%%
scores = [55, 66, 77, 88, 99]
percentile_rank = PercentileRank(scores, 88)
print(percentile_rank)

#%%
def Percentile(scores, percentile_rank):
    """Get the percentile value of a given percentil rank.
    
    Arguments:
        scores {link}
        percentile_rank {float} 
    
    Returns:
        [float] -- the percentile rank
    """
    scores.sort()
    for score in scores:
        if PercentileRank(scores, score) >= percentile_rank:
            return score

#%%
percentile = Percentile(scores, percentile_rank)
print(percentile)

#%%
def Percentile2(scores, percentile_rank):
    """Get the percentile value of a given percentile rank.
    
    More effective implementation.
    
    Arguments:
        scores {list} 
        percentile_rank {float} 
    
    Returns:
        [float] -- the percentile value 
    """
    scores.sort()
    index = int(percentile_rank * (len(scores) - 1) / 100)
    return scores[index]

#%%
percentile = Percentile2(scores, percentile_rank)
print(percentile)

#%% [markdown]
# ## 4.3 累積分布関数（CDF）

#%%
def EvalCdf(t, x):
    """ Compute the CDF of given value
    
    Arguments:
        t {list} -- value sequence
        x {float} -- value
    
    Returns:
        [float] -- CDF
    """
    count = 0.0
    for value in t:
        if value <= x:
            count += 1
    
    prob = count / len(t)
    return prob


#%%
t = [1, 2, 2, 3, 5]
for x in range(6): 
    print("CDF({0}) = {1}".format(x, EvalCdf(t, x)))
#%% [markdown]
# ## 4.4 CDF の表現

#%%
import thinkstats2
import first
import thinkplot

live , firsts,others = first.MakeFrames() 
cdf = thinkstats2.Cdf(live.prglngth, label='prglngth')
thinkplot.Cdf(cdf)
thinkplot.show(xlabel='weeks', ylabel='CDF')

#%%

print("10% {0} weeks".format(cdf.Value(0.1)))
print("90% {0} weeks".format(cdf.Value(0.9)))

#%% [markdown]

# ## 4.5 CDFを比較する

#%%
first_cdf = thinkstats2.Cdf(firsts.totalwgt_lb, label='first')
other_cdf = thinkstats2.Cdf(others.totalwgt_lb, label='other')
thinkplot.PrePlot(2)
thinkplot.Cdfs([first_cdf, other_cdf])
thinkplot.Show(xlabel='weight (pounds)', ylabel='CDF') 

#%% [markdown]

# ## 4.6 パーセンタイル派生統計量

# - 中央値(median)：50位パーセンタイル値
# - 四分位範囲(interquartile range, IQR)：75位 -　25位パーセンタイル値
# - 分位数(quantiles)：CDFにおいて等間隔で表現される統計量
#%% [markdown]
# ## 4.7 乱数

#%%
import numpy as np
weights = live.totalwgt_lb
cdf = thinkstats2.Cdf(weights, label='totalwgt_lb')
sample = np.random.choice(weights, 100, replace=True)
ranks = [cdf.PercentileRank(x) for x in sample]
rank_cdf = thinkstats2.Cdf(ranks)
thinkplot.Cdf(rank_cdf)
thinkplot.show(xlabel='percentile rank', ylabel='CDF')

#%% [markdown]
# ## 4.8 パーセンタイル順位を比較する

#%%
def PositionToPercentile(position, field_size):
    """Compute the percentile rank of a given position
    
    Arguments:
        position {int} 
        field_size {int}
    
    Returns:
        float -- the percentile rank
    """
    beat = field_size - position + 1
    percentile  = (100.0 * beat) / field_size
    return percentile

#%%
field_size=256
position = 26
percentile_rank = PositionToPercentile(position, field_size)
print(percentile_rank)

#%%
def PercentileToPositon(percentile, field_size):
    beat = percentile * field_size / 100.0
    position = field_size - beat + 1
    return position


#%%
position = PercentileToPositon(percentile_rank, field_size)
print(position)

#%%
new_field_size = 171
new_position = PercentileToPositon(percentile_rank, new_field_size)
print(new_position)

#%%
