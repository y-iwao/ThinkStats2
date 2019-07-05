# %% [markdown]
#  ## 3.1 Pmf
#  - 確率質量関数 Probability mass function
# %%
import thinkplot
import numpy as np
import nsfg
import thinkstats2
pmf = thinkstats2.Pmf([1, 2, 2, 3, 5])
pmf
# %%
# Pmf は正規化されているため全体の確率は１
pmf.Total()
# %%
# ブラケット演算子で確率を取得できる
pmf[2]
# %%
# ある値の確率だけ変更することも可能
pmf.Incr(2, 0.2)
pmf
# %%
# その場合、total は１にならない
pmf.Total()
# %%
# total を１にするには 再び正規化する必要がある
pmf.Normalize()
pmf.Total()
# %% [markdown]
#  ## 3.2 Pmf をプロット
# %%
# preg の読み込み
%matplotlib inline
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]
# %%
# 第1子と第2子以降に分ける
tmp = live.copy()
tmp.loc[tmp.prglngth <= 27] = np.nan
tmp.loc[tmp.prglngth > 47] = np.nan
firsts = tmp[tmp.birthord == 1]
others = tmp[tmp.birthord != 1]
first_pmf = thinkstats2.Pmf(firsts.prglngth)
others_pmf = thinkstats2.Pmf(others.prglngth)
# %%
# 棒グラフ表示
width = 0.45
thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(first_pmf, align='right', width=width)
thinkplot.Hist(others_pmf, align='left', width=width)
thinkplot.Config(xlabl='week', ylabel='probability', axis=[27, 46, 0, 0.6])
thinkplot.show()
# %%
# ステップ関数表示
thinkplot.PrePlot(2)
thinkplot.Pmfs([first_pmf, others_pmf])
thinkplot.show(xlabl='week', ylabel='probability', axis=[27, 46, 0, 0.6])
# %% [markdown]
# ## 3.3 その他の可視化
# %%
# 差を棒グラフで表示
weeks = range(35, 46)
diffs = []
for week in weeks:
    p1 = first_pmf.Prob(week)
    p2 = others_pmf.Prob(week)
    diff = 100 * (p1 - p2)
    diffs.append(diff)

thinkplot.Bar(weeks, diffs)
# %% [markdown]
# ## 3.4 クラスサイズのパラドックス
# %%
d = {7: 8, 12: 8, 17: 14, 22: 4, 27: 6, 32: 12, 37: 8, 42: 3, 47: 2}
pmf = thinkstats2.Pmf(d, label='actual')
print('mean of the actual pmf:', pmf.Mean())
# %%
def BiasPmf(pmf, label):
    """バイアスされたPmfを推定する
    
    Arguments:
        pmf {Pmf} -- actual pmf
        label {string} -- label
    
    Returns:
        Pmf -- バイアスされたPmf
    """
    new_pmf = pmf.Copy(label=label)

    for x, _ in pmf.Items():
        new_pmf.Mult(x, x)

    new_pmf.Normalize()
    return new_pmf

# %%
# バイアスした分布を計算
biased_pmf = BiasPmf(pmf, label='observed')
print('mean of the biased pmf:', biased_pmf.Mean())
thinkplot.Pmfs([pmf, biased_pmf])
thinkplot.show(xlabel='class size', ylabel='PMF')
# %%


def UnbiasPmf(pmf, label):
    """Pmfからバイアスを取り除く
    
    Arguments:
        pmf {Pmf} -- pmf
        label {string} -- label
    
    Returns:
        Pmf -- バイアスを取り除いたPmf
    """
    new_pmf = pmf.Copy(label=label)

    for x, _ in pmf.Items():
        new_pmf.Mult(x, 1.0/x)

    new_pmf.Normalize()
    return new_pmf


# %%
# バイアスした分布からバイアスを取り除く
unbiased_pmf = UnbiasPmf(biased_pmf, 'expected')
print('mean of the unbiased pmf:', unbiased_pmf.Mean())
thinkplot.Pmfs([pmf, unbiased_pmf])
thinkplot.show(xlabel='class size', ylabel='PMF')
# %%　[markdown]
# ## 3.5 DataFrame のインデックス処理
#%%
# NumPy 配列からDataFrameを作成 
import numpy as np
import pandas

array = np.random.randn(4, 2)
df = pandas.DataFrame(array)
df 
#%%
# カラム名を指定して作成
columns = ['A', 'B']
df = pandas.DataFrame(array, columns=columns)
df
#%%
# インデックスも指定して作成
index = ['a', 'b', 'c', 'd']
df = pandas.DataFrame(array, columns= columns, index=index)
df
#%%
# カラムを選択するとSeries が返される
df['A']
#%%
# 行の選択にはlocを用いる
df.loc['a']
#%%
# iloc も可
df.iloc[0]
#%%
# ラベルのリストも可
indeces = ['a', 'b']
df.loc[indeces]

#%%
# 範囲選択も可
df['a':'c']

#%%
# 整数指定の範囲選択では終端は含まれない
df[0:2]
