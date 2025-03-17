# UTF-8
# 各类算子

import statsmodels.api as sm
from sklearn.linear_model import Ridge
import pandas as pd
import sys
sys.path.append("/sdb/sharedFolder/workspace/libs/pytools")
db_dir = "/sdb/sharedFolder/workspace/Data/DataBase"
from mfrt.research_tools import DBTool
tl = DBTool(db_dir)


# 最小二乘法
def get_residual(x: pd.Series, y: pd.Series):
    xx = x.loc[(x.notna())&(y.notna())].values
    yy = y.loc[(x.notna())&(y.notna())].values
    residual = y.copy(deep=True)

    if len(xx) == 0:
        return residual

    ridge = Ridge(alpha=0.5)
    ridge.fit(sm.add_constant(xx), yy)
    res = yy - ridge.predict(sm.add_constant(xx))
    residual.loc[(x.notna())&(y.notna())] = res
    return residual

# 加权最小二乘法
def get_residuals_wls(x: pd.DataFrame, y: pd.Series, w: pd.Series):
    xx = x.loc[(x.notna())&(y.notna())&(w.notna())].values
    yy = y.loc[(x.notna())&(y.notna())&(w.notna())].values
    ww = w.loc[(x.notna())&(y.notna())&(w.notna())].values
    residual = y.copy(deep=True)

    if len(xx) < 2:
        return residual

    model = sm.WLS(yy, sm.add_constant(xx), weights=ww).fit()
    res = yy - model.predict(sm.add_constant(xx))
    residual.loc[(x.notna())&(y.notna())&(w.notna())] = res
    return residual

# 多元最小二乘法
def get_residual_multi(x: pd.DataFrame, y: pd.Series):
    xx = x.loc[(x.isna().sum(1)==0)&(y.notna())].values
    yy = y.loc[(x.isna().sum(1)==0)&(y.notna())].values
    residual = y.copy(deep=True)

    if len(xx) == 0:
        return residual

    ridge = Ridge(alpha=0.5)
    ridge.fit(sm.add_constant(xx), yy)
    res = yy - ridge.predict(sm.add_constant(xx))
    residual.loc[(x.isna().sum(1)==0)&(y.notna())] = res
    return residual

# 标准化到正态分布
def normalize(x: pd.DataFrame):
    return x.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# min-max标准化
def min_max(x: pd.DataFrame):
    return x.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# 分年度计算增长率
def yoy_ratio(x: pd.DataFrame):
    y = x.loc[(x['REPORT_PERIOD'].astype(int))%10000==1231]
    y = y.groupby('REPORT_PERIOD').first()
    y['yoy'] = y['VALUE'].pct_change()
    return y[['yoy','S_INFO_WINDCODE','ANN_DT']]

# 分年度计算增长
def yoy_diff(x: pd.DataFrame):
    y = x.loc[(x['REPORT_PERIOD'].astype(int))%10000==1231]
    y = y.groupby('REPORT_PERIOD').first()
    y['yoy'] = y['VALUE'].diff()
    return y[['yoy','S_INFO_WINDCODE','ANN_DT']]

# 分季度计算增长率
def qoq_ratio(x: pd.DataFrame):
    q = x.loc[(x['REPORT_PERIOD'].astype(int)%10000).isin([331,630,930,1231])]
    q = q.groupby('REPORT_PERIOD').first()
    q['qoq'] = q['VALUE'].pct_change()
    return q[['qoq','S_INFO_WINDCODE','ANN_DT']]

# 分季度计算增长
def qoq_diff(x: pd.DataFrame):
    q = x.loc[(x['REPORT_PERIOD'].astype(int)%10000).isin([331,630,930,1231])]
    q = q.groupby('REPORT_PERIOD').first()
    q['qoq'] = q['VALUE'].diff()
    return q[['qoq','S_INFO_WINDCODE','ANN_DT']]

# 剔除SIZE影响
def regress_size(x: pd.DataFrame):
    size_exposure = tl.load_daily_data("Barra_SIZE")
    factor = x.reindex(size_exposure.index, columns=size_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          size_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除BETA影响
def regress_beta(x: pd.DataFrame):
    beta_exposure = tl.load_daily_data("Barra_BETA")
    factor = x.reindex(beta_exposure.index, columns=beta_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          beta_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除MOMENTUM影响
def regress_momentum(x: pd.DataFrame):
    momentum_exposure = tl.load_daily_data("Barra_MOMENTUM")
    factor = x.reindex(momentum_exposure.index, columns=momentum_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          momentum_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除RESVOL影响
def regress_resvol(x: pd.DataFrame):
    resvol_exposure = tl.load_daily_data("Barra_RESVOL")
    factor = x.reindex(resvol_exposure.index, columns=resvol_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          resvol_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除GROWTH影响
def regress_growth(x: pd.DataFrame):
    growth_exposure = tl.load_daily_data("Barra_GROWTH")
    factor = x.reindex(growth_exposure.index, columns=growth_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          growth_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除BTOP影响
def regress_btop(x: pd.DataFrame):
    btop_exposure = tl.load_daily_data("Barra_BTOP")
    factor = x.reindex(btop_exposure.index, columns=btop_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          btop_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除LEVERAGE影响
def regress_leverage(x: pd.DataFrame):
    leverage_exposure = tl.load_daily_data("Barra_LEVERAGE")
    factor = x.reindex(leverage_exposure.index, columns=leverage_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          leverage_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除LIQUIDITY影响
def regress_liquidity(x: pd.DataFrame):
    liquidity_exposure = tl.load_daily_data("Barra_LIQUIDITY")
    factor = x.reindex(liquidity_exposure.index, columns=liquidity_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          liquidity_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除EARNYILD影响
def regress_earnyild(x: pd.DataFrame):
    earnyild_exposure = tl.load_daily_data("Barra_EARNYILD")
    factor = x.reindex(earnyild_exposure.index, columns=earnyild_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          earnyild_exposure.loc[x.name], x), axis=1)

    return factor

# 剔除SIZENL影响
def regress_sizenl(x: pd.DataFrame):
    sizenl_exposure = tl.load_daily_data("Barra_SIZENL")
    factor = x.reindex(sizenl_exposure.index, columns=sizenl_exposure.columns).progress_apply(lambda x: get_residual(
                                                                                          sizenl_exposure.loc[x.name], x), axis=1)

    return factor