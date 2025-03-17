# 因子生成阶段
factor_generation = """
作为专业的量化分析师，你的专长在于深入分析及优化因子以提升其信息系数( IC)表现。用户将提供一组因子及其当前的IC指标，每个因子由特定的算子与基础数据构成。 
你的任务是对这些因子进行细致的分析，理解其逻辑，目标是在维持因子结构相对简洁的前提 
下，通过调整算子或者数据来增强因子的有效性，改进方法可以是: 
## 算子替换: 例如 add -> sub, ts_mean -> ts_max等 
## 数据替换： 例如 s_fa_workingcapital -> s_fa_investcapital; 1 -> 2等 
注意这里我只是举了几个例子，具体如何替换根据你自己的理解进行，可替换的算子和数据是如下markdown格式的文本:
定义算子
时间序列一元算子
一元算子指的是输入一个变量，输出一个变量的函数
qoq_diff
本期数值和上一期数值的差值
qoq_ratio
本期数值和上一期数值的变化率
yoy_diff
本期数值和去年同期数值的差异
yoy_ratio
本期数值和去年同期数值的变化率
【应用】对于有明显季节性的行业和基本面指标，可以用yoy_diff和yoy_ratio来计算同比变化；如果关注短期趋势性变化，qoq_diff和qoq_ratio更为合适。
横截面一元算子
rank
所有股票按照指标数值排序，最小值是0，最大值是1
norm
所有股票指标数值减去均值，除以方差
minmax
所有股票指标的数值减去最小值，除以最大值到最小值的距离
【应用】如果某个财务指标的取值范围很大，需要通过标准化方式控制取值范围。
中性化算子
regress_size
指标对市值因子中性化
regress_nlsize
指标对非线性市值因子中性化
regress_resvol
指标对特质波动率因子中性化
regress_liquidity
指标对流动性因子中性化
regress_btop
指标对估值因子中性化
regress_growth
指标对成长因子中性化
regress_leverage
指标对财务杠杆因子中性化
regress_momentum
指标对动量因子中性化
regress_beta
指标对beta因子中性化
【应用】如果一个因子有明显的风格暴露，中性化可以提供更纯净的alpha。
二元算子
二元算子输入包含两个变量，输出一个变量
cs_regress_residual
某财务指标对另一个财务指标做线性回归，得到的残差项
add
某个财务指标和另一个财务指标相加
sub
某个财务指标和另一个财务指标相减
div
某个财务指标和另一个财务指标相除
mul
某个财务指标和另一个财务指标相乘
定义财务指标
利润相关指标
利润总额：tot_profit
近期净利润一致预期：con_profit_st
营业收入：oper_rev
营业利润：oper_profit
含少数股东权益的净利润：net_profit_incl_min_int_inc
不含少数股东权益的净利润: net_profit_excl_min_int_inc
息税前利润：ebit
息税折旧摊销前利润：ebitda
扣费后净利润：net_profit_after_ded_nr_lp
基本每股收益：s_fa_eps_basic
稀释每股收益：s_fa_eps_diluted
可分配利润：distributable_profit
可供股东分配的利润：distributable_profit_shrhder
综合收益总额：tot_compreh_inc
所得税：inc_tax
未确认投资损失：unconfirmed_invest_loss
长期净利润一致预期：con_profit_lt
资产负债表指标
货币资金：monetary_cap
交易性金融资产：tradable_fin_assets
应收票据：notes_rcv
营收账款：acct_rcv
流动资产合计：tot_cur_assets
商誉：goodwill
无形资产：intang_assets
非流动性资产合计：tot_non_cur_assets
资产总计：tot_assets
短期借款：st_borrow
应付票据：notes_payable
应付账款：acct_payable
应交税费：taxes_surcharges_payable
应付利息：int_payable
应付股利：dvd_payable
其他应付款：oth_payable
流动负债合计：tot_cur_liab
长期借款：lt_borrow
长期应付款：lt_payable
预计负债：provisions
非流动负债合计：tot_non_cur_liab
负债合计：tot_liab
股本：cap_stk
资本公积金：cap_rsrv
未分配利润：undistributed_profit
少数股东权益：minority_int
股东权益合计（不含少数股东权益）：tot_shrhldr_eqy_excl_min_int
股东权益合计（含少数股东权益）：tot_shrhldr_eqy_incl_min_int
负债及股东权益总计：tot_liab_shrhldr_eqy
现金流量指标
经营活动产现金流入小计：stot_cash_inflows_oper_act
经营活动现金流出小计：stot_cash_outflows_oper_act
经营活动产生的现金流量净额：net_cash_flows_oper_act
投资活动产现金流入小计：stot_cash_inflows_inv_act
投资活动现金流出小计：stot_cash_outflows_inv_act
投资活动产生的现金流量净额：net_cash_flows_inv_act
筹资活动产现金流入小计：stot_cash_inflows_fnc_act
筹资活动现金流出小计：stot_cash_outflows_fnc_act
筹资活动产生的现金流量净额：net_cash_flows_fnc_act
现金及现金等价物净增加额：net_incr_cash_cash_equ
财务费用：fin_exp
企业自由现金流量：free_cash_flow
情感指标
分析师情感：analyst_sentiment
上市公司市场价值指标
总市值：total_cap
流通市值：free_cap
季度价格变化：price_change_quarterly
季度价格波动：price_std_quarterly

改进生成新的因子, 目标是实现IC值的显著提升。请确保给出新的改进因子,不要给出已有因子,在改进过程中注重实效性与因子的可解释性，避免不必要的复杂度增加。完成优化后，直接输出最终优化的因子表达式的列表，给出10个优化因子，并将其格式化为 
JSON，以便于用户直接应用及后续的分析工作, 输出格式为: 
{ "优化因子列表" :  
[ 
{"因子":"***","改进原因":"***"}, 
.. .... 
{"因子":"***","改进原因":"***"} 
] 
}
"""


# 为因子生成测试代码
code_generation = """
你是一名专业的量化开发工程师，现在我给你回测代码的例子，同时给出了所有会用到的算子的代码，帮我实现{}的代码。
这是可能涉及到的算子的代码。
``` python
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
```
下面是一个回测代码的例子。

#数据位置
data_path = "./data"
#载入数据格式
pd.DataFrame，横轴是日期，纵轴是股票代码。存储格式是h5。
#入参
对于任意公式，入参包含数据位置，但不包括因子值。例如对于 rank(div(add(free_cash_flow, con_profit_st), tot_assets))，函数格式是：
factor_cal_21(data_path)
#输出
pd.DataFrame，横轴是日期，纵轴是股票代码ticker
例如，对于 rank(div(add(free_cash_flow, con_profit_st), tot_assets))，函数格式是：

python代码是
``` python
def factor_cal_21(data_path):
    free_cash_flow = pd.read_parquet(os.path.join(data_path, "free_cash_flow.h5"))
    tot_assets = pd.read_parquet(os.path.join(data_path, "tot_assets.h5"))
    con_profit_st = pd.read_hdf(os.path.join(data_path, "con_profit_st.h5"))
    factor = (free_cash_flow + con_profit_st) / tot_assets
    factor = factor.rank(axis=1, pct=True)
    factor = factor.dropna(how='all', axis=0)
    return factor
```
我们后续会将 factor_cal_21 的输出，用如下代码测试，以便打印所需要的单因子回测结果，
import sys
sys.path.append("/sdb/sharedFolder/workspace/libs/pytools")
db_dir = "/sdb/sharedFolder/workspace/Data/DataBase"
from mfrt.research_tools import DBTool
tl = DBTool(db_dir)
from mfrt.opt_backtest import OptBacktest
from mfrt.eval_netvalue import EvalNetValue
opt_bt=OptBacktest(db_dir)
eval_NV=EvalNetValue(db_dir)
from mfrt.eval_factor import EvalFactor
eval_F = EvalFactor(db_dir)
check_Barra_list=('BETA', 'MOMENTUM', 'SIZE', 'RESVOL', 'BTOP', 'LIQUIDTY',  'SPRET', 'SIZENL', 'GROWTH', 'LEVERAGE')
def test_factor(factor: pd.DataFrame, 
                start_dt, 
                end_dt, 
                display: bool=False, 
                if_trade_tmr: bool=True, 
                specific_return: bool=True,
                Universe=None,
                return_type='Vwap') -> dict:
    F_res=eval_F.quick_test(factor,start_dt=start_dt,end_dt=end_dt,Group_Num=10,Horizon=1,Return_Type=return_type,
                        Specific_Return=specific_return,Universe=Universe,check_Barra_list=check_Barra_list,
                        extraExp_check_dict=None,if_trade_tmr=if_trade_tmr, display=display)
    
    return F_res
test_factor(factor, start_dt='2018-01-01', end_dt='2024-10-31', display=True,
            if_trade_tmr=True, specific_return=True, Universe=None, return_type='Vwap')
"""


# 反馈阶段

feedback = """
我提供给你 {}} 因子的回测结果，请根据图表对结果进行分析，并对这个因子进行改进，给出具体的改进因子，输出格式为: { "优化因子列表" : [ {"因子":""," 改进原因 ":""}, .. .... {" 因子 ":""," 改进原因 ":""} ] } 
"""