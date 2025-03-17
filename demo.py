# import dify
# from dify import Task, Workflow
import openai  
from llm import *
from prompt import *
import json

# openai.api_key = 'your-openai-api-key'

model = ChatGPT(api_key="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")
ok_factors = []

# 定义生成因子的任务
def generate_new_factors():
    prompt = factor_generation
    # response = model.generate_reply(prompt)
    response = """
```json
{
  "优化因子列表": [
    {
      "因子": "rank(sub(net_profit_incl_min_int_inc, inc_tax))",
      "改进原因": "通过使用rank对净利润和所得税差异排序，避免单一财务期指标的局限性，确保更为稳定的收益表现。"
    },
    {
      "因子": "div(qoq_diff(ebitda), tot_assets)",
      "改进原因": "将息税折旧摊销前利润的季度差异与总资产比结合，增强盈利能力的即时性分析，避免因季节性波动造成的影响。"
    },
    {
      "因子": "regress_btop(norm(oper_profit))",
      "改进原因": "对营业利润进行标准化并对估值因子进行中性化处理，以减少市场价值变动带来的干扰。"
    },
    {
      "因子": "mul(yoy_ratio(oper_rev), regress_liquidity(s_fa_eps_basic))",
      "改进原因": "结合营业收入的同比增长率和基本每股收益的流动性中性化，强化收入增长和市场流动性的影响分析。"
    },
    {
      "因子": "norm(add(tot_cur_assets, intang_assets))",
      "改进原因": "通过规范化流动资产和无形资产的总和，旨在减少资产波动时的非正常表现，增强普适性。"
    },
    {
      "因子": "sub(ts_max(free_cash_flow), yoy_diff(tot_liab))",
      "改进原因": "最大化企业自由现金流同时对比负债的同比差异，旨在提升对企业财务健康的灵敏度。"
    },
    {
      "因子": "mul(cs_regress_residual(analyst_sentiment, price_change_quarterly), div(oper_profit, fin_exp))",
      "改进原因": "对分析师情感和季度价格变化做线性回归残差，并与营业利润对财务费用比率结合，进一步挖掘市场预期与实际表现的差异。"
    },
    {
      "因子": "rank(mul(qtr_diff(net_profit_after_ded_nr_lp), div(1, price_std_quarterly)))",
      "改进原因": "结合净利润季度差异与价格波动的倒数，通过排序强化收益稳定对价格波动的抵抗力。"
    },
    {
      "因子": "div(yoy_ratio(distributable_profit), regress_growth(tot_cur_liab))",
      "改进原因": "分配利润同比增长与流动负债的成长中性化相结合，以识别利润增长的内在风险。"
    },
    {
      "因子": "sub(norm(net_cash_flows_inv_act), cs_regress_residual(tot_assets, tot_liab))",
      "改进原因": "对投资活动净现金流进行标准化，并与资产总计和负债总计的回归残差进行调整，明确企业资本运作的真实意图和效果。"
    }
  ]
}
```
"""
    response = response.split("```json")[1].split("```")[0]
    response = json.loads(response)['优化因子列表']
    response = [item['因子'] for item in response]
    return response

# 定义生成测试代码的任务
def generate_test_code(new_factors):
    prompt = code_generation.format(new_factors)
    response = """为了实现 `rank(sub(net_profit_incl_min_int_inc, inc_tax))` 的代码，我们首先需要从给定的数据路径加载所需的数据文件，包括 `net_profit_incl_min_int_inc` 和 `inc_tax`。然后计算两者的差值，并对结果进行排名。根据你提供的示例代码格式，下面是实现该功能的函数代码：

```python
import os
import pandas as pd

def factor_cal_custom(data_path):
    # 加载所需数据
    net_profit_incl_min_int_inc = pd.read_parquet(os.path.join(data_path, "net_profit_incl_min_int_inc.h5"))
    inc_tax = pd.read_parquet(os.path.join(data_path, "inc_tax.h5"))
    
    # 计算净利润和税收的差额
    net_profit_after_tax = net_profit_incl_min_int_inc - inc_tax
    
    # 对差值进行排名，并按百分比形式表示
    factor = net_profit_after_tax.rank(axis=1, pct=True)
    
    # 移除全为空值的行
    factor = factor.dropna(how='all', axis=0)
    
    return factor
```

在这个函数中：
- `net_profit_incl_min_int_inc` 和 `inc_tax` 是从指定的数据路径中加载的两个数据帧。
- 使用 `sub` 操作计算两者的差值。
- 对得到的差值进行排名，并对排名值进行百分比化。
- 最后，返回的 `factor` 将是一个横轴为日期，纵轴为股票代码的 `DataFrame`，与示例中提供的类似。

你可以将这个函数保存并用于回测环境中。例如，调用 `test_factor` 函数来测试该因子的表现。"""
    # response = model.generate_reply(prompt)
    response = response.split("```python")[1].split("```")[0]
    return response

# 定义测试执行任务
def execute_test(test_code):
    
    result = ""
    return result

# 定义因子优化任务
def optimize_factors(cur_factor,test_result):
    prompt = feedback.format(cur_factor)
    response = model.generate_reply_with_files(prompt, files=test_result)
    return response

def is_factor_ok(result,no_advance_times):
    if no_advance_times>=3:
        return True
    abs_ic = result
    if abs_ic >= 0.03:
        return True
    return False
# 定义工作流
def build_workflow():
    # Step 1: 生成新的因子
    new_factors = generate_new_factors()[:1]
    # print(new_factors)
    for new_factor in new_factors:
        print(new_factor)
        # Step 2: 生成测试代码
        test_code = generate_test_code(new_factor)
        print(test_code)
        # break
        no_advance_times = 0
        test_result = execute_test(test_code)
        pre_ic = test_result
        if is_factor_ok(test_result, no_advance_times):
            ok_factors.append(new_factor)
            continue
        ok = False
        cur_factor = new_factor
        while not ok:
            # Step 3: 执行测试并获取结果
            optimized_factor = optimize_factors(cur_factor, test_result)
            test_code = generate_test_code(optimized_factor)
            test_result = execute_test(test_code)
            abs_ic = test_result
            if abs_ic > pre_ic:
                no_advance_times = 0
            else:
                no_advance_times += 1
            ok = is_factor_ok(test_result, no_advance_times)
            # Step 4: 优化因子
            
            # 早停机制，最重要的指标是ic，如果连续三次IC的实际结果都没有改进，需要早停；第二个条件是绝对值IC超过3%，则可以停止

    return ok_factors

# 创建 Dify 工作流
def main():
    # new_factors = generate_new_factors()
    new_factors = build_workflow()
    print(new_factors)

# 运行工作流
if __name__ == "__main__":
    main()
