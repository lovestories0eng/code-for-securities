import datetime
import numpy as np
import math
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

def getStockWeights(symbol, date, advantageStockList):
    indexWeights = get_index_weight(symbol,date)
    stockMapWeight = {}
    sum = 0
    for i in range(len(indexWeights)):
        # 选出在股票池中的股票
        if indexWeights["symbol"][i] in stockList:
            sum += indexWeights["weight"][i]
            # 选出优势股
            if indexWeights["symbol"][i] in advantageStockList:
                stockMapWeight[indexWeights["symbol"][i]] = indexWeights["weight"][i]
    for i in stockMapWeight:
        stockMapWeight[i] = stockMapWeight[i] / sum
    return stockMapWeight
    
def adaBoost(X, y):
    # 处理nan值
    X[np.isnan(X)] = X[~np.isnan(X)].mean()
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), random_state=0, n_estimators=300)
    model.fit(X, y)
    return model
    
# 全局参数
yzlist = ["dividend_rate_12_months","micd","vroc","tapi","vma","macd","vosc","equity_ratio","vstd","vrsi","cvlt","roc","vr","net_profit_div_income","capitalization","net_profit_margin_on_sales","net_cashflow_from_opt_act_growth_ratio","srdm","dbcd","circulating_cap"]

# 设置回测开始和结束时间
g.start_date = '20170101'
g.end_date = '20220615'

# 模型训练的开始时间为回测时间的前两个月
g.train_start_date = get_trade_days(None, datetime.datetime.strptime(g.start_date, '%Y%m%d').strftime("%Y%m%d"), 61)[0]

g.train_end_date = datetime.datetime.strptime(g.end_date, '%Y%m%d')
    
trade_days = get_trade_days(g.train_start_date.strftime("%Y%m%d"), g.train_end_date.strftime("%Y%m%d"), None).strftime('%Y-%m-%d')

stockList= get_index_stocks('000300.SH',g.train_start_date)
g.advantageStockList = stockList
g.disadvantageStockList = []

# 设置持仓周期
holding_period = 10
g.period = 10
g.day = 0

q = query(
    factor.symbol,
    factor.date,
    factor.dividend_rate_12_months,
    factor.micd,
    factor.vroc,
    factor.tapi,
    factor.vma,
    factor.macd,
    factor.vosc,
    factor.equity_ratio,
    factor.vstd,
    factor.vrsi,
    factor.cvlt,
    factor.roc,
    factor.vr,
    factor.net_profit_div_income,
    factor.capitalization,
    factor.net_profit_margin_on_sales,
    factor.net_cashflow_from_opt_act_growth_ratio,
    factor.srdm,
    factor.dbcd,
    factor.circulating_cap
    ).filter(
        factor.symbol.in_(stockList),
        factor.date.in_(trade_days)
    )

df = get_factors(q)
# 建立全局字典，为每一支股票存储相应的因子数据
stockData = {}

for i in stockList:
    stockData[i] = df[df["factor_symbol"] == i]
    stockData[i] = stockData[i].drop("factor_symbol", axis = 1)
    stockData[i].index = stockData[i]["factor_date"]
    stockData[i] = stockData[i].drop("factor_date", axis = 1)
    
# 获取所有股票的收益值
prices = get_price(stockList, g.train_start_date, g.train_end_date + datetime.timedelta(days = 1) * 2 * holding_period, '1d', ['close', 'is_paused'], skip_paused = False, fq = 'pre',is_panel = 0)

# 记录退市的股票及退市前一天的时间
stockDelist = {}
y = []
# 遍历所有日期
for i in stockData:
    tmp = []
    # 价格与交易日期对应的日期应该相等
    for j in range(len(stockData[i])):
        try:
            tmp.append((prices[i]["close"][j + holding_period] - prices[i]["close"][j]) / prices[i]["close"][j])
        except:
            # 记录退市的股票数据
            stockDelist[i] = prices[i].index[j-1]
            tmp.append(0)
    stockData[i]["reward"] = np.array(tmp)

for i in stockDelist:
    length = len(stockList)
    for j in range(length):
        if stockList[j] == i:
            del stockList[j]
            break
print(len(stockList))

stockChosen = 12
steps = stockChosen / 4
    
stockMapReward = {}
stockMapModel = {}
stockMapMarketValue = {}
stockToTrend = []

# 股票策略模版
# 初始化函数,全局只运行一次
def init(context):
    # 设置基准收益：沪深300指数
    set_benchmark('000300.SH')
    # 打印日志
    log.info('策略开始运行,初始化函数全局只运行一次')
    # 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
    set_commission(PerShare(type='stock',cost=0.0002))
    # 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
    set_slippage(PriceSlippage(0.005))
    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    # 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
    # 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
    set_volume_limit(0.25,0.5)
    context.securitys = stockList
    context.factors = yzlist
    # 回测区间、初始资金、运行频率请在右上方设置
    # 取前M日的RSRS斜率时间序列。（M = 600）
    # 取前N日最高价与最低价序列。（N = 18）
    context.M = 600
    context.N = 18
    context.S1=1.0
    context.S2=-0.7
    # 回测区间、初始资金、运行频率请在右上方设置
    
def get_RSRS(stock, n, m):
    values = history(stock, ['high', 'low'], n + m - 1, '1d', skip_paused=True)
    high_array = values.high.values
    low_array = values.low.values
    scores = []  # 各期斜率
    for i in range(m):
        high = high_array[i:i + 18]
        low = low_array[i:i + 18]
        # 计算单期斜率
        x = low  # low 作为自变量
        if len(x) == 0:
            continue
        try:
            X = sm.add_constant(x)  # 添加常数变量
        except:
            log.info(x)
        y = high  # high 作为因变量
        #将两个序列进行OLS线性回归。
        model = sm.OLS(y, X)  # 最小二乘法
        results = model.fit()
        try:
            score = results.params[1]
        except:
            score = 1
        scores.append(score)
        # 记录最后一期的 Rsquared(可决系数)
    if i == m - 1:
        R_squared = results.rsquared
    scores = np.array(scores)
    # 最近期的标准分
    z_score = (scores[-1] - scores.mean()) / scores.std()
    # RSRS 得分
    RSRS_socre = z_score * R_squared
    return RSRS_socre

#每日开盘前9:00被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):
    # 获取日期
    date = get_datetime().strftime('%Y%m%d')
    # 打印日期
    log.info('{} 盘前运行'.format(date))

## 开盘时运行函数
def handle_bar(context, bar_dict):
    # 获取时间
    time = get_datetime().strftime('%Y%m%d')
    # 打印时间
    log.info('{} 盘中运行'.format(time))

    # 每隔1天重新训练模型， 每隔一段时期根据权重重新分配优势股持仓
    # if g.day % 1 == 0:
    #     # 利用AdaBoost进行预测
    #     for i in stockList:
    #         X = np.array(stockData[i].iloc[g.day:g.day + 60, 0:20])
    #         y = np.array(stockData[i].iloc[g.day:g.day + 60, 20])
    #         stockMapModel[i] = adaBoost(X, y)
            
    if g.day % g.period == 0:
        if g.day != 0:
            # 求出每一支股票在这十天中的趋势
            for i in stockList:
                stockToTrend.append((stockData[i].iloc[g.day + 60, 20] / stockData[i].iloc[g.day + 50, 20]) - 1)
            # 按趋势从小到大排序
            sortedStocks = [i for _,i in sorted(zip(stockToTrend, stockList))]
            # 趋势最高的当作优势股
            g.advantageStockList = sortedStocks[-stockChosen:]
            # 趋势最低的当作劣势股票
            g.disadvantageStockList = sortedStocks[0:stockChosen]

        stockWeights = getStockWeights("000300.SH", g.train_start_date, g.advantageStockList)
        for i in g.advantageStockList:
            stockMarketValue = context.portfolio.stock_account.positions[i].market_value
            # 根据市场占比调整优势股的股数
            order_value(i, 7000000 * stockWeights[i] - stockMarketValue)
            # order_target_percent(i, stockWeights[i])
    else:
        stockValueSum = 0
        for i in stockList:
            X = np.array(stockData[i].iloc[g.day + 60, 0:20])
            X[np.isnan(X)] = X[~np.isnan(X)].mean()
            X = X.tolist()
            # 由于持仓期是十天，照理说前面九天的数据是不能用的，但先不管
            # stockMapReward[i] = stockMapModel[i].predict([X])
            stockMapReward[i] = stockData[i].iloc[g.day + 60, 20]
            stockMapMarketValue[i] = context.portfolio.stock_account.positions[i].market_value
            # stockMapTrendData[i].append(stockMapReward[i])
        # 找出前五支优势股和前五支劣势股
        stocks = list(stockMapReward.keys())
        rewards = list(stockMapReward.values())
        # 收益率从小到大排序
        sortedStocks = [i for _,i in sorted(zip(rewards, stocks))]
        # 收益率前几当作优势股
        g.advantageStockList = sortedStocks[-stockChosen:]
        # 收益率后几当作劣势股票
        g.disadvantageStockList = sortedStocks[0:stockChosen]
        count = 1
        # 优势股提升权重
        for positive in g.advantageStockList:
            order_value(positive, (stockMapMarketValue[positive] * count) / 15)
            count += 1           

        # 劣势股票降低权重
        for negative in g.disadvantageStockList:
            count -= 1
            order_value(positive, -(stockMapMarketValue[negative] * (count // steps + 1)) / 10)
        
        count = 1
        # 计算 RSRS
        # S(buy)=0.7,S(sell)=−0.7
        for positive in g.advantageStockList:
            # 将拟合后的β值作为当日RSRS斜率指标值。
            hz300_RSRS= get_RSRS(positive, context.N, context.M)
            # 4、当RSRS斜率大于S(buy)时，全仓买入，小于S(sell)时，卖出平仓。
            # 如果大于0.7 股票降低权重
            if hz300_RSRS > context.S1:
                order_value(positive,(stockMapMarketValue[positive]*((-count+1)/(10+count))))
            # 如果小于于0.7 股票降低权重
            if hz300_RSRS < -context.S2:
                order_value(positive,(stockMapMarketValue[positive]*((-count)/(10+count))))
            count +=1


## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
def after_trading(context):
    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印时间
    log.info('{} 盘后运行'.format(time))
    # 时间自增
    g.day += 1
    log.info('一天结束')