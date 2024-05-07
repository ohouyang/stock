from collections import OrderedDict

## spread, slippage, tranct, tranct.ratio
product_info = OrderedDict()

## http://www.qhsxf.com/

product_info["FG"] = OrderedDict([("spread", 1), ("tranct", 0.155), ("tranct.ratio", False), ("multiplier", 20), ("close", 0.155)])
## tc=3.1 yuan, 1 pt=20 yuan, so tranct=3.1/20=0.155

product_info["MA"] = OrderedDict([("spread", 1), ("tranct", 0.21), ("tranct.ratio", False), ("multiplier", 10), ("close", 0.21)])
## tc=2.1 yuan, 1pt=10 yuan, so tranct=2.1/10=0.21

## SM and SF don't have night session so we don't list them here

product_info["OI"] = OrderedDict([("spread", 2), ("tranct", 0.21), ("tranct.ratio", False), ("multiplier", 10), ("close", 0)])
## tc=2.1 yuan, 1pt=10 yuan, intraday is zero, but we usually open in the afternoon
## and close at night, they are not the same trading day,so we still use 2.1

product_info["SR"] = OrderedDict([("spread", 1), ("tranct", 0.31), ("tranct.ratio", False), ("multiplier", 10), ("close", 0)])
## tc=3.1 yuan, 1pt=10 yuan, intraday is zero, tc=3.1/10=0.31

product_info["TA"] = OrderedDict([("spread", 2), ("tranct", 0.62), ("tranct.ratio", False), ("multiplier", 5), ("close", 0)])
## tc=3.1 yuan, 1pt=5 yuan, intraday is zero, tc=3.1/5=0.62

## WH is not liquid enough for intraday trading

## CY doesn't have enough history

product_info["CF"] = OrderedDict([("spread", 5), ("tranct", 1.3), ("tranct.ratio", False), ("multiplier", 5), ("close", 0)])
## tc=4.4 yuan, 1pt=5 yuan, intraday is zero, tranct=4.4/5=0.88

product_info["RM"] = OrderedDict([("spread", 1), ("tranct", 0.16), ("tranct.ratio", False), ("multiplier", 10), ("close", 0)])
## tc=1.6 yuan, 1pt=10 yuan, intraday is zero, tract=1.6/10=0.16

## AP doesn't have long enough history

product_info["ZC"] = OrderedDict([("spread", 0.2), ("tranct", 0.041), ("tranct.ratio", False), ("multiplier", 100), ("close", 0.041)])
## tc=4.1 yuan, 1pt=100yuan, tranct=4.1/100=0.041

## JR, LR, PM, RI, RS are not liquid enough for intraday trading

product_info["j"] = OrderedDict([("spread", 0.5), ("tranct", 1.9e-4), ("tranct.ratio", True), ("multiplier", 100), ("close", 1.9e-4)])

product_info["jm"] = OrderedDict([("spread", 0.5), ("tranct", 1.9e-4), ("tranct.ratio", True), ("multiplier", 60), ("close", 1.9e-4)])

product_info["p"] = OrderedDict([("spread", 2), ("tranct", 0.26), ("tranct.ratio", False), ("multiplier", 10), ("close", 0.135)])
## tc=2.6yuan, 1pt=10 yuan, intraday is 1.35 yuan, tc=2.6/10=0.26

product_info["y"] = OrderedDict([("spread", 2), ("tranct", 0.26), ("tranct.ratio", False), ("multiplier", 10), ("close", 0.135)])
## tc=2.6 yuan, 1pt=10 yuan, intraday is 1.35 yuan, tc=2.6/10=0.26

## cs, l, pp don't have night session

product_info["m"] = OrderedDict([("spread", 1), ("tranct", 0.16), ("tranct.ratio", False), ("multiplier", 10), ("close", 0.085)])
## tc=1.6 yuan, 1pt=10 yuan, intraday is 0.85 yuan, tc=1.6/10=0.16

## bb and fb don't have enough liqiuidity for intraday trading

## v,c doesn't have night section

product_info["a"] = OrderedDict([("spread", 1), ("tranct", 0.21), ("tranct.ratio", False), ("multiplier", 10), ("close", 0.21)])
## tc=2.1 yuan 1pt=10 yuan, tc=2.1/10=0.21

## b doesn't have enough liquidity for intraday trading

product_info["i"] = OrderedDict([("spread", 0.5), ("tranct", 1.9e-4), ("tranct.ratio", True), ("multiplier", 100), ("close", 1.9e-4)])

product_info["jd"] = OrderedDict([("spread", 1), ("tranct", 1.55e-4), ("tranct.ratio", True), ("multiplier", 10), ("close", 1.55e-4)])

product_info["cu"] = OrderedDict([("spread", 10), ("tranct", 0.55e-4), ("tranct.ratio", True), ("multiplier", 5), ("close", 0.55e-4)])

product_info["zn"] = OrderedDict([("spread", 5), ("tranct", 0.62), ("tranct.ratio", False), ("multiplier", 5), ("close", 0)])
## tc=3.1 yuan, 1pt=5 yuan, intraday is zero, tc=3.1/5=0.62

product_info["al"] = OrderedDict([("spread", 10), ("tranct", 0.62), ("tranct.ratio", False), ("multiplier", 5), ("close", 0)])
## tc=3.1 yuan, 1pt=5, intraday is zero, tc=3.1/5=0.62

## sn and pb are not liquid enough for intraday trading

product_info["au"] = OrderedDict([("spread", 0.05), ("tranct", 0.0101), ("tranct.ratio", False), ("multiplier", 1000), ("close", 0)])
## tc=10.1 yuan, 1pt=1000 yuan, intraday is zero, tc=10.1/1000=0.0101

product_info["rb"] = OrderedDict([("spread", 1), ("tranct", 1.05e-4), ("tranct.ratio", True), ("multiplier", 10), ("close", 1.05e-4)])

product_info["hc"] = OrderedDict([("spread", 1), ("tranct", 1.05e-4), ("tranct.ratio", True), ("multiplier", 10), ("close", 1.05e-4)])

product_info["ni"] = OrderedDict([("spread", 10), ("tranct", 6.1), ("tranct.ratio", False), ("multiplier", 1), ("close", 6.1)])
## tc=6.1 yuan, 1pt=1, tranct=6.1/1=1

product_info["ag"] = OrderedDict([("spread", 1), ("tranct", 0.55e-4), ("tranct.ratio", True), ("multiplier", 15), ("close", 0.55e-4)])

## wr is not liquid enough for intraday trading

product_info["ru"] = OrderedDict([("spread", 5), ("tranct", 0.5e-4), ("tranct.ratio", True), ("multiplier", 10), ("close", 0.5e-4)])

product_info["bu"] = OrderedDict([("spread", 2), ("tranct", 1.05e-4), ("tranct.ratio", True), ("multiplier", 10), ("close", 1.05e-4)])

product_info["IF"] = OrderedDict([("spread", 0.2), ("tranct", 0.25e-4), ("tranct.ratio", True), ("multiplier", 300), ("close", 0.25e-4)])

product_info["IH"] = OrderedDict([("spread", 0.2), ("tranct", 0.25e-4), ("tranct.ratio", True), ("multiplier", 300), ("close", 0.25e-4)])

product_info["IC"] = OrderedDict([("spread", 0.2), ("tranct", 0.25e-4), ("tranct.ratio", True), ("multiplier", 200), ("close", 0.25e-4)])

product_info["T"] = OrderedDict([("spread", 0.005), ("tranct", 3.3*1e-6), ("tranct.ratio", False), ("multiplier",1e6), ("close", 0)])

product_info["TF"] = OrderedDict([("spread", 0.005), ("tranct", 3.3*1e-6), ("tranct.ratio", False), ("multiplier",1e6), ("close", 0)])

product_info["btc.usd.td"] = OrderedDict([("spread", 0.5), ("tranct", 7.5*1e-4), ("tranct.ratio", True), ("multiplier",1), ("close", 7.5*1e-4)])

product_info["eth.usd.td"] = OrderedDict([("spread", 0.05), ("tranct", 7.5*1e-4), ("tranct.ratio", True), ("multiplier",1), ("close", 7.5*1e-4)])

product_info["btc.usdt"] = OrderedDict([("spread", 0.01), ("tranct", 7.5*1e-4), ("tranct.ratio", True), ("multiplier",1), ("close", 7.5*1e-4)])

product_info["eth.usdt"] = OrderedDict([("spread", 0.01), ("tranct", 7.5*1e-4), ("tranct.ratio", True), ("multiplier",1), ("close", 7.5*1e-4)])

## fu has too short hisotry