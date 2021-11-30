---
title: Rebalance Trading Strategy
date: 2021-06-14 09:00:00
description: Understand concept of Rebalance Trade Profit Estimation and Backtesting System, Design Optimized Rebalancing Range
featured_image: '/images/demo.jpg'
---
{% comment %}
    {% raw %}

```python
PROJECT_LINK = 'rebalance_trading'
PATH = '/Users/touchpadthamkul/zatoDev/clone/dev'
# PATH = ''
url = 'https://zato.dev/'

# FRAMEWORK
from IPython.display import Markdown as md
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime, pytz
import numpy as np
import os

pio.renderers.default = 'colab'

def getVariableNames(variable):
    results = []
    globalVariables=globals().copy()
    for globalVariable in globalVariables:
        if id(variable) == id(globalVariables[globalVariable]):
            results.append(globalVariable)
    return results

def displayPlot(fig):
    project_id = PROJECT_LINK.replace(' ','_')
    fig_json = fig.to_json()
    fig_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date())+'-'+project_id+'_'+getVariableNames(fig)[0]
    filename = fig_name+'.html'
    if PATH != '':
        save_path = PATH + '/_includes/post-figures/'
    else:
        save_path = ''
    completeName = os.path.join(save_path, filename)
    template = """
<html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id='{1}'></div>
        <script>
            var plotly_data = {0};
            let config = {{displayModeBar: false }};
            Plotly.react('{1}', plotly_data.data, plotly_data.layout, config);
        </script>
    </body>
</html>
"""
    # write the JSON to the HTML template
    with open(completeName, 'w') as f:
        f.write(template.format(fig_json, fig_name))
    return md("{% include post-figures/" + filename + " full_width=true %}")

def displayImg(img_name):
    master_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date()) + '-' + PROJECT_LINK + '-' + img_name
    !cp -frp $img_name $master_name
    if PATH != '':     
        img_path = PATH + '/images/projects'
        !mv $master_name $img_path
        output = md("![](/images/projects/" + master_name +")")        
    else:
        img_path = PATH
        output = md("![]("+master_name +")")
    return output

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def runBrowser(url):
    url = 'https://zato.dev/blog/' + PROJECT_LINK
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("window-size=375,812")
    # browser = webdriver.Chrome('/Users/touchpadthamkul/PySelenium/chromedriver', chrome_options=chrome_options)
    browser = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=chrome_options)
    browser.get(url)

    
import ipynbname

def saveExport():        
    pynb_name = ipynbname.name() +'.ipynb'
    md_name = ipynbname.name() +'.md'
    if PATH != '':
        selected = int(input('1 posts \n2 projects\n'))
        if selected != 1:
            folder = '/_projects'
        else:
            folder = '/_posts'
        post_path = PATH + folder
    else:
        post_path = ''
    master_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date()) + '-' + PROJECT_LINK + '.md'
    !jupyter nbconvert --to markdown $pynb_name
    !mv $md_name $master_name
    !mv $master_name $post_path

# saveExport()
runBrowser(url)
```

    [WDM] - ====== WebDriver manager ======
    [WDM] - Current google-chrome version is 91.0.4472
    [WDM] - Get LATEST driver version for 91.0.4472


    
    


    [WDM] - Get LATEST driver version for 91.0.4472
    [WDM] - Trying to download new driver from https://chromedriver.storage.googleapis.com/91.0.4472.101/chromedriver_mac64.zip
    [WDM] - Driver has been saved in cache [/Users/touchpadthamkul/.wdm/drivers/chromedriver/mac64/91.0.4472.101]
    <ipython-input-206-075f41d0670a>:79: DeprecationWarning:
    
    use options instead of chrome_options
    



```python
!pip install dateparser

import pandas as pd
import requests
import datetime
import dateparser
import pytz
```

    Requirement already satisfied: dateparser in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (1.0.0)
    Requirement already satisfied: tzlocal in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (from dateparser) (2.1)
    Requirement already satisfied: pytz in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (from dateparser) (2020.1)
    Requirement already satisfied: python-dateutil in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (from dateparser) (2.8.1)
    Requirement already satisfied: regex!=2019.02.19 in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (from dateparser) (2020.11.13)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/sekai/lib/python3.8/site-packages (from python-dateutil->dateparser) (1.15.0)



```python
# requires dateparser package
def date_to_milliseconds(date_str):
  # get epoch value in UTC
  epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
  # parse our date string
  if type(date_str) == int:
    return date_str
  elif type(date_str) != datetime.datetime:
    d = dateparser.parse(date_str)
  else:
    d = date_str
  # if the date is not timezone aware apply UTC timezone
  if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
      d = d.replace(tzinfo=pytz.utc)
  # return the difference in time
  return int((d - epoch).total_seconds() * 1000.0)

def subData(market, start_date, end_date, interval):
  tick_interval = interval
  start_time = date_to_milliseconds(start_date)
  end_time = date_to_milliseconds(end_date)
  column = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
  url = f'https://api.binance.com/api/v3/klines?symbol={market}&interval={tick_interval}&startTime={start_time}&endTime={end_time}&limit=1000'
  # url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval'startTime=
  data = requests.get(url).json()
  raw_df = pd.DataFrame(data, columns=column).reset_index(drop=True)
  raw_df['Close'] = raw_df['Close'].astype(float)
  raw_df['datetime'] = pd.to_datetime(raw_df['Close time'], unit='ms')
  return raw_df.to_dict('records')

def getData(market, start_date, end_date, interval):
  new_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
  runner = True
  all_data = []
  while runner:
    temp_data = subData(market, new_start_date, end_date, interval)
#     print(pd.DataFrame(temp_data).datetime.min(), pd.DataFrame(temp_data).datetime.max())
    new_start_date = temp_data[-1].get('Close time')
    all_data.extend(temp_data)
    if temp_data[-1].get('datetime') > datetime.datetime.strptime(end_date, '%Y-%m-%d'):
      runner = False
  return pd.DataFrame(all_data)

```
    {% endraw %}
{% endcomment %}
# Rebalance Strategy

บทความนี้เป็นบทความแรกสำหรับการบันทึกการทดสอบคริปโตบอท เพราะ จะไม่ได้อะไรเลย ถ้าบทเรียนต่างๆไม่ได้ถูกนำมาบันทึกและปรับปรุงผลให้ดีขึ้น

### เข้าเรื่องเลย ...

ผมชอบเรียนรู้จากคนอื่น หลังจากช่วงตลาดขาขึ้น และ ขาลงของคริปโตในช่วงต่างๆ หนึ่งในบทเรียนสำคัญที่หลายท่านได้แชร์ไว้คือ การจัดการบริหารเงินทุน หรือ Money Management

### Rebalance Strategy คืออะไร ?

หลังจากเห็นรีวิวจากกลุ่ม ที่ได้ทดลองเช่าบอท สำหรับเทรดคริปโต น่าสนใจดี แต่มันจะได้กำไรจริงมั้ยอันนี้เราต้องพิสูจน์ด้วยตัวเอง

Rebalancing คือเทคนิคการจัดการสัดส่วนการลงทุนประเภทหนึ่งปรับสัดส่วนลงทุน โดยทำการซื้อ หรือ ขาย ตามเงื่อนไขที่กำหนด

### ประเภทของการ Rebalance

1. Calendar Rebalancing
รูปแบบพื้นฐานของการ Rebalance โดยทำการปรับพอร์ตจากการกำหนดช่วงระยะเวลา (Time Intervals) 
ตัวอย่างเช่น กำหนดรอบสำหรับการปรับพอร์ต BTC/USDT ทุก 1 วัน 
2. Percentage-of-Portfolio Rebalancing โดยทำการปรับพอร์ตจากสัดส่วนการลงทุนของพอร์ต
ตัวอย่างเช่น ตั้งเป้าการปรับพอร์ตโดยกำหนดให้ซื้อเมื่อราคาลดลง และขายทันที เมื่อราคาเพิ่มขึ้น

### เริ่มการทดลอง

อันดับแรกเราจะทำการ Simulation การ Rebalance โดยอ้างอิงจากราคาจริงของบิทคอยน์ BTC จากข้อมูลตั้งแต่วันที่ มกราคม 2021 ถึง มิถุนายน 2021 โดยกำหนด Time Intervals คือ 5 นาที



```python
market = 'BTCBUSD'
interval = '5m'
start_date = '2021-01-01'
end_date = '2021-06-14'

df = getData(market, start_date, end_date, interval)

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 47114 entries, 0 to 47113
    Data columns (total 13 columns):
     #   Column                        Non-Null Count  Dtype         
    ---  ------                        --------------  -----         
     0   Open time                     47114 non-null  int64         
     1   Open                          47114 non-null  object        
     2   High                          47114 non-null  object        
     3   Low                           47114 non-null  object        
     4   Close                         47114 non-null  float64       
     5   Volume                        47114 non-null  object        
     6   Close time                    47114 non-null  int64         
     7   Quote asset volume            47114 non-null  object        
     8   Number of trades              47114 non-null  int64         
     9   Taker buy base asset volume   47114 non-null  object        
     10  Taker buy quote asset volume  47114 non-null  object        
     11  Ignore                        47114 non-null  object        
     12  datetime                      47114 non-null  datetime64[ns]
    dtypes: datetime64[ns](1), float64(1), int64(3), object(8)
    memory usage: 4.7+ MB


จากข้อมูลข้างต้นทำให้ทราบว่าเรามีข้อมูลทั้งหมด 47,114 แถวเพื่อทำการทดสอบ โดยหลักๆเราจะใช้ข้อมูลราคาปิด และวันที่


```python
col = ['datetime', 'Close']
df.filter(col).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01 00:04:59.999</td>
      <td>29013.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-01 00:09:59.999</td>
      <td>28900.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-01 00:14:59.999</td>
      <td>28792.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-01 00:19:59.999</td>
      <td>28870.37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-01 00:24:59.999</td>
      <td>28882.74</td>
    </tr>
  </tbody>
</table>
</div>




```python

def rebalanceTrade(input_busd_unit, df):
    price_list = df.Close.tolist()
    now_busd_unit = input_busd_unit
    now_btc_unit = 0
    trade_history = []

    for i,price in enumerate(price_list):
        trade_dict = {}
        now_btc_value = now_btc_unit * price
        now_busd_value = now_busd_unit * 1

        balancer = (now_busd_value + now_btc_value) / 2

        btc_diff_value = (now_btc_value - balancer)
        btc_diff_unit = btc_diff_value * (-1/price)

        if btc_diff_unit > 0:
            action = 'BUY'
        elif btc_diff_unit < 0:
            action = 'SELL'
        else:
            action = 'HODL'

        action_value = btc_diff_unit * -1*price
        now_hodl_value = now_btc_value + now_busd_value
        if len(trade_history) > 0:
            buy_n_hold = 100 * (price - trade_history[0].get('btc_price')) / trade_history[0].get('btc_price')
        else:
            buy_n_hold = 0
        
        trade_dict['btc_price'] = price
        trade_dict['now_btc_unit'] = now_btc_unit
        trade_dict['now_btc_value'] = now_btc_value
        trade_dict['now_busd_unit'] = now_busd_unit
        trade_dict['now_busd_value'] = now_busd_unit * 1
        trade_dict['now_hodl_value'] = now_hodl_value
        trade_dict['btc_diff_value'] = btc_diff_value
        trade_dict['action'] = action
        trade_dict['btc_diff_unit'] = btc_diff_unit
        trade_dict['action_value'] = action_value
        trade_dict['port_value'] = 100 * (now_hodl_value - input_busd_unit) / input_busd_unit
        
        trade_dict['hodl_value'] = (buy_n_hold)
        now_busd_unit = now_busd_unit + action_value
        now_btc_unit = now_btc_unit + btc_diff_unit


        trade_history.append(trade_dict)


    trade_df = pd.DataFrame(trade_history)
    trade_df['datetime'] = df['datetime']
    trade_df['port_gain'] = trade_df.port_value.pct_change(periods=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    trade_df['btc_gain'] = trade_df['btc_price'].pct_change().fillna(0)
    return trade_df


input_busd_unit = 100

trade_df = rebalanceTrade(input_busd_unit, df)

trade_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>btc_price</th>
      <th>now_btc_unit</th>
      <th>now_btc_value</th>
      <th>now_busd_unit</th>
      <th>now_busd_value</th>
      <th>now_hodl_value</th>
      <th>btc_diff_value</th>
      <th>action</th>
      <th>btc_diff_unit</th>
      <th>action_value</th>
      <th>port_value</th>
      <th>hodl_value</th>
      <th>datetime</th>
      <th>port_gain</th>
      <th>btc_gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7897.59</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>-50.000000</td>
      <td>BUY</td>
      <td>6.331045e-03</td>
      <td>-50.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2020-01-07 00:29:59.999</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7886.00</td>
      <td>0.006331</td>
      <td>49.926623</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>99.926623</td>
      <td>-0.036688</td>
      <td>BUY</td>
      <td>4.652347e-06</td>
      <td>-0.036688</td>
      <td>-0.073377</td>
      <td>-0.146754</td>
      <td>2020-01-07 00:59:59.999</td>
      <td>0.000000</td>
      <td>-0.001468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7903.22</td>
      <td>0.006336</td>
      <td>50.072412</td>
      <td>49.963312</td>
      <td>49.963312</td>
      <td>100.035724</td>
      <td>0.054550</td>
      <td>SELL</td>
      <td>-6.902295e-06</td>
      <td>0.054550</td>
      <td>0.035724</td>
      <td>0.071288</td>
      <td>2020-01-07 01:29:59.999</td>
      <td>-1.486855</td>
      <td>0.002184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7903.30</td>
      <td>0.006329</td>
      <td>50.018368</td>
      <td>50.017862</td>
      <td>50.017862</td>
      <td>100.036230</td>
      <td>0.000253</td>
      <td>SELL</td>
      <td>-3.203115e-08</td>
      <td>0.000253</td>
      <td>0.036230</td>
      <td>0.072301</td>
      <td>2020-01-07 01:59:59.999</td>
      <td>0.014173</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7926.54</td>
      <td>0.006329</td>
      <td>50.165196</td>
      <td>50.018115</td>
      <td>50.018115</td>
      <td>100.183311</td>
      <td>0.073540</td>
      <td>SELL</td>
      <td>-9.277721e-06</td>
      <td>0.073540</td>
      <td>0.183311</td>
      <td>0.366568</td>
      <td>2020-01-07 02:29:59.999</td>
      <td>4.059609</td>
      <td>0.002941</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

gph = go.Scatter(
    x = trade_df['datetime'],
    y = trade_df['port_value'],
#     y = trade_df['port_gain'],
    mode = 'lines',
    name = 'port_value'
)

gph1 = go.Scatter(
    x = trade_df['datetime'],
    y = trade_df['hodl_value'],
#     y = trade_df['port_gain'],
    mode = 'lines',
    name = 'hodl_value'
)

gph2 = go.Scatter(
    x = trade_df['datetime'],
    y = trade_df['btc_price'],
#     y = trade_df['btc_gain'],
    mode = 'lines',
    name = 'btc_price'
)


fig.add_trace(gph, secondary_y=True)
fig.add_trace(gph1, secondary_y=True)
fig.add_trace(gph2, secondary_y=False)

fig.update_layout(template='plotly_dark')

displayPlot(fig)
```




{% include post-figures/2021-06-14-rebalance_trading_fig.html full_width=true %}


