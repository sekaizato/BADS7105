---
title: Assignment 2 Customer Segmentation
subtitle: Clustering Algorithm by K-mean Clustering
date: 2021-12-04 00:00:00
description: Project for study relationship between transaction of supermarket data.
featured_image: 2021-12-05-assignment2-assignment2-marketbasket.jpg
accent_color: '#4C60E6'
gallery_images:
  - 2021-12-05-assignment1-assignment1-dashboard.jpg
---
{% comment %}
    {% raw %}

```python
!pwd
```

    /Users/touchpadthamkul/zatoDev/project/bads_crm_final/master/BADS7105/writing



```python
PROJECT_LINK = 'assignment2'
PATH = '/Users/touchpadthamkul/zatoDev/project/bads_crm_final/master/BADS7105'


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
        output = md("![](/BADS7105/images/projects/" + master_name +")")
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
# runBrowser(url)
```
    {% endraw %}
{% endcomment %}
##  Objective

จากโจทย์วันนี้ อาจารย์ให้จัดกลุ่มลูกค้า จากพฤติกรรมการซื้อสินค้าจากข้อมูล Supermarket Data โดยใช้ Clustering Model


```python
import pandas as pd
from pycaret.clustering import *
```


```python
df = pd.read_csv('Supermarket Data.csv')
df = df[:1000]
df['SHOP_DATE'] = df['SHOP_DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
```


```python
df.head(5)
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
      <th>SHOP_WEEK</th>
      <th>SHOP_DATE</th>
      <th>SHOP_WEEKDAY</th>
      <th>SHOP_HOUR</th>
      <th>QUANTITY</th>
      <th>SPEND</th>
      <th>PROD_CODE</th>
      <th>PROD_CODE_10</th>
      <th>PROD_CODE_20</th>
      <th>PROD_CODE_30</th>
      <th>PROD_CODE_40</th>
      <th>CUST_CODE</th>
      <th>CUST_PRICE_SENSITIVITY</th>
      <th>CUST_LIFESTAGE</th>
      <th>BASKET_ID</th>
      <th>BASKET_SIZE</th>
      <th>BASKET_PRICE_SENSITIVITY</th>
      <th>BASKET_TYPE</th>
      <th>BASKET_DOMINANT_MISSION</th>
      <th>STORE_CODE</th>
      <th>STORE_FORMAT</th>
      <th>STORE_REGION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200732</td>
      <td>2007-10-05</td>
      <td>6</td>
      <td>17</td>
      <td>3</td>
      <td>6.75</td>
      <td>PRD0900001</td>
      <td>CL00072</td>
      <td>DEP00021</td>
      <td>G00007</td>
      <td>D00002</td>
      <td>CUST0000583261</td>
      <td>UM</td>
      <td>YF</td>
      <td>994107800547472</td>
      <td>L</td>
      <td>MM</td>
      <td>Top Up</td>
      <td>Grocery</td>
      <td>STORE00001</td>
      <td>LS</td>
      <td>E02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200733</td>
      <td>2007-10-10</td>
      <td>4</td>
      <td>20</td>
      <td>3</td>
      <td>6.75</td>
      <td>PRD0900001</td>
      <td>CL00072</td>
      <td>DEP00021</td>
      <td>G00007</td>
      <td>D00002</td>
      <td>CUST0000537317</td>
      <td>MM</td>
      <td>OF</td>
      <td>994107900512001</td>
      <td>L</td>
      <td>MM</td>
      <td>Full Shop</td>
      <td>Fresh</td>
      <td>STORE00001</td>
      <td>LS</td>
      <td>E02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200741</td>
      <td>2007-12-09</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>2.25</td>
      <td>PRD0900001</td>
      <td>CL00072</td>
      <td>DEP00021</td>
      <td>G00007</td>
      <td>D00002</td>
      <td>CUST0000472158</td>
      <td>MM</td>
      <td>YF</td>
      <td>994108700468327</td>
      <td>L</td>
      <td>MM</td>
      <td>Full Shop</td>
      <td>Grocery</td>
      <td>STORE00001</td>
      <td>LS</td>
      <td>E02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200731</td>
      <td>2007-09-29</td>
      <td>7</td>
      <td>17</td>
      <td>1</td>
      <td>2.25</td>
      <td>PRD0900001</td>
      <td>CL00072</td>
      <td>DEP00021</td>
      <td>G00007</td>
      <td>D00002</td>
      <td>CUST0000099658</td>
      <td>LA</td>
      <td>OF</td>
      <td>994107700237811</td>
      <td>L</td>
      <td>LA</td>
      <td>Full Shop</td>
      <td>Mixed</td>
      <td>STORE00001</td>
      <td>LS</td>
      <td>E02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200737</td>
      <td>2007-11-10</td>
      <td>7</td>
      <td>14</td>
      <td>3</td>
      <td>6.75</td>
      <td>PRD0900001</td>
      <td>CL00072</td>
      <td>DEP00021</td>
      <td>G00007</td>
      <td>D00002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>994108300002212</td>
      <td>L</td>
      <td>MM</td>
      <td>Full Shop</td>
      <td>Fresh</td>
      <td>STORE00001</td>
      <td>LS</td>
      <td>E02</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_csv = df[df['CUST_CODE'].notnull()].groupby(["CUST_CODE"]).agg(TotalSpend=('SPEND', 'sum'),
                                                        TotalVisits=('BASKET_ID', 'nunique'),
                                                        TotakSKUs=('PROD_CODE', 'nunique'),
                                                        FirstDate=('SHOP_DATE', 'min'),
                                                        LastDate=('SHOP_DATE', 'max')).reset_index()
df_csv.head(5)
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
      <th>CUST_CODE</th>
      <th>TotalSpend</th>
      <th>TotalVisits</th>
      <th>TotakSKUs</th>
      <th>FirstDate</th>
      <th>LastDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST0000005057</td>
      <td>1.20</td>
      <td>1</td>
      <td>1</td>
      <td>2006-10-11</td>
      <td>2006-10-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST0000012784</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-15</td>
      <td>2006-09-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST0000016697</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-10</td>
      <td>2007-02-23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST0000017325</td>
      <td>21.24</td>
      <td>10</td>
      <td>1</td>
      <td>2006-08-09</td>
      <td>2008-07-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST0000019732</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2008-03-12</td>
      <td>2008-03-12</td>
    </tr>
  </tbody>
</table>
</div>




```python
##calculate ticket size
df_csv['TicketSize'] = df_csv['TotalSpend']/df_csv['TotalVisits']
df_csv.head(5)
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
      <th>CUST_CODE</th>
      <th>TotalSpend</th>
      <th>TotalVisits</th>
      <th>TotakSKUs</th>
      <th>FirstDate</th>
      <th>LastDate</th>
      <th>TicketSize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST0000005057</td>
      <td>1.20</td>
      <td>1</td>
      <td>1</td>
      <td>2006-10-11</td>
      <td>2006-10-11</td>
      <td>1.200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST0000012784</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-15</td>
      <td>2006-09-29</td>
      <td>1.770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST0000016697</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-10</td>
      <td>2007-02-23</td>
      <td>1.770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST0000017325</td>
      <td>21.24</td>
      <td>10</td>
      <td>1</td>
      <td>2006-08-09</td>
      <td>2008-07-04</td>
      <td>2.124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST0000019732</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2008-03-12</td>
      <td>2008-03-12</td>
      <td>1.770</td>
    </tr>
  </tbody>
</table>
</div>




```python
##find max date in the dataset
max_date = df_csv['LastDate'].max()
max_date
```




    Timestamp('2008-07-05 00:00:00')




```python
df_csv['total_days'] = (df_csv['LastDate'] - df_csv['FirstDate']).dt.days + 1
```


```python
df_csv['recency'] = (max_date - df_csv['LastDate']).dt.days
```

## Cluster Customers


```python
from sklearn.cluster import KMeans
import numpy as np
```


```python
df_csv
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
      <th>CUST_CODE</th>
      <th>TotalSpend</th>
      <th>TotalVisits</th>
      <th>TotakSKUs</th>
      <th>FirstDate</th>
      <th>LastDate</th>
      <th>TicketSize</th>
      <th>total_days</th>
      <th>recency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST0000005057</td>
      <td>1.20</td>
      <td>1</td>
      <td>1</td>
      <td>2006-10-11</td>
      <td>2006-10-11</td>
      <td>1.200</td>
      <td>1</td>
      <td>633</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST0000012784</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-15</td>
      <td>2006-09-29</td>
      <td>1.770</td>
      <td>15</td>
      <td>645</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST0000016697</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-10</td>
      <td>2007-02-23</td>
      <td>1.770</td>
      <td>167</td>
      <td>498</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST0000017325</td>
      <td>21.24</td>
      <td>10</td>
      <td>1</td>
      <td>2006-08-09</td>
      <td>2008-07-04</td>
      <td>2.124</td>
      <td>696</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST0000019732</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2008-03-12</td>
      <td>2008-03-12</td>
      <td>1.770</td>
      <td>1</td>
      <td>115</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>CUST0000994485</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2006-04-14</td>
      <td>2006-04-14</td>
      <td>1.770</td>
      <td>1</td>
      <td>813</td>
    </tr>
    <tr>
      <th>339</th>
      <td>CUST0000995305</td>
      <td>11.06</td>
      <td>4</td>
      <td>2</td>
      <td>2007-08-01</td>
      <td>2008-06-10</td>
      <td>2.765</td>
      <td>315</td>
      <td>25</td>
    </tr>
    <tr>
      <th>340</th>
      <td>CUST0000997122</td>
      <td>12.12</td>
      <td>6</td>
      <td>1</td>
      <td>2006-09-23</td>
      <td>2008-06-11</td>
      <td>2.020</td>
      <td>628</td>
      <td>24</td>
    </tr>
    <tr>
      <th>341</th>
      <td>CUST0000999024</td>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>2006-12-17</td>
      <td>2006-12-17</td>
      <td>1.010</td>
      <td>1</td>
      <td>566</td>
    </tr>
    <tr>
      <th>342</th>
      <td>CUST0000999935</td>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>2007-06-30</td>
      <td>2007-06-30</td>
      <td>1.010</td>
      <td>1</td>
      <td>371</td>
    </tr>
  </tbody>
</table>
<p>343 rows × 9 columns</p>
</div>




```python
df_model = df_csv.drop(columns=["CUST_CODE", "FirstDate", "LastDate"])
df_model
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
      <th>TotalSpend</th>
      <th>TotalVisits</th>
      <th>TotakSKUs</th>
      <th>TicketSize</th>
      <th>total_days</th>
      <th>recency</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.20</td>
      <td>1</td>
      <td>1</td>
      <td>1.200</td>
      <td>1</td>
      <td>633</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>1.770</td>
      <td>15</td>
      <td>645</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>1.770</td>
      <td>167</td>
      <td>498</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.24</td>
      <td>10</td>
      <td>1</td>
      <td>2.124</td>
      <td>696</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>1.770</td>
      <td>1</td>
      <td>115</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>1.770</td>
      <td>1</td>
      <td>813</td>
      <td>2</td>
    </tr>
    <tr>
      <th>339</th>
      <td>11.06</td>
      <td>4</td>
      <td>2</td>
      <td>2.765</td>
      <td>315</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>340</th>
      <td>12.12</td>
      <td>6</td>
      <td>1</td>
      <td>2.020</td>
      <td>628</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341</th>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>1.010</td>
      <td>1</td>
      <td>566</td>
      <td>2</td>
    </tr>
    <tr>
      <th>342</th>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>1.010</td>
      <td>1</td>
      <td>371</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>343 rows × 7 columns</p>
</div>




```python
kmeans = KMeans(n_clusters=3, random_state=42).fit(df_model)
```


```python
df_csv["group"] = [str(g) for g in kmeans.labels_]
df_csv
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
      <th>CUST_CODE</th>
      <th>TotalSpend</th>
      <th>TotalVisits</th>
      <th>TotakSKUs</th>
      <th>FirstDate</th>
      <th>LastDate</th>
      <th>TicketSize</th>
      <th>total_days</th>
      <th>recency</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST0000005057</td>
      <td>1.20</td>
      <td>1</td>
      <td>1</td>
      <td>2006-10-11</td>
      <td>2006-10-11</td>
      <td>1.200</td>
      <td>1</td>
      <td>633</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST0000012784</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-15</td>
      <td>2006-09-29</td>
      <td>1.770</td>
      <td>15</td>
      <td>645</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST0000016697</td>
      <td>3.54</td>
      <td>2</td>
      <td>1</td>
      <td>2006-09-10</td>
      <td>2007-02-23</td>
      <td>1.770</td>
      <td>167</td>
      <td>498</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST0000017325</td>
      <td>21.24</td>
      <td>10</td>
      <td>1</td>
      <td>2006-08-09</td>
      <td>2008-07-04</td>
      <td>2.124</td>
      <td>696</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST0000019732</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2008-03-12</td>
      <td>2008-03-12</td>
      <td>1.770</td>
      <td>1</td>
      <td>115</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>CUST0000994485</td>
      <td>1.77</td>
      <td>1</td>
      <td>1</td>
      <td>2006-04-14</td>
      <td>2006-04-14</td>
      <td>1.770</td>
      <td>1</td>
      <td>813</td>
      <td>2</td>
    </tr>
    <tr>
      <th>339</th>
      <td>CUST0000995305</td>
      <td>11.06</td>
      <td>4</td>
      <td>2</td>
      <td>2007-08-01</td>
      <td>2008-06-10</td>
      <td>2.765</td>
      <td>315</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>340</th>
      <td>CUST0000997122</td>
      <td>12.12</td>
      <td>6</td>
      <td>1</td>
      <td>2006-09-23</td>
      <td>2008-06-11</td>
      <td>2.020</td>
      <td>628</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341</th>
      <td>CUST0000999024</td>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>2006-12-17</td>
      <td>2006-12-17</td>
      <td>1.010</td>
      <td>1</td>
      <td>566</td>
      <td>2</td>
    </tr>
    <tr>
      <th>342</th>
      <td>CUST0000999935</td>
      <td>1.01</td>
      <td>1</td>
      <td>1</td>
      <td>2007-06-30</td>
      <td>2007-06-30</td>
      <td>1.010</td>
      <td>1</td>
      <td>371</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>343 rows × 10 columns</p>
</div>




```python
kmeans.labels_
```




    array([2, 2, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 2, 0, 1, 2, 2, 0, 0, 0, 2, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0,
           0, 2, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0,
           0, 1, 0, 1, 2, 1, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 0, 0, 2, 2, 2, 1,
           1, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 2, 2,
           1, 2, 2, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 2, 0, 0,
           2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 1, 2,
           1, 1, 2, 0, 0, 2, 1, 2, 2, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2,
           2, 0, 0, 2, 1, 2, 0, 2, 2, 2, 0, 0, 2, 1, 2, 0, 0, 2, 0, 0, 0, 2,
           1, 1, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 1, 0,
           0, 1, 2, 2, 1, 0, 0, 1, 0, 2, 2, 2, 0, 0, 2, 0, 2, 1, 0, 2, 0, 0,
           0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 2, 2, 1, 0, 2, 1, 2, 2, 2, 2, 2, 0,
           2, 1, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 1, 0, 2, 0, 2, 1, 2,
           2, 2, 0, 2, 2, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0, 1, 2, 1, 1, 1, 1, 0,
           1, 0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 1, 0, 1, 2, 1,
           1, 1, 0, 2, 0, 1, 0, 0, 2, 1, 1, 2, 0], dtype=int32)




```python
df_csv.value_counts("group")
```




    group
    0    150
    2    117
    1     76
    dtype: int64




```python
import seaborn as sns
```


```python
sns.pairplot(df_csv, hue="group")
```


```python
displayImg("assignment2-pairplot.jpg")
```




    <seaborn.axisgrid.PairGrid at 0x139e44850>




![png](assignment2-main_files/assignment2-main_24_1.png)


วิเคราะห์ข้อมูลจาก Pair Plot แล้วพบว่า เมื่อจัดกลุ่มแล้ว สามารถแบ่งกลุ่มลูกค้า ได้ทั้งหมด 3 กลุ่ม

### Segment 1 คนบ้านใกล้ - หิวเมื่อไหร่ก็แวะมา (สีเขียว)
คือ กลุ่มลูกค้าประจำ สังเกตจากตัวแปร TotalVisits จะพบว่ามีจำนวนการเข้าร้านที่บ่อย และมียอด Spend สูงสุด ซึ่งน่าจะมีโอกาสที่เป็นลูกค้าที่อยู่อาศัยใกล้กับร้าน เนื่องจากมียอดการจ่ายที่สูง และมีการกลับมาซื้อเป็นประจำ
### Segment 2 ชาวออฟฟิศ - มาแล้วยังดีกว่ามาช้า (สีแดง)
คือ กลุ่มพนักงานออฟฟิศ สังเกตจากตัวแปร TicketSize พบว่าลูกค้ากลุ่มนี้ มีการเข้าร้านบ่อยปานกลาง ซื้อสินค้าแต่ละรอบด้วยเงินที่ไม่สูงนัก และยังกลับมาซื้อสินค้าบ้าง จากขนาดของ Basket Size ที่ไม่เยอะมาก แสดงว่าเป็นของจุกจิกที่มียอดไม่มาก เช่น อาหารว่าง ของทานเล่น
### Segment 3 ลูกค้าเก่า - มาช้ายังดีกว่าไม่มา (สีฟ้า)
คือ กลุ่มลูกค้าสมาชิกเก่า สังเกตจากตัวแปร Recency หรือ ระยะเวลาที่ลูกค้าหายไป (นับจากวันล่าสุด) พบว่าลูกค้ากลุ่มนี้ส่วนใหญ่ปัจจุบันไม่กลับมาซื้อแล้ว และจากตัวแปร Total days จะพบว่า ลูกค้ากลุ่มนี้ในช่วงที่ซื้อ ก็มีอายุที่ไม่เยอะ แสดงว่า เป็นลูกค้ากลุ่มนี้ อาจจะเป็นลูกค้าที่ย้ายไปซื้อสินค้าที่ร้านอื่นแล้ว
