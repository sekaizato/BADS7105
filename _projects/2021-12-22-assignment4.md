---
title: Assignment 4 Campaign Response Model
subtitle: Predicting incremental gains of promotional campaigns
date: 2021-12-08 00:00:00
description: To classify the customers who will respond to the next marketing campaign.
accent_color: '#4C60E6'
---
{% comment %}
    {% raw %}

```python
!pwd
```

    /Users/touchpadthamkul/zatoDev/project/bads_crm_final/master/BADS7105/writing



```python
PROJECT_LINK = 'assignment4'
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

saveExport()
# runBrowser(url)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-31-77b27c835cb2> in <module>
        100     get_ipython().system('mv $master_name $post_path')
        101 
    --> 102 saveExport()
        103 # runBrowser(url)


    <ipython-input-31-77b27c835cb2> in saveExport()
         87     md_name = ipynbname.name() +'.md'
         88     if PATH != '':
    ---> 89         selected = int(input('1 posts \n2 projects\n'))
         90         if selected != 1:
         91             folder = '/_projects'


    /opt/anaconda3/envs/sekai/lib/python3.8/site-packages/ipykernel/kernelbase.py in raw_input(self, prompt)
        858                 "raw_input was called, but this frontend does not support input requests."
        859             )
    --> 860         return self._input_request(str(prompt),
        861             self._parent_ident,
        862             self._parent_header,


    /opt/anaconda3/envs/sekai/lib/python3.8/site-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
        902             except KeyboardInterrupt:
        903                 # re-raise KeyboardInterrupt, to truncate traceback
    --> 904                 raise KeyboardInterrupt("Interrupted by user") from None
        905             except Exception as e:
        906                 self.log.warning("Invalid Message:", exc_info=True)


    KeyboardInterrupt: Interrupted by user

    {% endraw %}
{% endcomment %}
##  Objective


```python
import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
from xgboost import plot_importance

```


```python
df_response = pd.read_csv('Retail_Data_Response.csv')
df_transactions = pd.read_csv('Retail_Data_Transactions.csv', parse_dates=['trans_date'])
```


```python
df_response.head(5)
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
      <th>customer_id</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_transactions.head(5)
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
      <th>customer_id</th>
      <th>trans_date</th>
      <th>tran_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS5295</td>
      <td>2013-02-11</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS4768</td>
      <td>2015-03-15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS2122</td>
      <td>2013-02-26</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1217</td>
      <td>2011-11-16</td>
      <td>99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1850</td>
      <td>2013-11-20</td>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>




```python
campaign_date = dt.datetime(2015, 3, 17)

df_transactions['recent'] = campaign_date - df_transactions['trans_date']
df_transactions['recent'].astype('timedelta64[D]')
df_transactions['recent'] = df_transactions['recent'] / np.timedelta64(1, 'D')
df_transactions.head()
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
      <th>customer_id</th>
      <th>trans_date</th>
      <th>tran_amount</th>
      <th>recent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS5295</td>
      <td>2013-02-11</td>
      <td>35</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS4768</td>
      <td>2015-03-15</td>
      <td>39</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS2122</td>
      <td>2013-02-26</td>
      <td>52</td>
      <td>749.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1217</td>
      <td>2011-11-16</td>
      <td>99</td>
      <td>1217.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1850</td>
      <td>2013-11-20</td>
      <td>78</td>
      <td>482.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_transactions.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 125000 entries, 0 to 124999
    Data columns (total 4 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   customer_id  125000 non-null  object        
     1   trans_date   125000 non-null  datetime64[ns]
     2   tran_amount  125000 non-null  int64         
     3   recent       125000 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), int64(1), object(1)
    memory usage: 3.8+ MB



```python
df_transactions.groupby('customer_id').agg({'recent': lambda x: x.min(),
                                            'customer_id': lambda x: len(x),
                                            'tran_amount': lambda x: x.sum()
                                           })
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
      <th>recent</th>
      <th>customer_id</th>
      <th>tran_amount</th>
    </tr>
    <tr>
      <th>customer_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CS1112</th>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
    </tr>
    <tr>
      <th>CS1113</th>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
    </tr>
    <tr>
      <th>CS1114</th>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
    </tr>
    <tr>
      <th>CS1115</th>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
    </tr>
    <tr>
      <th>CS1116</th>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>CS8996</th>
      <td>98.0</td>
      <td>13</td>
      <td>582</td>
    </tr>
    <tr>
      <th>CS8997</th>
      <td>262.0</td>
      <td>14</td>
      <td>543</td>
    </tr>
    <tr>
      <th>CS8998</th>
      <td>85.0</td>
      <td>13</td>
      <td>624</td>
    </tr>
    <tr>
      <th>CS8999</th>
      <td>258.0</td>
      <td>12</td>
      <td>383</td>
    </tr>
    <tr>
      <th>CS9000</th>
      <td>17.0</td>
      <td>13</td>
      <td>533</td>
    </tr>
  </tbody>
</table>
<p>6889 rows × 3 columns</p>
</div>




```python
df_rfm = df_transactions.groupby('customer_id').agg({'recent': 'min',
                                            'customer_id': 'count',
                                            'tran_amount': 'sum'})

df_rfm.rename(columns={'recent': 'recency', 
                       'customer_id': 'frequency', 
                       'tran_amount': 'monetary_value'}, inplace=True)
```


```python
df_rfm = df_rfm.reset_index()
df_rfm.head()
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clv = df_transactions.groupby("customer_id").agg({'recent': 'min',
                                           'customer_id': 'count',
                                           'tran_amount': 'sum',
                                           'trans_date': lambda x: (x.max() - x.min()).days})

df_clv.rename(columns={'recent': 'recency', 
                       'customer_id': 'frequency', 
                       'tran_amount': 'monetary_value',
                       'trans_date' : 'AOU'}, inplace=True)

df_clv['ticket_size'] = df_clv['monetary_value'] / df_clv['frequency']
```


```python
df_clv = df_clv.reset_index()
df_clv.head()
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>AOU</th>
      <th>ticket_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
      <td>1309</td>
      <td>67.466667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
      <td>1354</td>
      <td>74.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
      <td>1309</td>
      <td>75.368421</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
      <td>1303</td>
      <td>75.409091</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
      <td>1155</td>
      <td>65.923077</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating addtional variables 
df_transactions = df_transactions.sort_values(by=['customer_id','trans_date'])
df_transactions['last_trans_date'] = df_transactions.groupby('customer_id')['trans_date'].shift(1)
df_transactions['day_from_last_purchase'] = (df_transactions['trans_date']-df_transactions['last_trans_date']).dt.days
df_transactions.sort_values(by=['customer_id','trans_date'])
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
      <th>customer_id</th>
      <th>trans_date</th>
      <th>tran_amount</th>
      <th>recent</th>
      <th>last_trans_date</th>
      <th>day_from_last_purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77247</th>
      <td>CS1112</td>
      <td>2011-06-15</td>
      <td>56</td>
      <td>1371.0</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89149</th>
      <td>CS1112</td>
      <td>2011-08-19</td>
      <td>96</td>
      <td>1306.0</td>
      <td>2011-06-15</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>68206</th>
      <td>CS1112</td>
      <td>2011-10-02</td>
      <td>60</td>
      <td>1262.0</td>
      <td>2011-08-19</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>36486</th>
      <td>CS1112</td>
      <td>2012-04-08</td>
      <td>56</td>
      <td>1073.0</td>
      <td>2011-10-02</td>
      <td>189.0</td>
    </tr>
    <tr>
      <th>93074</th>
      <td>CS1112</td>
      <td>2012-06-24</td>
      <td>52</td>
      <td>996.0</td>
      <td>2012-04-08</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102102</th>
      <td>CS9000</td>
      <td>2014-01-12</td>
      <td>16</td>
      <td>429.0</td>
      <td>2013-10-01</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>113120</th>
      <td>CS9000</td>
      <td>2014-05-08</td>
      <td>20</td>
      <td>313.0</td>
      <td>2014-01-12</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>103039</th>
      <td>CS9000</td>
      <td>2014-07-08</td>
      <td>26</td>
      <td>252.0</td>
      <td>2014-05-08</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>102384</th>
      <td>CS9000</td>
      <td>2014-08-24</td>
      <td>35</td>
      <td>205.0</td>
      <td>2014-07-08</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>113705</th>
      <td>CS9000</td>
      <td>2015-02-28</td>
      <td>34</td>
      <td>17.0</td>
      <td>2014-08-24</td>
      <td>188.0</td>
    </tr>
  </tbody>
</table>
<p>125000 rows × 6 columns</p>
</div>




```python
df_ticket_tbp = df_transactions.groupby('customer_id').agg(ticket_sd=('tran_amount','std'), ticket_mean=('tran_amount','mean'), MTBP=('day_from_last_purchase','mean'), SD_TBP=('day_from_last_purchase','std')).reset_index()
df_ticket_tbp['cv_ticket'] = df_ticket_tbp['ticket_sd']/df_ticket_tbp['ticket_mean']
df_ticket_tbp['cv_TBP'] = df_ticket_tbp['SD_TBP']/df_ticket_tbp['MTBP']
df_ticket_tbp
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
      <th>customer_id</th>
      <th>ticket_sd</th>
      <th>ticket_mean</th>
      <th>MTBP</th>
      <th>SD_TBP</th>
      <th>cv_ticket</th>
      <th>cv_TBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>19.766012</td>
      <td>67.466667</td>
      <td>93.500000</td>
      <td>50.873523</td>
      <td>0.292974</td>
      <td>0.544102</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>21.254102</td>
      <td>74.500000</td>
      <td>71.263158</td>
      <td>54.685812</td>
      <td>0.285290</td>
      <td>0.767378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>21.341692</td>
      <td>75.368421</td>
      <td>72.722222</td>
      <td>73.693168</td>
      <td>0.283165</td>
      <td>1.013351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>18.151896</td>
      <td>75.409091</td>
      <td>62.047619</td>
      <td>55.413425</td>
      <td>0.240712</td>
      <td>0.893079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>22.940000</td>
      <td>65.923077</td>
      <td>96.250000</td>
      <td>107.361010</td>
      <td>0.347981</td>
      <td>1.115439</td>
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
      <th>6884</th>
      <td>CS8996</td>
      <td>18.749017</td>
      <td>44.769231</td>
      <td>93.333333</td>
      <td>100.650912</td>
      <td>0.418792</td>
      <td>1.078403</td>
    </tr>
    <tr>
      <th>6885</th>
      <td>CS8997</td>
      <td>14.000981</td>
      <td>38.785714</td>
      <td>85.846154</td>
      <td>77.791223</td>
      <td>0.360983</td>
      <td>0.906170</td>
    </tr>
    <tr>
      <th>6886</th>
      <td>CS8998</td>
      <td>22.319648</td>
      <td>48.000000</td>
      <td>107.750000</td>
      <td>109.644985</td>
      <td>0.464993</td>
      <td>1.017587</td>
    </tr>
    <tr>
      <th>6887</th>
      <td>CS8999</td>
      <td>15.453498</td>
      <td>31.916667</td>
      <td>97.545455</td>
      <td>55.000661</td>
      <td>0.484183</td>
      <td>0.563846</td>
    </tr>
    <tr>
      <th>6888</th>
      <td>CS9000</td>
      <td>22.304708</td>
      <td>41.000000</td>
      <td>106.083333</td>
      <td>76.092597</td>
      <td>0.544017</td>
      <td>0.717291</td>
    </tr>
  </tbody>
</table>
<p>6889 rows × 7 columns</p>
</div>




```python
df_clv = df_clv.merge(df_ticket_tbp, how='left',on='customer_id')
df_clv
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>AOU</th>
      <th>ticket_size</th>
      <th>ticket_sd</th>
      <th>ticket_mean</th>
      <th>MTBP</th>
      <th>SD_TBP</th>
      <th>cv_ticket</th>
      <th>cv_TBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
      <td>1309</td>
      <td>67.466667</td>
      <td>19.766012</td>
      <td>67.466667</td>
      <td>93.500000</td>
      <td>50.873523</td>
      <td>0.292974</td>
      <td>0.544102</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
      <td>1354</td>
      <td>74.500000</td>
      <td>21.254102</td>
      <td>74.500000</td>
      <td>71.263158</td>
      <td>54.685812</td>
      <td>0.285290</td>
      <td>0.767378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
      <td>1309</td>
      <td>75.368421</td>
      <td>21.341692</td>
      <td>75.368421</td>
      <td>72.722222</td>
      <td>73.693168</td>
      <td>0.283165</td>
      <td>1.013351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
      <td>1303</td>
      <td>75.409091</td>
      <td>18.151896</td>
      <td>75.409091</td>
      <td>62.047619</td>
      <td>55.413425</td>
      <td>0.240712</td>
      <td>0.893079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
      <td>1155</td>
      <td>65.923077</td>
      <td>22.940000</td>
      <td>65.923077</td>
      <td>96.250000</td>
      <td>107.361010</td>
      <td>0.347981</td>
      <td>1.115439</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6884</th>
      <td>CS8996</td>
      <td>98.0</td>
      <td>13</td>
      <td>582</td>
      <td>1120</td>
      <td>44.769231</td>
      <td>18.749017</td>
      <td>44.769231</td>
      <td>93.333333</td>
      <td>100.650912</td>
      <td>0.418792</td>
      <td>1.078403</td>
    </tr>
    <tr>
      <th>6885</th>
      <td>CS8997</td>
      <td>262.0</td>
      <td>14</td>
      <td>543</td>
      <td>1116</td>
      <td>38.785714</td>
      <td>14.000981</td>
      <td>38.785714</td>
      <td>85.846154</td>
      <td>77.791223</td>
      <td>0.360983</td>
      <td>0.906170</td>
    </tr>
    <tr>
      <th>6886</th>
      <td>CS8998</td>
      <td>85.0</td>
      <td>13</td>
      <td>624</td>
      <td>1293</td>
      <td>48.000000</td>
      <td>22.319648</td>
      <td>48.000000</td>
      <td>107.750000</td>
      <td>109.644985</td>
      <td>0.464993</td>
      <td>1.017587</td>
    </tr>
    <tr>
      <th>6887</th>
      <td>CS8999</td>
      <td>258.0</td>
      <td>12</td>
      <td>383</td>
      <td>1073</td>
      <td>31.916667</td>
      <td>15.453498</td>
      <td>31.916667</td>
      <td>97.545455</td>
      <td>55.000661</td>
      <td>0.484183</td>
      <td>0.563846</td>
    </tr>
    <tr>
      <th>6888</th>
      <td>CS9000</td>
      <td>17.0</td>
      <td>13</td>
      <td>533</td>
      <td>1273</td>
      <td>41.000000</td>
      <td>22.304708</td>
      <td>41.000000</td>
      <td>106.083333</td>
      <td>76.092597</td>
      <td>0.544017</td>
      <td>0.717291</td>
    </tr>
  </tbody>
</table>
<p>6889 rows × 12 columns</p>
</div>




```python
combined_df = df_clv.merge(df_response, how='left', on='customer_id').dropna().reset_index(drop=True)
combined_df
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>AOU</th>
      <th>ticket_size</th>
      <th>ticket_sd</th>
      <th>ticket_mean</th>
      <th>MTBP</th>
      <th>SD_TBP</th>
      <th>cv_ticket</th>
      <th>cv_TBP</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
      <td>1309</td>
      <td>67.466667</td>
      <td>19.766012</td>
      <td>67.466667</td>
      <td>93.500000</td>
      <td>50.873523</td>
      <td>0.292974</td>
      <td>0.544102</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
      <td>1354</td>
      <td>74.500000</td>
      <td>21.254102</td>
      <td>74.500000</td>
      <td>71.263158</td>
      <td>54.685812</td>
      <td>0.285290</td>
      <td>0.767378</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
      <td>1309</td>
      <td>75.368421</td>
      <td>21.341692</td>
      <td>75.368421</td>
      <td>72.722222</td>
      <td>73.693168</td>
      <td>0.283165</td>
      <td>1.013351</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
      <td>1303</td>
      <td>75.409091</td>
      <td>18.151896</td>
      <td>75.409091</td>
      <td>62.047619</td>
      <td>55.413425</td>
      <td>0.240712</td>
      <td>0.893079</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
      <td>1155</td>
      <td>65.923077</td>
      <td>22.940000</td>
      <td>65.923077</td>
      <td>96.250000</td>
      <td>107.361010</td>
      <td>0.347981</td>
      <td>1.115439</td>
      <td>1.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6879</th>
      <td>CS8996</td>
      <td>98.0</td>
      <td>13</td>
      <td>582</td>
      <td>1120</td>
      <td>44.769231</td>
      <td>18.749017</td>
      <td>44.769231</td>
      <td>93.333333</td>
      <td>100.650912</td>
      <td>0.418792</td>
      <td>1.078403</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6880</th>
      <td>CS8997</td>
      <td>262.0</td>
      <td>14</td>
      <td>543</td>
      <td>1116</td>
      <td>38.785714</td>
      <td>14.000981</td>
      <td>38.785714</td>
      <td>85.846154</td>
      <td>77.791223</td>
      <td>0.360983</td>
      <td>0.906170</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6881</th>
      <td>CS8998</td>
      <td>85.0</td>
      <td>13</td>
      <td>624</td>
      <td>1293</td>
      <td>48.000000</td>
      <td>22.319648</td>
      <td>48.000000</td>
      <td>107.750000</td>
      <td>109.644985</td>
      <td>0.464993</td>
      <td>1.017587</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6882</th>
      <td>CS8999</td>
      <td>258.0</td>
      <td>12</td>
      <td>383</td>
      <td>1073</td>
      <td>31.916667</td>
      <td>15.453498</td>
      <td>31.916667</td>
      <td>97.545455</td>
      <td>55.000661</td>
      <td>0.484183</td>
      <td>0.563846</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6883</th>
      <td>CS9000</td>
      <td>17.0</td>
      <td>13</td>
      <td>533</td>
      <td>1273</td>
      <td>41.000000</td>
      <td>22.304708</td>
      <td>41.000000</td>
      <td>106.083333</td>
      <td>76.092597</td>
      <td>0.544017</td>
      <td>0.717291</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6884 rows × 13 columns</p>
</div>




```python
combined_df.head()
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>AOU</th>
      <th>ticket_size</th>
      <th>ticket_sd</th>
      <th>ticket_mean</th>
      <th>MTBP</th>
      <th>SD_TBP</th>
      <th>cv_ticket</th>
      <th>cv_TBP</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
      <td>1309</td>
      <td>67.466667</td>
      <td>19.766012</td>
      <td>67.466667</td>
      <td>93.500000</td>
      <td>50.873523</td>
      <td>0.292974</td>
      <td>0.544102</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
      <td>1354</td>
      <td>74.500000</td>
      <td>21.254102</td>
      <td>74.500000</td>
      <td>71.263158</td>
      <td>54.685812</td>
      <td>0.285290</td>
      <td>0.767378</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
      <td>1309</td>
      <td>75.368421</td>
      <td>21.341692</td>
      <td>75.368421</td>
      <td>72.722222</td>
      <td>73.693168</td>
      <td>0.283165</td>
      <td>1.013351</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
      <td>1303</td>
      <td>75.409091</td>
      <td>18.151896</td>
      <td>75.409091</td>
      <td>62.047619</td>
      <td>55.413425</td>
      <td>0.240712</td>
      <td>0.893079</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
      <td>1155</td>
      <td>65.923077</td>
      <td>22.940000</td>
      <td>65.923077</td>
      <td>96.250000</td>
      <td>107.361010</td>
      <td>0.347981</td>
      <td>1.115439</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "colab"
pio.templates.default = "plotly_dark"

# combined_df["response"] = combined_df["response"].astype(str)
```


```python
feature_list = combined_df.columns.tolist()[1:]
fig = px.scatter_matrix(combined_df, dimensions=feature_list, color="response")
fig.update_layout(height=1200)
displayPlot(fig)
```




{% include post-figures/2021-12-22-assignment4_fig.html full_width=true %}




```python
#selecting columns to be trained
df_modeling_clv = combined_df[['customer_id', 
                        'recency', 
                        'frequency', 
                        'monetary_value', 
                        # 'AOU',
                        'ticket_size', 
                        # 'ticket_sd', 
                        'ticket_mean', 
                        'MTBP',
                        'SD_TBP',
                        'cv_ticket', 
                        # 'cv_TBP', 
                        'response']]
df_modeling_clv = df_modeling_clv[~(df_modeling_clv['response'].isnull())]
df_modeling_clv
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
      <th>customer_id</th>
      <th>recency</th>
      <th>frequency</th>
      <th>monetary_value</th>
      <th>ticket_size</th>
      <th>ticket_mean</th>
      <th>MTBP</th>
      <th>SD_TBP</th>
      <th>cv_ticket</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CS1112</td>
      <td>62.0</td>
      <td>15</td>
      <td>1012</td>
      <td>67.466667</td>
      <td>67.466667</td>
      <td>93.500000</td>
      <td>50.873523</td>
      <td>0.292974</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CS1113</td>
      <td>36.0</td>
      <td>20</td>
      <td>1490</td>
      <td>74.500000</td>
      <td>74.500000</td>
      <td>71.263158</td>
      <td>54.685812</td>
      <td>0.285290</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CS1114</td>
      <td>33.0</td>
      <td>19</td>
      <td>1432</td>
      <td>75.368421</td>
      <td>75.368421</td>
      <td>72.722222</td>
      <td>73.693168</td>
      <td>0.283165</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CS1115</td>
      <td>12.0</td>
      <td>22</td>
      <td>1659</td>
      <td>75.409091</td>
      <td>75.409091</td>
      <td>62.047619</td>
      <td>55.413425</td>
      <td>0.240712</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CS1116</td>
      <td>204.0</td>
      <td>13</td>
      <td>857</td>
      <td>65.923077</td>
      <td>65.923077</td>
      <td>96.250000</td>
      <td>107.361010</td>
      <td>0.347981</td>
      <td>1.0</td>
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
      <th>6879</th>
      <td>CS8996</td>
      <td>98.0</td>
      <td>13</td>
      <td>582</td>
      <td>44.769231</td>
      <td>44.769231</td>
      <td>93.333333</td>
      <td>100.650912</td>
      <td>0.418792</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6880</th>
      <td>CS8997</td>
      <td>262.0</td>
      <td>14</td>
      <td>543</td>
      <td>38.785714</td>
      <td>38.785714</td>
      <td>85.846154</td>
      <td>77.791223</td>
      <td>0.360983</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6881</th>
      <td>CS8998</td>
      <td>85.0</td>
      <td>13</td>
      <td>624</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>107.750000</td>
      <td>109.644985</td>
      <td>0.464993</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6882</th>
      <td>CS8999</td>
      <td>258.0</td>
      <td>12</td>
      <td>383</td>
      <td>31.916667</td>
      <td>31.916667</td>
      <td>97.545455</td>
      <td>55.000661</td>
      <td>0.484183</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6883</th>
      <td>CS9000</td>
      <td>17.0</td>
      <td>13</td>
      <td>533</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>106.083333</td>
      <td>76.092597</td>
      <td>0.544017</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6884 rows × 10 columns</p>
</div>




```python
response_rate = df_response.groupby('response').agg({'customer_id': lambda x: len(x)}).reset_index()
response_rate.head()
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
      <th>response</th>
      <th>customer_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6237</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>647</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_data = df_modeling_clv.drop(columns=['response','customer_id']).values
y_data = df_modeling_clv['response'].tolist()
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.15)
```


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42)

clf.fit(X_train, y_train)
```




    LogisticRegression(random_state=42)




```python
predicted = clf.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted)
```




    0.9090029041626331




```python
plot_roc_curve(clf, X_test, y_test)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x14059c100>




![png](assignment4-main_files/assignment4-main_30_1.png)

