---
title: Assignment 3 Product Recommendation Engine
subtitle: Cross Selling Models by Collaborative Filtering Algorithm
date: 2021-12-04 00:00:00
description: Project for study relationship between transaction of supermarket data.
featured_image: 2021-12-05-assignment1-assignment1-dashboard.jpg
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
PROJECT_LINK = 'assignment3'
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

จากโจทย์วันนี้ อาจารย์ให้เพื่อนทุกคนส่งแบบสอบถามว่าเคย/ไ่ม่เคยซื้อสินค้า เพื่อนำมาศึกษาด้วย Market Basket Analysis หรือ Collaborative Filtering และทดลองแนะนำสินค้า ด้วยสินค้าที่เพื่อนนิยมซื้อไปด้วยกันมากที่สุด

เริ่มจากดึงข้อมูลแบบสอบถามจากเพื่อนในห้อง ข้อมูลทั้งหมด 47 แถว และมีสินค้าทั้งหมด 41 ชิ้น


```python
import pandas as pd
raw_df = pd.read_csv("cross_selling.csv")
raw_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 47 entries, 0 to 46
    Data columns (total 42 columns):
     #   Column                         Non-Null Count  Dtype 
    ---  ------                         --------------  ----- 
     0   Timestamp                      47 non-null     object
     1   playstation5                   46 non-null     object
     2   เครื่องทำขนมปัง                46 non-null     object
     3   Ergonomic Wrist Rest           46 non-null     object
     4   เครื่องอบผ้า                   46 non-null     object
     5   เครื่องชงกาแฟแคปซูล            46 non-null     object
     6   เก้าอี้ LA-Z-Boy               46 non-null     object
     7   เครื่องให้อาหารสัตว์อัตโนมัติ  46 non-null     object
     8   บัตตาเลี่ยน                    45 non-null     object
     9   แก้วเก็บความเย็น               46 non-null     object
     10  ลู่วิ่งออกกำลังกาย             46 non-null     object
     11  Kindle                         46 non-null     object
     12  เครื่องซักผ้า                  46 non-null     object
     13  Bluetooth Speaker              46 non-null     object
     14  ห้องน้ำแมวอัตโนมัติ            46 non-null     object
     15  PS5                            46 non-null     object
     16  ทรายแมว                        46 non-null     object
     17  ลำโพง pixel                    46 non-null     object
     18  Logitech Mx Master 3 Mouse     46 non-null     object
     19  ตุ๊กตา ty                      46 non-null     object
     20  น้ำพุแมว                       46 non-null     object
     21  Robot ดูดฝุ่น                  46 non-null     object
     22  Mechanical keyboard            45 non-null     object
     23  Nintendo switch                45 non-null     object
     24  หนังสือ python                 46 non-null     object
     25  gaming chair                   45 non-null     object
     26  Deskmat                        46 non-null     object
     27  Dew - ไฟโรเซ่                  46 non-null     object
     28  เทียนหอม jo malone             46 non-null     object
     29  กระติกน้ำ 2 ลิตร               45 non-null     object
     30  ที่นอน memory form             46 non-null     object
     31  พลาสเตอร์บรรเทาปวด ตราเสือ     46 non-null     object
     32  การ์ดจอ RTX 3080               46 non-null     object
     33  ขนมจีนน้ำยาปู                  46 non-null     object
     34  Salmon Sashimi                 46 non-null     object
     35  จักรยานเสือหมอบ                46 non-null     object
     36  ไฟแต่งห้องมินิมอล              45 non-null     object
     37  External Harddisk              46 non-null     object
     38  หม้อทอดไร้น้ํามัน              45 non-null     object
     39  airpods                        46 non-null     object
     40  ยาดม                           46 non-null     object
     41  ไฟส่องหน้าไลฟ์สด               46 non-null     object
    dtypes: object(42)
    memory usage: 15.5+ KB


เตรียมข้อมูลสำหรับ Collaborative Filtering ทำการ Clean ข้อมูล เคยซื้อ-ไม่เคยซื้อ และรวมไว้ในคอลัมน์ Transaction และ Item


```python
pre_df = raw_df.reset_index().rename(columns={"index": "Transaction"}).drop(columns="Timestamp").dropna()
df1 = pd.melt(pre_df, id_vars=['Transaction'], value_vars=pre_df.columns.tolist()[1:]).rename(columns={'variable': 'Item'})
df = df1[~df1["value"].str.contains("ไม่")].sort_values("Transaction").drop(columns="value").reset_index(drop=True)
df.loc[df["Item"]=='NONE'].shape[0]
df = df[df["Item"] != 'NONE']
df
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
      <th>Transaction</th>
      <th>Item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>การ์ดจอ RTX 3080</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>บัตตาเลี่ยน</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>หม้อทอดไร้น้ํามัน</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Deskmat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Mechanical keyboard</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>457</th>
      <td>46</td>
      <td>ยาดม</td>
    </tr>
    <tr>
      <th>458</th>
      <td>46</td>
      <td>Bluetooth Speaker</td>
    </tr>
    <tr>
      <th>459</th>
      <td>46</td>
      <td>airpods</td>
    </tr>
    <tr>
      <th>460</th>
      <td>46</td>
      <td>Mechanical keyboard</td>
    </tr>
    <tr>
      <th>461</th>
      <td>46</td>
      <td>แก้วเก็บความเย็น</td>
    </tr>
  </tbody>
</table>
<p>462 rows × 2 columns</p>
</div>



ทำ One Hot Encoding เพื่อรวม Transaction และทำให้ข้อมูลอยู่ในรูปแบบ Binary 


```python
hot_encoded_df=df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
hot_encoded_df = hot_encoded_df.applymap(encode_units)
hot_encoded_df.head(5)
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
      <th>Item</th>
      <th>Bluetooth Speaker</th>
      <th>Deskmat</th>
      <th>Dew - ไฟโรเซ่</th>
      <th>Ergonomic Wrist Rest</th>
      <th>External Harddisk</th>
      <th>Kindle</th>
      <th>Logitech Mx Master 3 Mouse</th>
      <th>Mechanical keyboard</th>
      <th>Nintendo switch</th>
      <th>PS5</th>
      <th>...</th>
      <th>เก้าอี้ LA-Z-Boy</th>
      <th>เครื่องชงกาแฟแคปซูล</th>
      <th>เครื่องซักผ้า</th>
      <th>เครื่องทำขนมปัง</th>
      <th>เครื่องอบผ้า</th>
      <th>เครื่องให้อาหารสัตว์อัตโนมัติ</th>
      <th>เทียนหอม jo malone</th>
      <th>แก้วเก็บความเย็น</th>
      <th>ไฟส่องหน้าไลฟ์สด</th>
      <th>ไฟแต่งห้องมินิมอล</th>
    </tr>
    <tr>
      <th>Transaction</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



ใช้ฟังก์ชัน association_rules จาก Library mlxtend


```python
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(hot_encoded_df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values("support", ascending=False).reset_index(drop=True).head(5)
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Salmon Sashimi)</td>
      <td>(ยาดม)</td>
      <td>0.825</td>
      <td>0.825</td>
      <td>0.725</td>
      <td>0.878788</td>
      <td>1.065197</td>
      <td>0.044375</td>
      <td>1.443750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(ยาดม)</td>
      <td>(Salmon Sashimi)</td>
      <td>0.825</td>
      <td>0.825</td>
      <td>0.725</td>
      <td>0.878788</td>
      <td>1.065197</td>
      <td>0.044375</td>
      <td>1.443750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(ยาดม)</td>
      <td>(Bluetooth Speaker)</td>
      <td>0.825</td>
      <td>0.700</td>
      <td>0.625</td>
      <td>0.757576</td>
      <td>1.082251</td>
      <td>0.047500</td>
      <td>1.237500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Bluetooth Speaker)</td>
      <td>(ยาดม)</td>
      <td>0.700</td>
      <td>0.825</td>
      <td>0.625</td>
      <td>0.892857</td>
      <td>1.082251</td>
      <td>0.047500</td>
      <td>1.633333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(แก้วเก็บความเย็น)</td>
      <td>(ยาดม)</td>
      <td>0.750</td>
      <td>0.825</td>
      <td>0.625</td>
      <td>0.833333</td>
      <td>1.010101</td>
      <td>0.006250</td>
      <td>1.050000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## สรุป

สินค้าที่ควรแนะนำให้ลูกค้าคือ
1. ยาดม และ Bluetooth Speakere
2. ยาดม และ Salmon Sashimi
3. ยาแด
```


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

mpl.font_manager.fontManager.addfont('Sarabun-Regular.ttf')
mpl.rc('font', family='Sarabun')
fig, ax=plt.subplots(figsize=(10,4))
GA=nx.from_pandas_edgelist(rules,source='antecedents',target='consequents')
nx.draw(GA,with_labels=True, font_family="Sarabun")
plt.show()
```


![png](assignment1_files/assignment1_15_0.png)

