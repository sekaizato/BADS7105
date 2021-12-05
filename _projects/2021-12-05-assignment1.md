---
title: Assignment 1 Customer Insights with Multidimensional Analysis
subtitle: Multi Dimensional Dashboard for Supermarket
date: 2021-12-04 00:00:00
description: Project for analyze customer relationship management on supermarket by User Empathy Map.
featured_image: 2021-12-05-assignment1-assignment1-dashboard.jpg
accent_color: '#4C60E6'
gallery_images:
  - demo.jpg
---
{% comment %}
    {% raw %}

```python
!pwd
```

    /Users/touchpadthamkul/zatoDev/project/bads_crm_final/master/BADS7105/writing



```python
PROJECT_LINK = 'assignment1'
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

saveExport()
# runBrowser(url)
```

    1 posts 
    2 projects
     2


    [NbConvertApp] Converting notebook assignment1.ipynb to markdown
    [NbConvertApp] Writing 6497 bytes to assignment1.md

    {% endraw %}
{% endcomment %}
##  Objective

โปรเจคนี้จะทำการวิเคราะห์ข้อมูลของ Supermarket จากบริษัท Dunnhumby โดยผลลัพธ์สุดท้ายจะเป็นงาน Dashboard เพื่อตอบโจทย์การทำงานของ User ด้วย Framework User Empathy Map ( Problem-Reason-Action) <br>
โจทย์นี้จะทำการออกแบบ Dashboard ให้กับพนักงานฝ่ายดูแลลูกค้า CRM โดยมีหน้าที่ได้รับมอบหมายคือการ ติดตามลูกค้าเก่าที่หายไป ดูแลจนกว่าจะกลับมาใช้ โดยมีเป้าที่ต้องทำให้ได้คือ Retention Rate จะต้องมากกว่า 30% <br>
โดยเราจะใช้ข้อมูลจาก บริษัท Dunnhumby ซึ่งเป็น บริษัทที่ทำ analytics ให้กับ Tesco ทำด้านเกี่ยวกับ Customer Data Science (Dr.Thanachart Ritbumroong)

## Problem-Reason-Action

### Problem

- ลูกค้าเริ่มไม่มาซื้อสินค้า

### Reason

- ความถี่การใช้งานเริ่มน้อยลง
- ช่วงเวลาระยะห่างการมาซื้อสูงขึ้น
- ลูกค้าสมาชิกมาซื้อห่างขึ้น

### Action

- ส่งโปรโมชั่นหาลูกค้าที่เป็นสมาชิกเริ่มไม่มา (นำสินค้าที่ขายดีมาแนะนำลูกค้าแบบ Personalized )

## Data Understanding

### Data Set Details

- shop_week : Identifies the week of the basket
- shop_date : Date when shopping has been made
- shop_weekday : Identifies the day of the week
- shop_hour : Hour slot of the shopping
- Quantity : Number of items of the same product bought in this basket
- spend : Spend associated to the items bought
- prod_code : Product Code
- prod_code_10 : Product Hierarchy Level 10 Code
- prod_code_20 : Product Hierarchy Level 20 Code
- prod_code_30 : Product Hierarchy Level 30 Code
- prod_code_40 : Product Hierarchy Level 40 Code
- cust_code : Customer Code
- cust_price_sensitivity : Customer's Price Sensitivity
- cust_lifestage : Customer's Lifestage
- basket_id : Basket ID. All items in a basket share the same basket_id value.
- basket_size : Basket size
- basket_price_sensitivity : Basket price sensitivity
- basket_type : Basket type
- basket_dominant_mission : Shopping dominant mission
- store_code : Store Code
- store_format : Format of the Store
- store_region : Region the store belongs to

### Key Behavior

- Visit Frequency
- Spending Per Customer (CLV)
- Meantime Between Purchases
- Price sensitivity
    - Basket Size
- Propensity to Churn
- Number of repeat vs New Customers

### Dimensions

- Store
- Region
- Time
- Major Product
- Member vs Non-member
- Customer Life Stage
- Product Category (Different Product)

## Dashboard

สร้าง Dashboard เพื่อตอบโจทย์ User (Update4-multiple)


```python
displayImg("assignment1-dashboard.jpg")
```

![](BADS7105/images/projects/2021-12-05-assignment1-assignment1-dashboard.jpg)
![](images/projects/2021-12-05-assignment1-assignment1-dashboard.jpg)
![](/2021-12-05-assignment1-assignment1-dashboard.jpg)
![](2021-12-05-assignment1-assignment1-dashboard.jpg)


![](/images/demo.jpg)

