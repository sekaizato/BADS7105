---
title: Assignment 6 Customer Movement Analysis
subtitle: Visualization of Churn Customer
date: 2021-12-06 00:00:00
description: Project for study relationship between transaction of supermarket data.
accent_color: '#4C60E6'

---
{% comment %}
    {% raw %}

```python
PROJECT_LINK = 'assignment6'
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
## Objective
Visualization for Churn Customer


```python
WITH
    yearmonthTable AS (
        SELECT DISTINCT FORMAT_DATETIME("%Y-%m", SHOP_DATE) AS year_month
        FROM (SELECT PARSE_DATE('%Y%m%d', CAST(SHOP_DATE AS STRING)) AS SHOP_DATE
        FROM `sekai420.bads.crm_supermarket_data`
        WHERE CUST_CODE IS NOT NULL)
        ),
    customerIDTable AS (
        SELECT DISTINCT CUST_CODE as customerID
        FROM `sekai420.bads.crm_supermarket_data`
        WHERE CUST_CODE IS NOT NULL
        ),
    mindateTable AS (
        SELECT CUST_CODE, FORMAT_DATETIME("%Y-%m", MIN(PARSE_DATE('%Y%m%d', CAST(SHOP_DATE AS STRING)))) AS startDate
        FROM `sekai420.bads.crm_supermarket_data`
        WHERE CUST_CODE IS NOT NULL
        GROUP BY CUST_CODE
    ),
    crossTable AS (
        SELECT * FROM yearmonthTable
        CROSS JOIN customerIDTable
    ),
    customerTable AS (
        SELECT crossTable.customerID as customerID, year_month, startDate FROM crossTable
        LEFT JOIN mindateTable
        ON crossTable.customerID = mindateTable.CUST_CODE
    ),
    checkTable AS (
        SELECT DISTINCT FORMAT_DATETIME("%Y-%m", PARSE_DATE('%Y%m%d', CAST(SHOP_DATE AS STRING))) AS year_month, CUST_CODE, 1 AS check
        FROM `sekai420.bads.crm_supermarket_data`
        WHERE CUST_CODE IS NOT NULL
    ),
    preTable AS (
        SELECT customerTable.customerID, customerTable.year_month, customerTable.startDate, IFNULL(check, 0) as check
        FROM customerTable
        LEFT JOIN checkTable
        ON (customerTable.customerID = checkTable.CUST_CODE) AND (customerTable.year_month = checkTable.year_month)
    ),
    cleanTable AS (
        SELECT *, IFNULL(LAG(check) OVER (PARTITION BY customerID ORDER BY year_month), 0) AS previous
        FROM preTable
    ),
    masterTable AS (
        SELECT *, (CASE
        WHEN startDate = year_month THEN "New Customer"
        WHEN previous = 1 AND check = 1 THEN "Repeat"
        WHEN previous = 0 AND check = 1 THEN "Reactivated"
        WHEN previous = 1 AND check = 0 THEN "Churn"
        -- WHEN previous = 0 AND check = 0 AND PARSE_DATE('%Y-%m', startDate) < PARSE_DATE('%Y-%m', year_month) THEN "Churn"
        END) AS status,
        PARSE_DATE('%Y-%m', year_month) AS ym,
        PARSE_DATE('%Y-%m', startDate) AS sd
        FROM cleanTable
    )

SELECT *, CASE WHEN status != "Churn" THEN 1 ELSE -1 END as val
FROM masterTable
WHERE status IS NOT NULL
ORDER BY customerID
```


```python
displayImg("churnchart.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment6-churnchart.png)


