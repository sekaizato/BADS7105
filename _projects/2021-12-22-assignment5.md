---
title: Assignment 5 AB Testing
subtitle: randomized experimentation process 
date: 2021-12-07 00:00:00
description: method of comparing two versions of a webpage or app against each other
accent_color: '#4C60E6'
---
{% comment %}
    {% raw %}

```python
PROJECT_LINK = 'assignment5'
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
displayImg("abtesting1.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting1.png)




```python
displayImg("abtesting2.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting2.png)




```python
displayImg("abtesting3.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting3.png)




```python
displayImg("abtesting4.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting4.png)




```python
displayImg("abtesting5.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting5.png)




```python
displayImg("abtesting6.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting6.png)




```python
displayImg("abtesting7.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting7.png)




```python
displayImg("abtesting8.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting8.png)




```python
displayImg("abtesting9.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting9.png)




```python
displayImg("abtesting10.png")
```




![](/BADS7105/images/projects/2021-12-22-assignment5-abtesting10.png)


