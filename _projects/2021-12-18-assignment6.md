```python
print('hello')
```

    hello



```python
import numpy as np
import pandas as pd
import re

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import umap

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

import pythainlp
from pythainlp.corpus.common import thai_words
from pythainlp.util import Trie
import collections

```

ก่อนจะสร้างโมเดล Natural Language Processing (NLP) ต้องทำ Text Embedding เพื่อแปลงจากตัวอักษร เป็นตัวเลข เพื่อให้คอมพิวเตอร์เข้าใจภาษามนุย์ โดยเราจะทำการใช้ Google's Universal Sentence Encoder เพื่อใช้ฟังก์ชันเปลี่ยนตัวอักษรเป็นตัวเลข


```python
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3' #'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3' for a large model

model = hub.load(module_url)
```

ดึงข้อมูลรีวิวจาก Wongnai ทั้งหมดทั้งหมด 300 รีวิว


```python
df = pd.read_csv('Wongnai Reviews - Small.csv')
df.head()
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
      <th>Review ID</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>เป็นคนที่ชอบทาน Macchiato เป็นประจำ มีวันนึงเด...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Art of Coffee Kasetsart เป็นร้านกาแฟรสชาติเยี่...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>กวงทะเลเผา อาหารทะเลเค้าสดจริงๆเนื้อปูหวานไม่ค...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>วันนี้มีโอกาสตื่นเช้าครับเลยถึงโอกาสออกมาหาอะไ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>ชอบมาทานร้านนี้ถ้าอยากกินอาหารเวียดนามใกล้บ้าน...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 300 entries, 0 to 299
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   Review ID  300 non-null    int64 
     1   Review     300 non-null    object
     2   KMeans ID  300 non-null    int32 
    dtypes: int32(1), int64(1), object(1)
    memory usage: 6.0+ KB


### Step1 - document embedding and dimension reduction


```python
#embed sentences using Universal Sentence Encoder (USE)
#reduce array dimensions using umap (you can chagne n_components)

embed_comments_array = model(df['Review'].values).numpy()
reducer = umap.UMAP(random_state=42,n_components=50)
umap_embed_comments_array = reducer.fit_transform(embed_comments_array)

umap_embed_comments_array
```




    array([[4.38797  , 7.3366637, 4.3473525, ..., 5.737648 , 4.4198318,
            5.942006 ],
           [4.5256295, 7.3365474, 4.4462757, ..., 5.623406 , 4.375579 ,
            5.9466367],
           [5.2760754, 6.866459 , 3.9091702, ..., 5.8130536, 4.945471 ,
            5.4968505],
           ...,
           [4.4710193, 6.991462 , 4.4716787, ..., 5.644046 , 4.444365 ,
            5.9356337],
           [4.4344482, 7.070292 , 4.534156 , ..., 5.6133623, 4.41475  ,
            5.9124784],
           [4.376653 , 7.1932964, 4.4902244, ..., 5.676849 , 4.370813 ,
            5.965906 ]], dtype=float32)



### Step 2 - document clustering using KMeans


```python
#run kmeans with various number of k. evaluate no. of k based on the elbow plot

wcss=[]
max_k = 10
for i in range(1, max_k):
  kmeans = KMeans(i)
  kmeans.fit(umap_embed_comments_array)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)

number_clusters = range(1, max_k)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
```




    Text(0, 0.5, 'WCSS')




![png](assignment6-main_files/assignment6-main_10_1.png)



```python
#run kmeans with no. of clusters you see fit the most

k = 3

kmeans = KMeans(n_clusters = k)
kmeans.fit(umap_embed_comments_array)

df['KMeans ID'] = kmeans.labels_

#merge all reviews of each cluster into one big sentence 

df_kmeans = pd.DataFrame(columns=["KMeans ID", "texts"])


for i in range(0, k):
  row = []
  row.append(i)
  row.append(df['Review'][df['KMeans ID'] == i].to_string())
  df_kmeans.loc[len(df_kmeans)] = row
    
df_kmeans
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
      <th>KMeans ID</th>
      <th>texts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0      เป็นคนที่ชอบทาน Macchiato เป็นประจำ มีว...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2      กวงทะเลเผา อาหารทะเลเค้าสดจริงๆเนื้อปูห...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>13     เคยเป็นไหมกันไหมคะ หลังอาหารมื้อใหญ่ ต่...</td>
    </tr>
  </tbody>
</table>
</div>



สร้างคำ และแพทเทิร์นที่ต้องการยกเว้น


```python
#create regex compiler for removal of a character you don't want

special_characters = "/[!@#$%^&*']/g"
specialchar_pattern = re.compile(special_characters)

#create regex compiler for removal of any emoji

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

#create regex compiler for removal of digit

number_pattern = re.compile("[0-9]")

#create regex compiler for removal of white space

space_pattern = re.compile("\s+")

#create regex compiler for removal of .

dot_pattern = re.compile(r"\.+")

#create regex compiler for removal of \

backslash_pattern = re.compile(r"\\+")
```


```python
#define a function to tokenize a sentence into words - you can define words you want to remove as well as new words for tokenization

stopwords = list(pythainlp.corpus.thai_stopwords())
removed_words = ['u', 'b', 'n', 'nn', 'nn-', '\n', 'ร้าน', 'ทาน', 'กิน']
screening_words = stopwords + removed_words

new_words = {"สตารบัก"}

words = new_words.union(thai_words())

custom_dictionary_trie = Trie(words)

def tokenize_to_list(sentence):
  merged = []
  words = pythainlp.word_tokenize(str(sentence), engine='newmm', custom_dict=custom_dictionary_trie)
  for word in words:
    if word not in screening_words:
      merged.append(word)
  return merged
```


```python
#clean and tokenize sentences. count the occurences of each word

df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: emoji_pattern.sub(r'', x))
df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: specialchar_pattern.sub(r'', x))
df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: number_pattern.sub(r'', x))
df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: space_pattern.sub(r'', x))
df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: dot_pattern.sub(r'', x))
df_kmeans['texts'] = df_kmeans['texts'].apply(lambda x: backslash_pattern.sub(r'', x))
df_kmeans['texts_tokenized'] = df_kmeans['texts'].apply(lambda x: tokenize_to_list(x))
df_kmeans['texts_count'] = df_kmeans['texts_tokenized'].apply(lambda x: collections.Counter(x).most_common())
```


```python
#results of tokenization

df_kmeans
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
      <th>KMeans ID</th>
      <th>texts</th>
      <th>texts_tokenized</th>
      <th>texts_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>เป็นคนที่ชอบทานMacchiatoเป็นประจำมีวันนึงเดArt...</td>
      <td>[คน, ชอบ, Macchiato, เป็นประจำ, นึง, เด, Artof...</td>
      <td>[(ร้านกาแฟ, 25), (กาแฟ, 23), (ชอบ, 11), (คาเฟ่...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>กวงทะเลเผาอาหารทะเลเค้าสดจริงๆเนื้อปูหวานไม่คว...</td>
      <td>[กวง, ทะเล, เผา, อาหารทะเล, เค้า, สด, เนื้อ, ป...</td>
      <td>[(ร้านอาหาร, 14), (อร่อย, 11), (กก, 10), (อาหา...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>เคยเป็นไหมกันไหมคะหลังอาหารมื้อใหญ่ต่อให้อิ่ร้...</td>
      <td>[ไหม, ไหม, หลังอาหาร, มื้อ, ต่อให้, อิ่, ร้านก...</td>
      <td>[(ชา, 18), (นม, 14), (ไข่มุก, 14), (เครื่องดื่...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#show top keywords of each cluster

top_N_words = 10

for i in range(0, len(df_kmeans)):
  print(f"Cluster ID : {i}\n")
  print(f"Most common words include : {list(df_kmeans['texts_count'][i])[:top_N_words]}\n")

#tune a model by remove unwanted characters and words and add more words to a custom dictionary
```

    Cluster ID : 0
    
    Most common words include : [('ร้านกาแฟ', 25), ('กาแฟ', 23), ('ชอบ', 11), ('คาเฟ่', 6), ('น', 6), ('แวะ', 6), ('ดี', 6), ('รี่', 5), ('อร่อย', 5), ('น่ารัก', 5)]
    
    Cluster ID : 1
    
    Most common words include : [('ร้านอาหาร', 14), ('อร่อย', 11), ('กก', 10), ('อาหาร', 9), ('รีวิว', 7), ('บ้าน', 6), ('ส้มตำ', 6), ('ซอย', 6), ('สาขา', 6), ('เพื่อน', 5)]
    
    Cluster ID : 2
    
    Most common words include : [('ชา', 18), ('นม', 14), ('ไข่มุก', 14), ('เครื่องดื่ม', 4), ('รีวิว', 4), ('ร้า', 3), ('น้ำ', 3), ('ตั้งอยู่', 3), ('ลอง', 3), ('เดิน', 3)]
    


## Step 3 - Result Discussion

### Cluster 0 : ร้านกาแฟ และ คาเฟ่
คำใกล้เคียง :
-  ('ร้านกาแฟ', 25)
-  ('กาแฟ', 23)
-  ('ชอบ', 11)
-  ('คาเฟ่', 6)
-  ('น', 6)
-  ('แวะ', 6)
-  ('ดี', 6)
-  ('รี่', 5)
-  ('อร่อย', 5)
-  ('น่ารัก', 5)


```python
df[df["KMeans ID"] == 0].Review
```




    0      เป็นคนที่ชอบทาน Macchiato เป็นประจำ มีวันนึงเด...
    1      Art of Coffee Kasetsart เป็นร้านกาแฟรสชาติเยี่...
    5      เป็นร้านที่สะดุดตาที่สุดบนถนนจรัญ เลยก็ว่าได้ ...
    7      Starbucks \nเกี่ยวกับร้าน: \nร้าน Starbucks นี...
    8      \nร้านเบอเกอรี่ร้านนี้อยู่ในร้านล้างรถชื่อว่า ...
                                 ...                        
    292    ร้านกาแฟน่ารักๆ ในปั๊มนํ้ามันปิโตรนาสบนเส้นทาง...
    296    ร้านสะอาดดี ตกแต่งสวยงาม มีที่จอดรถ ราคาเมนูต่...
    297    เช้าๆ รีบๆ วิ่งมาเข่าห้องเรียนแทบไม่ทันแต่ต้อง...
    298    ร้านนี้เป็นร้านกาแฟเล็กๆ ข้างๆ ร้านๆ Happy Man...
    299    ทรูคอฟฟี่สาขาซีคอนอยู่ในศูนย์บริการของทรู ชั้น...
    Name: Review, Length: 130, dtype: object



### Cluster 1 : ร้านอาหาร
คำใกล้เคียง
-  ('ร้านอาหาร', 14)
-  ('อร่อย', 11)
-  ('อาหาร', 9)
-  ('รีวิว', 7)
-  ('บ้าน', 6)
-  ('ส้มตำ', 6)
-  ('ซอย', 6)
-  ('สาขา', 6)
-  ('เพื่อน', 5)


```python
df[df["KMeans ID"] == 1].Review
```




    2      กวงทะเลเผา อาหารทะเลเค้าสดจริงๆเนื้อปูหวานไม่ค...
    3      วันนี้มีโอกาสตื่นเช้าครับเลยถึงโอกาสออกมาหาอะไ...
    4      ชอบมาทานร้านนี้ถ้าอยากกินอาหารเวียดนามใกล้บ้าน...
    6      สวัสดีเพื่อนๆสมาชิกนักรีวิวร้านอาหารชาว Wongna...
    9      ร้านส้มตำเจ๊พิมอยู่ถนนเลี่ยงตลาดหนองมน\nเป็นร้...
                                 ...                        
    286    สวัสดีค่ะ ชื่อร้านนี้อาจจะน่ากลัวสักหน่อยนะคะ ...
    289    อร่อยดีครับ เพิ่งมาครั้งแรกขับรถหลงๆ มาเห็นร้า...
    291    ร้านนี้เป็นหนึ่งในความตั้งใจว่า อยากจะทำเรื่อง...
    293    ร้านนี้หาไม่ยากครับ อยู่ที่ปากซอย6 ถนนนิมมาน \...
    295    ค่ำนี้คุณเพื่อนอยากส้มตำ หมูเฮาเลยพากันลงมากิน...
    Name: Review, Length: 118, dtype: object



## Cluster 2 : ร้านชานมไข่มุก
คำใกล้เคียง
- ('ชา', 18)
-  ('นม', 14)
-  ('ไข่มุก', 14)
-  ('เครื่องดื่ม', 4)
-  ('รีวิว', 4)
-  ('ร้า', 3)
-  ('น้ำ', 3)
-  ('ตั้งอยู่', 3)
-  ('ลอง', 3)
-  ('เดิน', 3)


```python
df[df["KMeans ID"] == 2].Review
```




    13     เคยเป็นไหมกันไหมคะ หลังอาหารมื้อใหญ่ ต่อให้อิ่...
    19     ร้านกาแฟร้านนี้ เมื่อเข้าไปแล้วบรรยายกาศจะออกแ...
    20     เช้าๆบ่ายๆหรือเย็นๆ อาหารว่างที่คนเรามักหยิบจั...
    23     ระหว่างมุ่งหน้าสู่ชัยภูมิ เราแวะเติมพลังกันที่...
    29     เป็นร้านเครื่องดื่มขนาดกระทัดรัด พื้นที่น้อย เ...
    30     ร้านขายเครื่องดื่มผลิตภัณฑ์ชาเขียว ไอศรีมชาเขี...
    31     ชานมไข่มุก จริงๆคืออ้วนมาก\nไม่ค่อยอยากกินเลย\...
    34     เดินผ่านร้านนี้มาหลายครั้งแล้วค่ะ สังเกตว่าร้า...
    42     ร้านอยู่โซนซุ้มขายของเล็กๆตรงกลางนะคะ ใกล้ๆกับ...
    44     บนห้าง Big-C ราชดำริ นิวคุงได้ไปเจอร้านขายน้ำท...
    47     ...น่าจะใช่นะ Ochaya มีสาขาเยอะมากๆทั้งตึกออฟฟ...
    58     ร้าน April store เป็นร้านกาแฟขนาดเล็ก ตั้งอยู่...
    61     วันนี้มาเดินเที่ยวสยามคะ ถ้าชาไข่มุกอร่อยๆต้อง...
    66     ร้าน April store อยู่บริเวณสวนของ ยู เซ็นเตอร์...
    74     วันก่อนนิวคุงไปเดินเล่นหาอะไรกินในย่านหอการค้า...
    84     วันนี้ได้ฤกษ์มารีวิวร้านชานมไข่มุก ร้านที่โปรด...
    87     เป็นคนชอบทานชาไข่มุก ลองมาแล้วหลายร้าน ติดใจรส...
    90     ร้าน A Little Sweet เป็นร้านของหวานสไตล์ desse...
    93     \n\n\n\n\nร้าน Dakasi เป็นร้านเครื่องดื่มครับ ...
    97     ร้านวโรชา เป็นร้านก๋วยเตี๋ยวเล็กๆ ที่เปิดมานาน...
    106    ร้านชานมไข่มุกร้านนี้ที่เราจะมาแนะนำมีชื่อร้าย...
    107    จากโปรโมชั่นของ openrice ทำให้ได้ลองชิมชายี่ห้...
    121    มาเดินโฮมโปร กับหลาน คุณหลานเกิดอยากดื่มชานมหว...
    122    Tea Story ร้านชานมไข่มุกกิ๊บเก๋ตั้งอยู่กลางห้า...
    131    เป็นร้านที่ตั้งอยุตรงถนนนิมมานเลย เห็นแต่ไกล ร...
    132    เครปพันชั้น! ไปมาหลายหนแล้ว แต่ไม่เคยรีวิว\nจั...
    133    เล่าตอนที่ไปถึงร้าน\nร้าน นมโจ 100% เป็นร้านขา...
    134    ของหวานฝรั่งเศสสไตล์ญี่ปุ่น เป็นอาหารที่ถูกจับ...
    137    อยากจะบอกไปตามตรงเลยว่า ร้านนี้แหล่ะร้านชานมไข...
    146    ข้อมูลที่ลงรีวิวเป็นความเห็นส่วนตัวที่ได้พบเจอ...
    148    ร้านไอศครีมน่ารักๆ เหมาะกับพาครอบครัวมานั่งทาน...
    152    เป็นร้านที่แวะเติมความหวานบ่อยมาก\n>>Location\...
    166    ร้านนี้เป็นแหล่งรวมเครื่องดื่มนานาชนิดทั้งนม ช...
    173    จะว่าไปผมก็ไม่ได้ทานชาไข่มุกมานานเหมือนกันครับ...
    175    ลักษณะร้าน : เป็นร้านนม ที่ขายในช่วงเวลาเย็นๆถ...
    176    เมื่อวานไปเดิน Terminal 21 ค่ะ อยากลองอะไรใหม่...
    189    ร้าน ไออุ่น ชาไข่มุก มีนํ้าปั่นหลากหลายรสชาติ ...
    190    Wawa Cha ร้านชานมไข่มุกจากไต้หวันภายใต้การบริห...
    192    [สถานที่ตั้ง]\nThe Mall Bangkapi Fl.4 (ตรงข้าม...
    200    ร้านอยู่ Big C สุขาภิบาล 5 ติดร้าน Oriental\nP...
    211    ร้านนี้เป็นร้านขายขนมปังปิ้งแล้วก็น้ำปั่นอยู่ห...
    219    ประโยชน์ของน้ำแตงโมปั่น\nแตงโม มีมากมายหลายพัน...
    234    ทานชาเขียวกีวี่+ไข่มุกเป็นประจำค่ะ \nทานกี่สาข...
    236    ร้านนี้เป็นร้านเล็กๆที่ตั้งอยู่เเถวสยาม ตรงข้า...
    238    นึกถึงชานมไข่มุกต้นตำหรับไต้หวันแท้ๆ ต้องนึกถึ...
    239    ร้านนี้มีรีวิวมากมายยย....เลยตามรอยเหล่านักรีว...
    242    Dakasi ร้านชานมไข่มุกสูตรไต้หวันที่ตอนนี้มีสาข...
    250    นานๆ ทีจะได้มาทำธุระที่เซ็นทรัลปิ่นเกล้า เดินๆ...
    258    Fuku matcha อยู่ชั้นเดียวกับโรงหนัง sf เดอะมอล...
    271    หลายคนคงเคยได้ยินข่าวว่าทานชานมไข่มุกมากๆไม่ดี...
    287    มา Ochaya สั่งแบบเมนู basic เลย ชานมไข่มุก ราค...
    294    ร้านบ้านไร่กาแฟเป็นร้านติด BTS เอกมัย ร้านใหญ่...
    Name: Review, dtype: object


