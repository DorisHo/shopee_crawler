# 蝦皮商品資料分析
## 簡介
1. 目標：
	- 了解產品資料中的特徵（例如：店家評價）是否會影響安卓手機銷量
	- 了解不同品牌在蝦皮商城中的銷售概況
2. 資料來源：用爬蟲的方式獲取蝦皮商城中的安卓手機商品資料
3. 開發環境：Google Colab／Python 3

## 實作
### 套件
```python
import csv
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn import preprocessing
```
### 資料獲取
把 User-Agent 設為 Googlebot 才可被蝦皮網頁伺服器認可身份
```python
headers = {
    "User-Agent": "Googlebot"
}
```
分成幾個部分，分層獲取商品資料
- 獲取單頁搜尋結果的所有商品網址<br>
<code>url</code> : 單頁搜尋結果網址

```python
def goods(url):
  # allow_redirects=False 避免網站被重新導向
  resp = requests.get(url, headers=headers, allow_redirects=False)
  soup = BeautifulSoup(resp.text, "lxml")
  goods_item = soup.find_all("div", "col-xs-2-4 shopee-search-item-result__item")

  item_list = []
  for i in goods_item:
    item_list.append({
        "Title": i.find("div", "O6wiAW").getText(),
        "Link": "https://shopee.tw/" + i.find("a").get("href")
    })

  return item_list
 ```
- 獲取單個商品頁面中的資訊（商品名稱、價格、品牌、銷售量、評價星級、評價數量、賣家）<br>
<code>goods_url</code> : 商品頁面網址
```python
def goods_info(goods_url):
  resp_goods = requests.get(goods_url, headers=headers, allow_redirects = False)
  soup = BeautifulSoup(resp_goods.text, "lxml")

  goods = {} # 儲存商品資訊(dict)
  # 商品名稱
  goods["Title"] = soup.find("div", "qaNIZv").find("span").text
  # 價格
  goods["Price"] = soup.find("div", "_3n5NQx").text.split(" -")[0].lstrip("$").replace(",","")
  # 品牌
  if soup.find("a", "_2H-513") != None:
    goods["Brand"] = soup.find("a", "_2H-513").text 
  else:
    goods["Brand"] = "其他"
  # 銷售量  
  if "萬" in soup.find("div", "_22sp0A").text:
    goods["Sales_Volume"] = int(soup.find("div", "_22sp0A").text.split("萬")[0].replace(".",""))*10000
  else:
    goods["Sales_Volume"] = soup.find("div", "_22sp0A").text.replace(",","") 
  # 評價星級 & 評價數量
  if soup.find("div", "_3Oj5_n") != None:
    if "萬" in soup.find_all("div", "_3Oj5_n")[1].text:
      goods["Reviews_Num"] = int(soup.find_all("div", "_3Oj5_n")[1].text.split("萬")[0].replace(".",""))*10000
    else:
      goods["Reviews_Num"] = soup.find_all("div", "_3Oj5_n")[1].text.replace(",","")
    goods["Star"] = soup.find("div", "_3Oj5_n _2z6cUg").text
  else:
    goods["Reviews_Num"] = 0
    goods["Star"] = 0

  # 賣家
  tmp = eval(soup.find_all("script")[1].getText()) # eval(): 把字串換成字典型式
  goods["Store_Name"] = tmp["offers"]["seller"]["name"]
  goods["Store_RateCount"] = tmp["offers"]["seller"]["aggregateRating"]["ratingCount"]
  goods["Store_RateValue"] = tmp["offers"]["seller"]["aggregateRating"]["ratingValue"]

  return goods
```
- 獲取單頁搜尋結果中，所有的商品資訊<br>
<code>response</code> : 單頁搜尋結果中的所有商品網址
```python
def get_goods(response):
  goods_detail_list = []

  for x in response:
    goods_detail = goods_info(x["Link"])
    goods_detail_list.append(goods_detail)
  
  return goods_detail_list
```
- 獲取所有商品資料<br>
觀察搜尋結果的網址結構，不同頁面之間的差異在<code>page=</code>後的數字，第一頁為 0 。
```python
goods_detail_list = []
goods_list_url = "https://shopee.tw/Android%E7%A9%BA%E6%A9%9F-cat.70.2609?newItem=true&officialMall=true&page="

for i in range(8):
  page_url = goods_list_url + str(i) + "&sortBy=pop"
  response = goods(page_url)
  goods_detail = get_goods(response)
  goods_detail_list.extend(goods_detail)

df = pd.DataFrame(goods_detail_list)
# print("總共印出%d筆" %(len(goods_detail_list)))
```
### 資料處理
- 資料型態轉換
```python
df["Price"] = df["Price"].astype("int")
df["Brand"] = df["Brand"].astype("category")
df["Sales_Volume"] = df["Sales_Volume"].astype("int")
df["Reviews_Num"] = df["Reviews_Num"].astype("int")
df["Star"] = df["Star"].astype("float")
df["Store_RateCount"] = df["Store_RateCount"].astype("int")
df["Store_RateValue"] = df["Store_RateValue"].astype("float")
```
- 資料清洗<br>
資料完整度高，僅把未寫品牌的列刪除
```python
df = df[~df["Brand"].isin(["其他"])]
df.info()
```
- 把原始資料整理成我們所需的資料，取平均值或是總數等
```python
# 商品資料
df_goods = df[["Title", "Price", "Brand", "Sales_Volume", "Star", "Reviews_Num", "Store_Name"]]
# 商店資料
df_store = df[["Store_Name", "Store_RateCount", "Store_RateValue"]]
df_store = df_store.drop_duplicates().reset_index(drop=True)
# 品牌資料
groupB = df_goods.groupby("Brand")
df_brand = pd.DataFrame(groupB["Sales_Volume"].sum())
df_brand["Brand_Sales"] = groupB["Sales_Volume"].sum() # 銷售量
df_brand["Brand_Mean_Price"] = round(groupB["Price"].mean(),2) # 平均價格
df_brand["Brand_Mean_Star"] = round(groupB["Star"].mean(),1) # 平均評價星級
df_brand["Brand_Reviews_Num"] = groupB["Reviews_Num"].sum() # 評論數量
df_brand["Store_Num"] = groupB["Store_Name"].nunique() # 販售店家數量
df_brand.drop(columns=["Sales_Volume"], inplace=True)
df_brand.drop(["其他"], axis=0, inplace=True)
```
- 品牌資料標準化
後面會有品牌排名，將資料標準化後較好去做比較
```python
df_brand_copy = df_brand.copy()
df_brand_copy["Brand_Mean_Price"] = df_brand_copy["Brand_Mean_Price"]*(-1)
min_max = preprocessing.MinMaxScaler(feature_range=(1, 10))
min_max_data = np.around(min_max.fit_transform(df_brand_copy), 2)
brand_process = pd.DataFrame(min_max_data, index=df_brand_copy.index, columns=df_brand_copy.columns)
brand_process["Nor_sum"] = brand_process.sum(axis=1)
```
### 視覺化分析
- 中文字體
```python
# 安裝字體
!wget "https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKtc-hinted.zip"
!unzip "NotoSansCJKtc-hinted.zip"
!mv NotoSansCJKtc-Regular.otf /usr/share/fonts/truetype/
# 使用中文字體
import matplotlib.font_manager as fm
path = "/usr/share/fonts/truetype/NotoSansCJKtc-Regular.otf"
fontProp = fm.FontProperties(fname=path, size=10)
```
- 熱圖 : 顯示商品資料相關性
```python
cor = df.corr()
fig, ax = plt.subplots(figsize=(13,13))
# 熱圖
plt.title("商品資料相關性", fontproperties=fontProp, size=20)
sns.heatmap(cor, # 使用資料
            annot=True, # True: 顯示相關係數
            square=True, # 圖案是否為正方形
            cmap="Greens", # 顏色主題
            ax=ax, # 軸
            annot_kws={"size":15}) # 相關係數字的樣式調整
            
# x 軸名稱放到上面
ax.xaxis.tick_top()
# 設定軸名稱方向與大小
ax.set_xticklabels(cor.index, rotation=0, fontsize=11)
ax.set_yticklabels(cor.index, rotation=0, fontsize=11)
```
- 品牌價格/銷量散點圖
	1. x軸 : 價格
	2. y軸 : 銷量
	3. 散點大小與「評價數量、評價星級」成正比
```python
figB, axB = plt.subplots(figsize=(13,13))

xs = df_brand["Brand_Mean_Price"]
ys = df_brand["Brand_Sales"]
labels = df_brand.index
x_max = xs.max()+1000
y_max = ys.max()+200

plt.title("品牌價格/銷量散點圖", fontproperties=fontProp, size=20)
plt.xlabel("Price", size=15)
plt.ylabel("Sales", size=15)
plt.axis([0, x_max, -200, y_max])
plt.axvline(xs.mean(), color="y", linestyle="--", label="Mean Price", alpha=0.8) # 平均線
plt.axhline(ys.mean(), color="r", linestyle="--", label="Mean Sales", alpha=0.7) # 平均線

axB.fill_between([xs.mean(), x_max], ys.mean(), y_max, alpha=0.35, color="#add01f")
axB.fill_between([0, xs.mean()], -200, ys.mean(), alpha=0.25, color="#add01f")
s = (brand_process["Brand_Reviews_Num"]+brand_process["Brand_Mean_Star"])**2 # 散點大小
plt.scatter(xs, ys, s=s, color="c", alpha=0.8)
plt.grid(True, linestyle="-.") # 網格

for label, x, y in zip(labels, xs, ys):
  plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 15), ha="center", va="top", fontproperties=fontProp, alpha=0.8)

plt.legend()
plt.show()
```
- 品牌排名長條圖
    1. 把品牌標準化資料的所有特徵數值加總，做個排序。
    2. 不同特徵用不同色塊表示，了解不同品牌的優劣勢。
```python
figRank, axRank = plt.subplots(figsize=(13,13))

plt.title("品牌排名" ,fontproperties=fontProp, size=20)
brand_process_copy = brand_process.sort_values(by="Nor_sum") 
x = brand_process_copy.index
color = ["#eecc50", "#a2ee50", "#50acee", "#9a50ee", "#ee507f"]
axRank.set_yticklabels(x, fontproperties=fontProp, size=12)

for i in range(5):
  plt.barh(x, 
           brand_process_copy[brand_process_copy.columns[i]], # 色塊寬
           color=color[i], # 顏色
           left=brand_process_copy[brand_process_copy.columns[:i]].sum(axis=1), # 從哪個座標開始(左邊開始算)
           label=brand_process_copy.columns[i], # 圖例名稱
           height=0.5, # 粗細度
           alpha=0.8) # 透明度

plt.legend() # 圖例
```
