# Jane Street Market Prediction

[Jane Street](https://www.janestreet.com/), the sponsor of this [competition](https://www.kaggle.com/c/jane-street-market-prediction), is a *quantitative trading firm with a unique focus on technology and collaborative problem solving*. The challenge is to build a model that receives short-term trade opportunities and decides for each one whether to act and execute the opportunity or to dismiss it.

It's a tough forecasting problem given..


* the fact that a strategy that works well with past data is unlikely to do so in the future. The relationship of the features with the response is constantly changing. It's difficult to avoid overfitting.


* the uncertainty introduced by volatility. Different volatility regimes require different strategies.


* a low signal-to-noise ratio and multicollinearity that could hinder the learning process.


* a time constraint during inference. The inference is done in "real time" as opportunities come up one at a time in a loop. The average iteration should not exceed (roughly) 16 ms.


The code of this project can be found in [here](https://github.com/codefluence/jane_street).

<br>

### EDA

The data for the training consists of around 2.5 million data points spanning 500 days (about 2 years). The final test data will be disjoint in time, it's in the future and spans roughly 1 year.

The trade opportunities are described by 130 anonymized market data features, all of them continuous except for a binary feature with values -1 and +1 (very likely the **side** of the trade opportunity: buy or sell).

In addition, each trade opportunity has a **date id** and a **trade opportunity id** (also anonymized but in chronological order according to Jane Street).

There is no way to link different trade opportunities to the same security, so even if the data has a time component this problem is not a pure time series problem.

**weight** is another variable provided by Jane Street, representing the weight of selected trades in the final score. It could be seen as the quantity of the trade. Around 17% of the trade opportunities have weight = 0, these have no impact in the final score but could be useful during training.

The **response** variable (resp) represents the profit/loss per security unit in some specific fixed time horizon determined by Jane Street. The profit/loss of a trade is then weight * response.

Responses at other time horizons (resp_1, resp_2, resp_3, resp_4) are also available but these won't be used in the evaluation metric.

    
![png](./img/output_5_1.png)
    


All time horizons have a positive trend throughout the two years (financial markets usually have a positive trend in the long term). resp1 and resp2 carry less volatility and lower returns which suggests their time horizons are shorter. Resp3 is somewhere in the middle and resp4 is close to resp but probably further in time.

There is a negative correlation between the absolute value of response and weight.


```python
weights = StandardScaler().fit_transform(np.array(df.weight).reshape(-1, 1))
resp_abs = StandardScaler().fit_transform(np.array(abs(df.resp)).reshape(-1, 1))
```


```python
fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(weights, resp_abs, cmap='rainbow', s=0.0005)
ax.set_xlim((-1, 15))
ax.set_ylim((-1, 15))
ax.set_xlabel ("weight (scaled)", fontsize=12)
ax.set_ylabel ("abs(response) (scaled)", fontsize=12)

plt.show()
```


    
![png](output_8_0.png)
    


The potential size of the response is related to its volatility. Jane Street is putting less weight on risky opportunities - it's difficult to recover from big losses. The criteria in the risk measurement is unknown but probably related to historical volatility and liquidity.

The following plots show the evolution of returns and the total weight allocated by Janes Street day by day (500 days).


```python
df['feature_0'] = df.feature_0 * -1
df['alloc'] = df.feature_0 * df.weight
df['cap'] = df.weight * df.resp

SELL = df.feature_0 == -1
BUY = df.feature_0 == 1
```


```python
fig, (ax_1,ax0,ax1,ax2,ax3) = plt.subplots(5,1,figsize=(22,18))

ax_1.set_title('Cumulative return',fontsize=14)
df.groupby('date')[['cap']].sum().cumsum().plot(ax=ax_1,color=['black'],alpha=.8)
df[BUY].groupby('date')[['cap']].sum().cumsum().plot(ax=ax_1,color=['steelblue'],alpha=.8)
df[SELL].groupby('date')[['cap']].sum().cumsum().plot(ax=ax_1,color=['darkorange'],alpha=.8)
ax_1.axhline(0,color='black',ls='--',linewidth=0.5)
ax_1.legend(loc='lower left', frameon=False, labels=('TOTAL','BUY','SELL'))

ax0.set_title('Daily return',fontsize=14)
df.groupby('date')[['cap']].sum().plot(ax=ax0,color=['black'],alpha=.8)
df[BUY].groupby('date')[['cap']].sum().plot(ax=ax0,color=['steelblue'],alpha=.8, label='lol')
df[SELL].groupby('date')[['cap']].sum().plot(ax=ax0,color=['darkorange'],alpha=.8, label='lol')
ax0.legend(loc='lower left', frameon=False, labels=('TOTAL','BUY','SELL'))
ax0.axhline(0,color='black',ls='--',linewidth=0.5)

ax1.set_title('Allocation sum',fontsize=14)
df.groupby('date')[['alloc']].sum().plot(ax=ax1,color=['black'],alpha=.8)
df[BUY].groupby('date')[['alloc']].sum().plot(ax=ax1,color=['steelblue'],alpha=.8)
df[SELL].groupby('date')[['alloc']].sum().plot(ax=ax1,color=['darkorange'],alpha=.8)
ax1.axhline(0,color='black',ls='--',linewidth=0.5)
ax1.legend(loc='lower right', frameon=False, labels=('TOTAL','BUY','SELL'))

ax2.set_title('Allocation standard deviation', fontsize=14)
df.groupby('date')[['alloc']].std().plot(ax=ax2,color=['black'],alpha=.8)
df[BUY].groupby('date')[['alloc']].std().plot(ax=ax2,color=['steelblue',],alpha=.8)
df[SELL].groupby('date')[['alloc']].std().plot(ax=ax2,color=['darkorange'],alpha=.8)
ax2.axhline(df.weight.std(0),color='black',ls='--',linewidth=0.5)
ax2.legend(loc='lower right', frameon=False, labels=('TOTAL','BUY','SELL'))

ax3.set_title('Number of opportunities', fontsize=14)
df.groupby('date')[['weight']].count().plot(ax=ax3,color=['black'],alpha=.8)

ax_1.set_xlabel('')
ax0.set_xlabel('')
ax1.set_xlabel('')
ax2.set_xlabel('')
ax3.set_xlabel('')

plt.plot()
```




    []




    
![png](output_11_1.png)
    


A couple of observations:

* Buy opportunities are more profitable in the long term. Still, sell opportunities help to reduce risk in the short term through hedging.


* It looks Jane Street increases the allocated total weight in the opportunities during volatile periods.

<br>


```python
day_1 = df.loc[df['date'] == 10]
day_2 = df.loc[df['date'] == 126]
day_3 = df.loc[df['date'] == 275]
day_4 = df.loc[df['date'] == 440]

three_days = pd.concat([day_1, day_2, day_3, day_4])
three_days['tid'] = np.arange(three_days.shape[0])
wdif = max(three_days.weight) - min(three_days.weight)

BUY = three_days.feature_0 == -1
SEL = three_days.feature_0 == 1

A = 0.5; B = 50

def show_feature(i):
    
    fig, axs = plt.subplots(figsize=(22, 4))
    
    axs.scatter(three_days[BUY].tid, three_days[BUY].iloc[:,i], s=A+(B-A)*(three_days[BUY].weight)/wdif, color='royalblue', alpha=0.66)
    axs.scatter(three_days[SEL].tid, three_days[SEL].iloc[:,i], s=A+(B-A)*(three_days[SEL].weight)/wdif, color='indianred', alpha=0.66)
    axs.axvline(day_1.shape[0],color='black',ls='--',linewidth=0.5)
    axs.axvline(day_1.shape[0]+day_2.shape[0],color='black',ls='--',linewidth=0.5)
    axs.axvline(day_1.shape[0]+day_2.shape[0]+day_3.shape[0],color='black',ls='--',linewidth=0.5)
    axs.axhline(0,color='black',ls='--',linewidth=0.5)
    axs.set_ylabel('feature_'+str(i-7),fontsize=14)
    
    if i >= 84:
        axs.set_ylim([-15, 15])
        
    plt.show()
```

<br>

The anonymous features can be classified in groups based on trend, heteroscedasticity and pattern.

The following plots show an example for each group, with values from four different days with different market conditions (bullish/bearish, volatile/nonvolatile). The color represents the side of the trade (red: sell, blue: buy). The size of the dot represents the weight.

* With no trend and no heteroscedasticity: 1, 2, 9, 10, 15, 16, 19, 20, 25, 26, 29, 30, 35, 36, 46-52, 69-76, 79-82


```python
show_feature(7+15)
```


    
![png](output_16_0.png)
    


* With no trend and with heteroscedasticity: 3-8, 11-14, 17, 18, 21-24, 27, 28, 31-34, 37-40, 77-78, 83


```python
show_feature(7+28)
```


    
![png](output_18_0.png)
    


* With trend and no heteroscedasticity: 109, 112, 115, 122-129


```python
show_feature(7+125)
```


    
![png](output_20_0.png)
    


* With trend and heteroscedasticity: 53-59, 84, 89, 90, 95, 96, 101, 102, 107, 108, 110, 111, 113, 114, 116-121


```python
show_feature(7+108)
```


    
![png](output_22_0.png)
    


* With a "stratified" pattern (probably related to price or tick size values): 41-45


```python
show_feature(7+45)
```


    
![png](output_24_0.png)
    


* With a time pattern (feature 64 seems to represent time - note that the opening and the closing of the market are busier): 60-68


```python
show_feature(7+64)
show_feature(7+65)
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    


* With a pattern where one of the sides is "fixed" around a specific value: 85-88, 91-94, 97-100, 103-106


```python
show_feature(7+91)
```


    
![png](output_28_0.png)
    


<br>

### Utility score

As described in the [evaluation description](https://www.kaggle.com/c/jane-street-market-prediction/overview/evaluation) of the competition:

*This competition is evaluated on a utility score. Each row in the test set represents a trading opportunity for which you will be predicting an action value, 1 to make the trade and 0 to pass on it. Each trade j has an associated weight and resp, which represents a return.*

*For each date i, we define:*

$p_i = \sum_j(weight_{ij} * resp_{ij} * action_{ij})$

$t = \frac{\sum p_i }{\sqrt{\sum p_i^2}} * \sqrt{\frac{250}{|i|}}$

*where  is the number of unique dates in the test set. The utility is then defined as:*

$u = min(max(t,0), 6)  \sum p_i$

In summary, picking up profitable trades is not good enough to get a high score, it's also important that the profit is evenly distributed across time. Having a bad trading day will be specially penalized.

The following plot shows the utility score map of the first half of the training data (1 year) when the response of the selected opportunities are within a [floor,ceiling] range.


```python
import torch

def utility_score(context, actions, device='cuda', mode='metrics'):

    # context columns: 'date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id', 'regime'
    dates, weights, resps = context[:,0], context[:,1], context[:,6]

    if mode == 'loss':
        # generalization
        resps = context[np.arange(context.shape[0]), 2+np.random.choice(5, context.shape[0])]
        resps = torch.normal(mean=resps, std=torch.abs(resps)/2)

    dates_involved = torch.unique(dates)
    daily_profit = []

    for d in dates_involved:
        pnl = torch.mul(torch.mul(weights,d==dates),resps).unsqueeze(dim=0)
        daily_profit.append(torch.matmul(pnl, actions))
        
    p = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    vol = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        
    for dp in daily_profit:
        p = p + dp
        vol = vol + dp**2
    
    t = p / vol**.5 * (250/len(dates_involved))**.5

    ceiling = torch.tensor(6, dtype=torch.float32, requires_grad=False, device=device)
    floor = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    t = torch.min(torch.max(t, floor), ceiling)

    # if profit is negative the utility score is not clipped to 0 in loss mode (for learning purposes)
    if mode == 'loss' and p < 0.0:
        u = p
    else:
        u = torch.mul(p, t)

    if mode == 'loss':
        return -u
    else:
        return t.cpu().item(), p.cpu().item(), u.cpu().item()
```


```python
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

X = np.arange(0, 0.04, 0.04/20)
Y = np.arange(0, 0.2, 0.2/20)
X, Y = np.meshgrid(X, Y)

resp = df.resp.to_numpy()[:len(df)//2]

context = df.iloc[:,np.concatenate((np.arange(7),[-1]))].to_numpy(dtype=np.float32)[:len(df)//2]
context = torch.tensor(context, dtype=torch.float32, requires_grad=False, device='cuda')

squarer_util = lambda x,y: utility_score(context, 
                torch.tensor(np.bitwise_and(x < resp , resp < y)*1, dtype=torch.float32, requires_grad=False, device='cuda'))[2]

squarer_profit = lambda x,y: utility_score(context, 
                torch.tensor(np.bitwise_and(x < resp , resp < y)*1, dtype=torch.float32, requires_grad=False, device='cuda'))[1]

vfunc_util = np.vectorize(squarer_util)
Z_util = vfunc_util(X,Y)
Z_util = np.nan_to_num(Z_util)

vfunc_profit = np.vectorize(squarer_profit)
Z_profit = vfunc_profit(X,Y)
Z_profit = np.nan_to_num(Z_profit)
```


```python
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z_util, rstride=1, cstride=1, ls='--')
ax.plot_wireframe(X, Y, Z_profit, rstride=1, cstride=1, color='green')

ax.set_xlabel('response floor')
ax.set_ylabel('response ceiling')
ax.set_zlabel('utility')
ax.legend(loc='lower left', frameon=False, labels=('utility','profit'))

plt.show()
```


    
![png](output_33_0.png)
    


<br>

### Feature correlation

The anonymized features form clear correlation clusters but none of them have significant correlation with the response (first row):


```python
corr_t = df.iloc[np.random.choice(np.arange(2), p =[0.9, 0.1], size=len(df)) == 1, np.concatenate((np.array([6]),np.arange(7,137)))].corr(method='pearson')

fig = plt.figure(figsize=(14,12))
ax = plt.subplot(1,1,1)
sns.heatmap(corr_t,ax= ax, cmap='coolwarm')
sns.set(font_scale=1)

plt.show()
```


    
![png](output_35_0.png)
    


The correlation of features to the response keeps constantly changing. The plot below shows two of the features that are more correlated to the response in the first 200 day. The variance is high and both feature correlations to the response seem uncorrelated with each other.


```python
days = pd.DataFrame(df.groupby('date')[['resp','feature_25']].corr())['feature_25']
days.index=np.arange(1000)
days25=days.iloc[::2]
days25.index=np.arange(500)

days = pd.DataFrame(df.groupby('date')[['resp','feature_7']].corr())['feature_7']
days.index=np.arange(1000)
days18=days.iloc[::2]
days18.index=np.arange(500)

fig, axo = plt.subplots(1,1,figsize=(20,4))

axo.set_title('correlation to response',fontsize=14)
days18[:200].plot(ax=axo, alpha=.8, color=['firebrick'])
days25[:200].plot(ax=axo, alpha=.8, color=['royalblue'])
axo.axhline(0,color='black',ls='--',linewidth=0.5)
axo.legend(loc='lower left', frameon=False, labels=('feature_25','feature_7'))
axo.set_xlabel('day')

plt.show()
```


    
![png](output_37_0.png)
    


<br>

### Missing values and imputation

Missing values seem to follow a time pattern. The same set of features are always blank at the beginning of the day and during what it seems a market break. The missing values are probably caused by missing historical data required to compute the values.

I used the mean of the previous 100 trade opportunities (for the same feature_0 value) for imputation, kind of a moving average imputation.


```python
msno.matrix(df.loc[df['date'] == 0], color=(0.35, 0.35, 0.75))
plt.show()
```


    
![png](output_40_0.png)
    


<br>

### Day grouping for model validation

To fit the prediction models, the data was split in 4 folds, each one containing mutually exclusive days. The goal is to avoid day information leakage, since each day has its own particular mood and is impacted by financial news that only apply to that day. The model should not rely on the particular information of the day because it won't probably reproduce again in the future.

Between keeping training and validation days as much as isolated as possible (first x days are for training, last y days are for validation) and trying to include as much time variance in the validation set as possible (spreading validation days across the whole period) I went for the latter. In my experiments I didn't find obvious overfitting caused by mixing validation days with training days. The critical part is to avoid trade opportunities from the same day on both training and validation.

```
1st partition:
training days: 0, 1, 2,  4, 5, 6,  8, 9, 10, ...
validation days: 3, 7, 11, ...

2nd partition:
training days: 1, 2, 3,  5, 6, 7,  9, 10, 11, ...
validation days: 0, 4, 8, ...

etc
```



<br>

### Trend classification model

In this prediction model the approach to decide the action to take is to determine the expected response trend. A positive expected trend would trigger the action to take the trade opportunity. The chosen model is a fully connected neural network with a cross-entropy loss function classifying opportunities with positive or negative trends. A more statistical model could also have been a good choice given the nature of the problem.

The **target** in the classification could be response > 0, however the response has a fragile relationship with the features and this relationship is constantly changing over time, so using too much information from the response could lead to overfitting easily. In order to mitigate this I used instead the mean of the responses from all the time horizons provided. This value will be less noisy as it includes shorter time horizons easier to predict.

PCA and an encoder/decoder network to **denoise** and compress the original features were tested but with no significant change in the results so I decided to use the original features (scaled) straight away to avoid wasting the limited time per inference iteration. It seems that the network deals with noise and collinearity well.

**Feature engineering** is not obvious with anonymized data. I end up feeding the model with a few pairs of features multiplying each other. The idea is to help the learning with significant feature interactions. To select the pairs I trained several TabNet models. Interactions with the highest coefficients in TabNet explanations were selected (6 in total). The addition of these interactions seemed to help with the score a bit in both local CV and public score.

**Generalization** is always key in any prediction problem, but performance will be specially sensitive to overfitting with financial data. Some decisions to take care of generalization were:

* The model limited to 3 layers with the following setup:

```
batch_norm -> fully_connected -> leaky_relu -> dropout
```

* Dropout set to high rates (0.35 in the 1st layer, 0.4 in the 1ns layer, 0.45 in the 1th layer). The same setup has been tested with lower dropout rates (0.2) and the validation score was just slightly better, so I kept the extra regularization.


* Blending of pairs of data points as proposed in the paper [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412v2.pdf). This way the new data points fill the "empty" space in the training data, which helps to soft the fitting. Also, with this technique is possible to learn the information in data points with weight=0. The blending proportion follows a beta distribution. I chose parameter values to make the distribution close to a uniform in order to increase the level of "mixup".


```python
seq = np.arange(0,1,0.001)

plt.plot(seq, beta.pdf(seq, 0.8, 0.8))
plt.title('blending proportion distribution')
plt.ylim((0, 3))
plt.show()
```


    
![png](output_44_0.png)
    


The utility score is based on the weighting provided by Jane Street and the response. To bring the attention of the model to trade opportunities that will be more important in the score, data points are **weighted** in the cross-entropy loss function:

```
min(jane street weight * abs(mean(responses from all time horizons)), 0.4)
```


### Utility maximization model

Another completely different idea I tried was to plug the utility score function directly as loss function (multiplied by -1 to maximize).

Compared to the cross-entropy function, the utility score function already weights the importance of trade opportunities based on response and Jane Street weight. To soften the influence of the response values which would introduce quite a lot of overfitting, the utility score function is modified to randomly pick the response from different time horizons. This avoids overfitting in two ways: more predictable shorter time horizons and more time variability in the response.

On top of that, noise was added to the response in proportion to its magnitude (standard deviation = response/*2).

Same layers, dropout rates and data blending technique used in the classification network is applied here, but in another attempt to make this model different and still reduce overfitting, the following features were used instead of the original ones:

* A prediction of the general market direction of the day.


* A prediction of the general volatility of the day.


* Z-scores of the original features of the trade opportunity in regard to the means of the features of the previous 100 opportunities (for the same side, buy or sell).


To get the market trend/volatility of the day I fit linear models to the means/stds of the responses grouped by day.


```python
clf_day_trend = pickle.load(open('D:/docs/science/Machine Learning/jane_street/preprocessing/clf_day_trend.pkl','rb'))
clf_day_volat = pickle.load(open('D:/docs/science/Machine Learning/jane_street/preprocessing/clf_day_volat.pkl','rb'))

df['market_trend'] = -100 * df.feature_0 * df.resp
days = df.groupby('date').mean()
days['market_volat'] = df.groupby('date')[['market_trend']].std()

days['market_trend'] = StandardScaler().fit_transform(days['market_trend'].to_numpy().reshape(-1, 1))
days['market_volat'] = StandardScaler().fit_transform(days['market_volat'].to_numpy().reshape(-1, 1))

day_features = days.iloc[:,6:6+130]

pred_trend = StandardScaler().fit_transform(clf_day_trend.predict(day_features).reshape(-1, 1)).squeeze()
pred_volat = StandardScaler().fit_transform(clf_day_volat.predict(day_features).reshape(-1, 1)).squeeze()
```


```python
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,7))

ax1.set_title('daily response mean (scaled)',fontsize=14)
days['market_trend'].plot(ax=ax1,color=['dimgray'],alpha=.8)
pd.Series(pred_trend).plot(ax=ax1,color=['indianred'],alpha=.8)
ax1.legend(loc='lower right', frameon=False, labels=('ground truth','prediction'))

ax2.set_title('daily response standard deviation (scaled)',fontsize=14)
days['market_volat'].plot(ax=ax2,color=['dimgray'],alpha=.8)
pd.Series(pred_volat).plot(ax=ax2,color=['indianred'],alpha=.8)
ax2.legend(loc='lower right', frameon=False, labels=('ground truth','prediction'))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()
```


    
![png](output_48_0.png)
    


During inference, the means of the features of the last 100 trade opportunities (per side) is computed to feed to the linear models that will generate the predictions as features for the neural network. The error of these linear models is low because it's much easier to predict the market regime of the current day (information already contained in the historical data of the trades features) than to predict the trend and volatility of a single trade.

The hope is to provide the network model with some context to each trade opportunity so it's able to somehow adjust the risk depending on the market regime and the divergence of the trade from that regime (Z-scores).

The results with this model are close to the results with the trend classifier:

| Split | Trend classification | Utility maximization |
| --- | --- | --- |
| CV0 | auc=0.530, utility=2275.5 | auc=0.53-val, utility=1942.3 |
| CV1 | auc=0.531, utility=1317.1 | auc=0.52-val, utility=1356.7 |
| CV2 | auc=0.527, utility=0983.8 | auc=0.52-val, utility=0364.6 |
| CV3 | auc=0.524, utility=2777.6 | auc=0.52-val, utility=3411.9 |
