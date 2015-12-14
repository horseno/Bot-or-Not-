import pandas as pd
import urllib
import zipfile
import urllib2
import zipfile

##Download data files 
##It takes 1-2 minutes to download the bids set.

url = ['https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4294/train.csv.zip?sv=2012-02-12&se=2015-12-17T18%3A41%3A07Z&sr=b&sp=r&sig=1nKtdkTXmunUWNFLItzn6UH5Tutb7Y6MIOY4g0GOaGo%3D',
        'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/4294/bids.csv.zip?sv=2012-02-12&se=2015-12-17T18%3A48%3A57Z&sr=b&sp=r&sig=MjlxY5JPtdA4MuAGkww1dH0kaSKIs1sRsUbCO8JAR6g%3D']
filename = ["train.csv.zip", "bids.csv.zip"]

for fileIdx in range(2):
    response = urllib2.urlopen(url[fileIdx])

    with open(filename[fileIdx], 'w') as outfile:
        outfile.write(response.read())
    #unzip files 
    with zipfile.ZipFile(filename[fileIdx], "r") as z:
        z.extractall("./")


path= './'

dfTrain = pd.read_csv(path+'train.csv', index_col=False, header=0)
dfBids = pd.read_csv(path+'bids.csv', index_col=False, header=0)

# merging all features to this final dataframe 
df = dfTrain[['bidder_id','outcome']] 

### F1. Average Number of Bids in a Single Auction 

df_auction_bid = dfBids.groupby(['bidder_id','auction']).count().reset_index()[['bidder_id','auction','bid_id']]
df_avg_auction_per_bidder = df_auction_bid.groupby(['bidder_id']).mean().reset_index().rename(columns={'bid_id':'avg_bid_num'})
df = pd.merge(df_avg_auction_per_bidder,df, on='bidder_id')

### F2.Total Number of Bids Placed

totalBids = dfBids.groupby(['bidder_id']).count().reset_index()
totalBids = totalBids[['bidder_id','bid_id']].rename(columns={'bid_id':'total_bids'})
df = pd.merge(totalBids, df, on ='bidder_id')

### F3.Number of Unique countries that bids are placed in

df_country_bidder = dfBids[['bidder_id','country','bid_id']].groupby(['bidder_id','country']).count().reset_index()
df_country_bidder = df_country_bidder.groupby(['bidder_id']).count().reset_index()[['bidder_id','country']]
df = pd.merge(df_country_bidder,df, on='bidder_id')

### F4.Number of Unique IPs used 

df_ip_bidder = dfBids[['bidder_id','ip','bid_id']].groupby(['bidder_id','ip']).count().reset_index()
df_ip_bidder = df_ip_bidder.groupby(['bidder_id']).count().reset_index()[['bidder_id','ip']]
df = pd.merge(df_ip_bidder,df, on='bidder_id')

### F5.Number of Unique Devices used

df_dev_bidder = dfBids[['bidder_id','device','bid_id']].groupby(['bidder_id','device']).count().reset_index()
df_dev_bidder = df_dev_bidder.groupby(['bidder_id']).count().reset_index()[['bidder_id','device']]
df = pd.merge(df_dev_bidder,df, on='bidder_id')

### F6.Bot Score for Country

country_bot = pd.merge(dfBids[['bidder_id','country']],dfTrain[['bidder_id','outcome']],on='bidder_id')
country_bot0 = country_bot[country_bot['outcome']==0]
country_bot1 = country_bot[country_bot['outcome']==1] 

country_bot0 = country_bot0.groupby(['country']).count().rename(columns={'bidder_id':'b0','outcome':'o0'})
country_bot1 = country_bot1.groupby(['country']).count().rename(columns={'bidder_id':'b1','outcome':'o1'})

country_bot = pd.concat([country_bot0,country_bot1],axis=1).reset_index().fillna(0)
country_bot['percentage'] = country_bot['b1']/(country_bot['b0']+country_bot['b1'])
country_bot = country_bot[['index','percentage']].rename(columns={'index':'country'})

country_score = pd.merge(dfBids[['bidder_id','country']],country_bot,on='country') 
country_score = country_score.groupby('bidder_id').sum().reset_index()
df= pd.merge(country_score,df,on='bidder_id').rename(columns={'percentage':'country_score','device':'device_num','ip':'ip_num','country':'country_num'})

### F7.Number of Unique Merchandises

df_merch_bidder = dfBids[['bidder_id','merchandise','bid_id']].groupby(['bidder_id','merchandise']).count().reset_index()
df_merch_bidder = df_merch_bidder.groupby('bidder_id').count().reset_index()[['bidder_id','merchandise']]
df = pd.merge(df_merch_bidder,df, on='bidder_id').rename(columns={'merchandise':'merchandise_num'})

### F8.Bot Score for Merchandise

merch_bot = pd.merge(dfBids[['bidder_id','merchandise']],dfTrain[['bidder_id','outcome']],on='bidder_id')
merch_bot0 = merch_bot[merch_bot['outcome']==0]
merch_bot1 = merch_bot[merch_bot['outcome']==1] 

merch_bot0 = merch_bot0.groupby(['merchandise']).count().rename(columns={'bidder_id':'b0','outcome':'o0'})
merch_bot1 = merch_bot1.groupby(['merchandise']).count().rename(columns={'bidder_id':'b1','outcome':'o1'})

merch_bot = pd.concat([merch_bot0,merch_bot1],axis=1).reset_index().fillna(0)
merch_bot['percentage'] = merch_bot['b1']/(merch_bot['b0']+merch_bot['b1'])
merch_bot = merch_bot[['index','percentage']].rename(columns={'index':'merchandise'})

merch_score = pd.merge(dfBids[['bidder_id','merchandise']],merch_bot,on='merchandise') 
merch_score = merch_score.groupby('bidder_id').sum().reset_index()
df = pd.merge(merch_score,df,on='bidder_id').rename(columns={'percentage':'merchandise_score'})

### F9. Number of Unique URLs 

df_url_bidder = dfBids[['bidder_id','url','bid_id']].groupby(['bidder_id','url']).count().reset_index()
df_url_bidder = df_url_bidder.groupby('bidder_id').count().reset_index()[['bidder_id','url']]
df = pd.merge(df_url_bidder,df, on='bidder_id').rename(columns={'url':'url_num'})

### F10. Bot Score for URL

url_bot = pd.merge(dfBids[['bidder_id','url']],dfTrain[['bidder_id','outcome']],on='bidder_id')
url_bot0 = url_bot[url_bot['outcome']==0]
url_bot1 = url_bot[url_bot['outcome']==1] 

url_bot0 = url_bot0.groupby(['url']).count().rename(columns={'bidder_id':'b0','outcome':'o0'})
url_bot1 = url_bot1.groupby(['url']).count().rename(columns={'bidder_id':'b1','outcome':'o1'})

url_bot = pd.concat([url_bot0,url_bot1],axis=1).reset_index().fillna(0)
url_bot['percentage'] = url_bot['b1']/(url_bot['b0']+url_bot['b1'])
url_bot = url_bot[['index','percentage']].rename(columns={'index':'url'})

url_score = pd.merge(dfBids[['bidder_id','url']],url_bot,on='url') 
url_score = url_score.groupby('bidder_id').sum().reset_index()
df = pd.merge(url_score,df,on='bidder_id').rename(columns={'percentage':'url_score'})

### F11.Bot Score for Device  
dev_bot = pd.merge(dfBids[['bidder_id','device']],dfTrain[['bidder_id','outcome']],on='bidder_id')
dev_bot0 = dev_bot[dev_bot['outcome']==0]
dev_bot1 = dev_bot[dev_bot['outcome']==1] 

dev_bot0 = dev_bot0.groupby(['device']).count().rename(columns={'bidder_id':'b0','outcome':'o0'})
dev_bot1 = dev_bot1.groupby(['device']).count().rename(columns={'bidder_id':'b1','outcome':'o1'})

dev_bot = pd.concat([dev_bot0,dev_bot1],axis=1).reset_index().fillna(0)
dev_bot['percentage'] = dev_bot['b1']/(dev_bot['b0']+dev_bot['b1'])
dev_bot = dev_bot[['index','percentage']].rename(columns={'index':'device'})

dev_score = pd.merge(dfBids[['bidder_id','device']],dev_bot,on='device') 
dev_score = dev_score.groupby('bidder_id').sum().reset_index()
df = pd.merge(dev_score,df,on='bidder_id').rename(columns={'percentage':'device_score'})

### Time Related Features

df_time_diff=  dfBids.copy()[['bidder_id','time']]
df_time_diff['diffs'] = dfBids[['bidder_id','time']].groupby('bidder_id')['time'].diff()
df_time_diff
### F12.Average Time Interval Between Two Continuous Bids

df_time_diff_avg = df_time_diff.groupby(['bidder_id']).mean().reset_index()
df_time_diff_avg = df_time_diff_avg.fillna(1E30).reset_index()[['bidder_id','diffs']].rename(columns={'diffs':'avg_time_interval'})
# Fill large number for those bidder with only 1 bid.
df = pd.merge(df_time_diff_avg, df, on = 'bidder_id')

### F13.Min Time Interval Between Two Continuous Bids

df_time_diff_min = df_time_diff.groupby(['bidder_id']).min().reset_index()
df_time_diff_min = df_time_diff_min.fillna(1E30).reset_index()[['bidder_id','diffs']].rename(columns={'diffs':'min_time_interval'})
# Fill large number for those bidder with only 1 bid.
df = pd.merge(df_time_diff_min, df, on = 'bidder_id')

### F14.Median Time Interval Between Two Continuous Bids

df_time_diff_med = df_time_diff.groupby(['bidder_id']).median().reset_index()
df_time_diff_med = df_time_diff_med.fillna(1E30).reset_index()[['bidder_id','diffs']].rename(columns={'diffs':'med_time_interval'})
# Fill large number for those bidder with only 1 bid.
df = pd.merge(df_time_diff_med, df, on = 'bidder_id')

### F15.Average of Median Time Interval Between Two Continuous Bids in Single Auctions 

df_time_diff_auction=  dfBids.copy()[['bidder_id','time','auction']]
df_time_diff_auction['diffs'] = dfBids[['bidder_id','time','auction']].groupby(['bidder_id','auction'])['time'].diff()

df_time_diff_med_auction = df_time_diff_auction.groupby(['bidder_id','auction']).median().reset_index()
df_time_diff_med_auction = df_time_diff_med_auction.groupby('bidder_id').mean().reset_index()
df_time_diff_med_auction = df_time_diff_med_auction.fillna(1E30).reset_index()[['bidder_id','diffs']].rename(columns={'diffs':'avg_med_time_interval'})
df = pd.merge(df_time_diff_med_auction, df, on = 'bidder_id')

df.to_csv('features.csv')
