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