import snscrape.modules.twitter as sntwitter
import pandas as pd

#query="python"
query="(from:raseluddin102) until:2022-12-24 since:2006-01-01"
tweets=[]
limits=5000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    #print(vars(tweet))
    #break
    if len(tweets)==limits:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

df=pd.DataFrame(tweets, columns=['Date','user','Tweet'])
print(df)