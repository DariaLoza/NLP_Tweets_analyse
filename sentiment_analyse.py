from textblob import TextBlob

file1 = open("preparing_tweets_boomers.txt", "r", encoding='utf-8')

while True:
    line = file1.readline()
    analysis = TextBlob(line).sentiment

    if not line:
        break
    print(line.strip())
    print(analysis)


file1.close
