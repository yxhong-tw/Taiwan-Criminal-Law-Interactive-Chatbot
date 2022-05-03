# Google search news '刑法'
# coding=utf-8
from bs4 import BeautifulSoup
import requests
import json


def get_today_news(news_count):
    url = 'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNREZ1TlhjU0JYcG9MVlJYS0FBUAE?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant'
    # get html page and parse to soup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get news url
    content = soup.find_all('a', class_='DY5T1d RZIKme')

    post_url = 'https://api.reurl.cc/shorten'
    headers = {'Content-Type': 'application/json', 'reurl-api-key': '4070ff49d794e03c14553b663c974755ecd0b231949c04df8a38b58d65165567c4f5d6'}

    # get 'news_count' news urls, titles
    urls, titles= [], []
    for i in range(news_count):
        temp_url = 'https://news.google.com' + content[i].get('href')[1:]

        data = json.dumps({'url' : temp_url, 'utm_source' : 'FB_AD'})
        reponse = requests.post(post_url, headers=headers, data=data)

        short_url = json.loads(reponse.text)['short_url']

        urls.append(short_url)
        titles.append(content[i].text)

    # print(urls)
    # print(titles)
    return urls, titles

if __name__ == '__main__':
    get_today_news(3)