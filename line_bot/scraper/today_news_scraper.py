# Google search news '刑法'
# coding=utf-8
from bs4 import BeautifulSoup
import requests


def get_today_news(news_count):
    url = 'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNREZ1TlhjU0JYcG9MVlJYS0FBUAE?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant'
    # get html page and parse to soup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get news url
    content = soup.find_all('a', class_='DY5T1d RZIKme')

    # get 'news_count' news urls, titles
    urls, titles= [], []
    for i in range(news_count):
        url = 'https://news.google.com' + content[i].get('href')[1:]
        urls.append(url)
        titles.append(content[i].text)
    # print(urls)
    # print(titles)
    return urls, titles

if __name__ == '__main__':
    get_today_news(3)