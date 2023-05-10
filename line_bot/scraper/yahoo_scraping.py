# Yahoo news
# coding=UTF-8

# 1. pick raw text excluding link and figure
# 2. exclude first <p> with ".*報導$"
# 3. remove "【《〈()""

from bs4 import BeautifulSoup
import requests
import re


def get_news_content(url):
    # get html page and parse to soup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # get news content
    content = soup.find('div', class_='caas-body')
    with open('page2.html', 'w') as f:
        f.write(content.prettify())

    print(content.text)

    # get all <p>
    # all_paragraph = content.select("p[")
    # law_detail = ''

    # for idx, a in enumerate(content):
    # 	if idx == 0: continue
    # 	law_detail += f'{idx}. {a.text}'
    # 	law_detail += '\n' if idx != len(content) - 1 else ''

    # return law_detail

if __name__ == '__main__':
    get_news_content('https://tw.news.yahoo.com/%E9%87%8D%E6%A9%9F%E8%BB%8A%E7%89%8C-%E6%9C%89%E9%BB%9E%E7%BF%B9-%E9%A9%97%E8%BB%8A%E9%81%8E%E9%97%9C%E8%AD%A6%E5%8D%BB%E5%8F%96%E7%B7%A0%E9%96%8B%E7%BD%B0-102700092.html')