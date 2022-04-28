# coding=utf-8
from bs4 import BeautifulSoup
import requests
import re

def get_law_detail(article_of_fact):
	laws = ['刑法', '毒品危害防制條例', '商標法', '槍砲彈藥刀械管制條例', '藥事法', '家庭暴力防治法',
			'性騷擾防治法', '廢棄物清理法', '妨害兵役治罪條例', '性侵害犯罪防治法', '公職人員選舉罷免法',
			'入出國及移民法', '建築法', '著作權法', '區域計畫法', '森林法', '醫療法', '水土保持法',
			'醫師法', '個人資料保護法']
	pcodes = ['C0000001', 'C0000008', 'J0070001', 'D0080047', 'L0030001', 'D0050071',
			'D0050074', 'O0050001', 'F0120002', 'D0080079', 'D0020010',
			'D0080132', 'D0070109', 'J0070017', 'D0070030', 'M0040001', 'L0020021', 'M0110001'
			'L0020001', 'I0050021']

	# Convert fact to url
	re_string = '^([\u4e00-\u9fa5]*)第(\d+)條(之(\d)+)?$'
	match = re.findall(re_string, article_of_fact)[0]
	try:
		pcode = pcodes[laws.index(match[0])]
	except:
		return '不存在此法條！請重新查詢'

	fino = match[1] + '-' + match[3] if match[3] != '' else match[1]
	url = f'https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={pcode}&flno={fino}'

	# get html page and parse to soup
	response = requests.get(url)
	soup = BeautifulSoup(response.text, "html.parser")

	# get law-article content
	law_detail = ''
	content = soup.find('div', class_='law-article')
	for idx, a in enumerate(content):
		if idx == 0: continue
		law_detail += f'{idx}. {a.text}'
		law_detail += '\n' if idx != len(content) - 1 else ''
	# print(law_detail)
	return law_detail

if __name__ == '__main__':
	get_law_detail('刑法第339條之1')