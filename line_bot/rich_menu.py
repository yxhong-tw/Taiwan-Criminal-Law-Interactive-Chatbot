import requests
import json

LINE_CHANNEL_ACCESS_TOKEN = 'UmN0XsFJwcHP8lL7cPrvu30LXwqhMfni5+cTHdjrHKfOGW3DdgNh04ZmflN74CzwyhcRqiDCSzbnGDXhKxRwzXeYmO/1ELsnZFZnKJneME5cWq+hmbUjCongPvcsaSVOI1Ml6KfKoHybjIGM67pFXQdB04t89/1O/w1cDnyilFU='

token = LINE_CHANNEL_ACCESS_TOKEN

Authorization_token = "Bearer " + LINE_CHANNEL_ACCESS_TOKEN

import requests
import json

# ====================================================
'''
這區不能刪掉
'''

token = 'UmN0XsFJwcHP8lL7cPrvu30LXwqhMfni5+cTHdjrHKfOGW3DdgNh04ZmflN74CzwyhcRqiDCSzbnGDXhKxRwzXeYmO/1ELsnZFZnKJneME5cWq+hmbUjCongPvcsaSVOI1Ml6KfKoHybjIGM67pFXQdB04t89/1O/w1cDnyilFU='

headers = {"Authorization":"Bearer UmN0XsFJwcHP8lL7cPrvu30LXwqhMfni5+cTHdjrHKfOGW3DdgNh04ZmflN74CzwyhcRqiDCSzbnGDXhKxRwzXeYmO/1ELsnZFZnKJneME5cWq+hmbUjCongPvcsaSVOI1Ml6KfKoHybjIGM67pFXQdB04t89/1O/w1cDnyilFU=" , "Content-Type":"application/json"}

# ====================================================
# '''
# Step 1 : 設定一次就可以註解掉了
# '''

# body = {
#     "size": {"width": 2500, "height": 1686},
#     "selected": "false",
#     "name": "Menu",
#     "chatBarText": "更多資訊",
#     "areas":[
#         {
#           "bounds": {"x": 113, "y": 45, "width": 1036, "height": 762},
#           "action": {"type": "message", "text": "身體資訊"}
#         },
#         {
#           "bounds": {"x": 1321, "y": 45, "width": 1036, "height": 762},
#           "action": {"type": "message", "text": "營養素"}
#         },
#         {
#           "bounds": {"x": 113, "y": 910, "width": 1036, "height": 762},
#           "action": {"type": "message", "text": "吃"}
#         },
#         {
#           "bounds": {"x": 1321, "y": 910, "width": 1036, "height": 762},
#           "action": {"type": "message", "text": "運動gogo"}
#         }
#     ]
#   }

# req = requests.request('POST', 'https://api.line.me/v2/bot/richmenu',
#                        headers=headers,data=json.dumps(body).encode('utf-8'))

# print(req.text)
# 在這裡要記起 rich_menu_id


# ====================================================
'''
Step 2 : import 要的東西
這段也不能刪掉
'''
from linebot import (
    LineBotApi, WebhookHandler
)

line_bot_api = LineBotApi(token)
rich_menu_id = 'richmenu-1fd5ca3d00e7c206b31818f7baf36101'

# ====================================================

# """
# 設定照片，只能執行過一次
# """
# path = 'line_bot/image/RichMenu_DesignTemplate/Large/Large/richmenu-template-guide-02.png'

# with open(path, 'rb') as f:
#     line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", f)
    
# ====================================================


req = requests.request('POST', 'https://api.line.me/v2/bot/user/all/richmenu/'+rich_menu_id,
                       headers=headers)
print(req.text)

rich_menu_list = line_bot_api.get_rich_menu_list()


# ====================================================
"""
上面要重新設定的話要把 ID 刪掉重來
"""

# line_bot_api.delete_rich_menu(rich_menu_id)