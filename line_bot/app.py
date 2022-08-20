import socket
import time
import re
import threading

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
from werkzeug.serving import make_server

from line_bot.scraper.law_scraper import get_law_detail
from line_bot.scraper.today_news_scraper import get_today_news


class App_Thread(threading.Thread):
    def __init__(self, parameters):
        threading.Thread.__init__(self)

        app = Flask(__name__)

        self.server = make_server(
            host=parameters['web_server_IP']
            , port=parameters['web_server_port']
            , app=app)
    
        client_socket = socket.socket(
            family=socket.AF_INET
            , type=socket.SOCK_STREAM)
        client_socket.settimeout(10)

        while True:
            try:
                client_socket.connect(
                    (parameters['server_socket_IP']
                    , parameters['server_socket_port'])
                )

                break
            except:
                time.sleep(secs=3)

        line_bot_api = LineBotApi(
            channel_access_token=parameters['LINE_CHANNEL_ACCESS_TOKEN'])
        handler = WebhookHandler(channel_secret=parameters['CHANNEL_SECRET'])

        # Listen all POST request from /callback.
        @app.route("/callback", methods=['POST'])
        def callback():
            # Get X-Line-Signature header value.
            signature = request.headers['X-Line-Signature']

            # Get request body as text.
            body = request.get_data(as_text=True)
            app.logger.info('Request body: ' + body)

            # Handle webhook body
            try:
                handler.handle(body=body, signature=signature)
            except InvalidSignatureError:
                abort(400)

            return 'OK'


        # Process receiving message.
        @handler.add(event=MessageEvent, message=TextMessage)
        def handle_message(event):
            msg = event.message.text

            # Search the detail of article.
            if re.match('^([\u4e00-\u9fa5]*)第(\d+)條(之(\d)+)?$', msg):
                law_detail = get_law_detail(msg)
                line_bot_api.reply_message(
                    reply_token=event.reply_token
                    , messages=TextSendMessage(text=law_detail)
                )
            # Send news of law on Google News.
            elif re.match('\s*今日新聞\s*', msg):
                urls, titles = get_today_news(news_count=3)
                today_news_message = ('熱騰騰的新聞來囉！' + '\n')

                for idx, (url, title) in enumerate(zip(urls, titles)):
                    today_news_message += f'．{title} ({url})'

                    if idx != (len(url) - 1):
                        today_news_message += '\n'
                
                line_bot_api.reply_message(
                    reply_token=event.reply_token
                    , messages=TextSendMessage(text=today_news_message)
                )
            # Show how to use message.
            elif re.match('\s*如何使用\s*', msg):
                usage_message = (
                    '歡迎使用臺灣刑法互動聊天機器人！' + '\n' + \
'．輸入「今日新聞」可以查看今日和法律相關的新聞' + '\n' + \
'．輸入法條（例如「刑法第185條之3」）可以得到法條的詳細內容' + '\n' + \
'．輸入一個事件（例如「小翔喝酒後開車。」）可以得到事件可能被起訴的罪名'
                )

                line_bot_api.reply_message(
                    reply_token=event.reply_token
                    , messages=TextSendMessage(text=usage_message)
                )
            else:
                if len(msg) < 8:
                    line_bot_api.reply_message(
                        reply_token=event.reply_token
                        , messages=TextSendMessage(
                            text='長度不足，請輸入更長的敘述！')
                    )
                else:
                    client_socket.sendall(msg.encode())

                    if msg == 'shutdown':
                        client_socket.close()
                    else:
                        try:
                            serverMessage = str(
                                client_socket.recv(1024)
                                , encoding='UTF-8')
                        except socket.timeout:
                            serverMessage = '請求資料超時，請再試一次！'

                        msg = (f'情境「{msg}」\n' + serverMessage)

                        line_bot_api.reply_message(
                            reply_token=event.reply_token
                            , messages=TextSendMessage(text=msg)
                        )


        @handler.add(PostbackEvent)
        def handle_message(event):
            print(event.postback.data)


        @handler.add(MemberJoinedEvent)
        def welcome(event):
            gid = event.source.group_id
            uid = event.joined.members[0].user_id
            profile = line_bot_api.get_group_member_profile(
                group_id=gid
                , user_id=uid)
            name = profile.display_name

            message = TextSendMessage(
                text=(f'{name}，歡迎使用臺灣刑法互動聊天機器人！' + '\n' + \
'．輸入「今日新聞」可以查看今日和法律相關的新聞' + '\n' + \
'．輸入法條（例如「刑法第185條之3」）可以得到法條的詳細內容' + '\n' + \
'．輸入一個事件（例如「小翔喝酒後開車。」）可以得到事件可能被起訴的罪名'))

            line_bot_api.reply_message(
                reply_token=event.reply_token
                , messages=message
            )


    def run(self):
        self.server.serve_forever()


    def shutdown(self):
        self.server.shutdown()