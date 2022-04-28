import socket
import time
# import requests
import re
import threading

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
from werkzeug.serving import make_server

from line_bot.message import *
from line_bot.new import *
from line_bot.Function import *
from line_bot.law_scraper import get_law_detail


class App_Thread(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)

        app = Flask(__name__)

        self.server = make_server(config['web_server_IP'], config['web_server_port'], app)
    
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while True:
            try:
                client_socket.connect((config['server_socket_IP'], config['server_socker_port']))

                break
            except:
                time.sleep(3)

        # Channel Access Token
        line_bot_api = LineBotApi(config['LINE_CHANNEL_ACCESS_TOKEN'])

        # Channel Secret
        handler = WebhookHandler(config['CHANNEL_SECRET'])

        # Add Rich menu
        authorization_token = 'Bearer ' + config['LINE_CHANNEL_ACCESS_TOKEN']
        headers = {'Authorization': authorization_token, 'Content-Type': 'application/json'}

        # req = requests.request('POST', 'https://api.line.me/v2/bot/user/all/richmenu/'+ config['rich_menu_ID'], headers=headers)
        # print(req.text)

        rich_menu_list = line_bot_api.get_rich_menu_list()

        # listen all POST request from /callback
        @app.route("/callback", methods=['POST'])
        def callback():
            # get X-Line-Signature header value
            signature = request.headers['X-Line-Signature']

            # get request body as text
            body = request.get_data(as_text=True)
            app.logger.info('Request body: ' + body)

            # handle webhook body
            try:
                handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400)

            return 'OK'


        # process message
        @handler.add(MessageEvent, message=TextMessage)
        def handle_message(event):
            msg = event.message.text

            if re.match('(\D*)第(\d+)條(之(\d)+)?', msg):
                law_detail = get_law_detail(msg)
                line_bot_api.reply_message(event.reply_token,
                    TextSendMessage(text=law_detail))
            else:
                client_socket.sendall(msg.encode())

                if msg == 'shutdown':
                    client_socket.close()
                else:
                    serverMessage = str(client_socket.recv(1024), encoding='utf-8')

                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=serverMessage))


        @handler.add(PostbackEvent)
        def handle_message(event):
            print(event.postback.data)


        @handler.add(MemberJoinedEvent)
        def welcome(event):
            uid = event.joined.members[0].user_id
            gid = event.source.group_id
            profile = line_bot_api.get_group_member_profile(gid, uid)
            name = profile.display_name
            message = TextSendMessage(text=f'{name}歡迎加入')
            line_bot_api.reply_message(event.reply_token, message)


    def run(self):
        self.server.serve_forever()

    
    def shutdown(self):
        self.server.shutdown()