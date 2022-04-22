#!/usr/bin/env python3

from line_bot.app import run_app
run_app()
import socket


# # Create socket channel to model script
# model_HOST, model_PORT = '172.17.0.3', 8000
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((model_HOST, model_PORT))

# msg = '小名偷情我老婆'
# # try:
# client.sendall(msg.encode())
# # except:
# # 	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 	client.connect((model_HOST, model_PORT))
# # 	client.sendall(msg.encode())

# serverMessage = str(client.recv(1024), encoding='utf-8')
# print(serverMessage)