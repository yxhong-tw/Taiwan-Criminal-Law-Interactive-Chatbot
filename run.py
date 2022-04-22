from line_bot.app import run_app
from simple_IO.serve_function import serving
from multiprocessing import Process, Pool
import time

if __name__ == '__main__':
    model_process = Process(target=serving, args=())
    line_bot_process = Process(target=run_app, args=())
    model_process.start()
    time.sleep(5)
    line_bot_process.start()
    model_process.join()
    line_bot_process.join()
    
# serving()