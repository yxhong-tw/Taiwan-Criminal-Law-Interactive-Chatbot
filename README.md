# Taiwan-Criminal-Law-Interactive-Chatbot

## Before running
### Download model and checkpoint
Due to the size of model and checkpoint files are too large, you need to download these two files and put them into the correct folder, and the program can execute correctly.
- [pytorch_model.bin](https://drive.google.com/file/d/1jkSh7_UOzY637J1VMWC8uGoWCBf_uoVK/view?usp=sharing): `./simple_IO/bert/`
- [checkpoint_9.pkl](https://drive.google.com/file/d/1WgM6t02EvVF98F8Z1eyiBGvdwqC3Mmtr/view?usp=sharing): `./simple_IO/model/ljp/LJPBertExercise/`

### Install the modules
Run `pip install -r requirements.txt` to install all needed modules.

## How to use simple_IO
`simple_IO` is designed to use on WIDM Lab 110 undergraduate student competition.

The project architecture is like:
- Taiwan-Criminal-Law-Interactive-Chatbot
    - line-bot
    - simple_IO
    - other function folders

You can refer to `simple_IO_example.py` which is in `./line_bot/` to see how to import and use `simple_IO`.

If you want to make the message received from Line-bot can be used as 'fact' and return the outputs (accuse, article_source, article), you can see `./simple_IO/tools/serve_tool.py` file.

## How to run line-bot  
1. move to root directory ```Taiwan-Criminal-Law-Interactive-Chatbot/```  
2. run ngrok port 5000  
```ngrok http 5000``` 
3. open another screen and run ```run.py``` file  
```python3 run.py```
