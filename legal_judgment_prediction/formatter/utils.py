def set_special_tokens(add_tokens_at_beginning, max_len, data):
    if add_tokens_at_beginning == True:
        data.insert(0, '[CLS]')
        data.append('[SEP]')

    if len(data) > max_len:
        data = data[0:max_len-1]
        data.append('[SEP]')
    else:
        while len(data) < max_len:
            data.append('[PAD]')

    return data