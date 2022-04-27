import logging
import json

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(ch)

def process(mode):
    data_list = []

    with open(f'./data/{mode}_50.json', 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))

        file.close()

    for index in range(len(data_list)):
        fact = delete_blanks(data_list[index]["fact"])
        fact = delete_parentheses(fact)
        fact_paragraph_list = separate_paragraphs(fact)
        result = delete_odd_chars(fact_paragraph_list)

        data_list[index]["fact"] = result

    with open(f'./data/{mode}_50_processed.json', 'w') as file:
        for data in data_list:
            file.write(json.dumps(data, ensure_ascii=False).encode('utf-8').decode() + '\n')

        file.close()


def delete_blanks(string):
    result = ''

    for char in string:
        if char != ' ':
            result += char

    return result


def delete_parentheses(string):
    result = ''
    in_parentheses = False
            
    for char in string:
        if in_parentheses == False:
            if char == '（':
                in_parentheses = True
            else:
                result += char
        else:
            if char == '）':
                in_parentheses = False

    return result


def separate_paragraphs(string):
    current_index = 0
    chinese_number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    paragraph_list = []

    for index in range(len(string)):
        if index + 1 < len(string):
            if string[index] == '。' and string[index + 1] in chinese_number:
                for temp_index in range(index + 1, len(string)):
                    if string[temp_index] == '、':
                        paragraph_list.append(string[current_index:index + 1])
                        current_index = index + 1
                    if string[temp_index] not in chinese_number:
                        break
        else:
            paragraph_list.append(string[current_index:])

    paragraph_list = paragraph_list[:-1]

    for index in range(len(paragraph_list)):
        for temp_index in range(len(paragraph_list[index])):
            if paragraph_list[index][temp_index] == '、':
                paragraph_list[index] = paragraph_list[index][temp_index + 1:]
                break

    return paragraph_list


def delete_odd_chars(paragraph_list):
    result = ''

    for paragraph in paragraph_list:
        index = 0

        while index < len(paragraph):
            if paragraph[index] == '○':
                if paragraph[index + 1] != '○':
                    index += 2
                else:   # paragraph[index + 1] == '○'
                    index += 1
            elif paragraph[index].isnumeric():
                if paragraph[index] != '0':
                    result += paragraph[index]

                    if paragraph[index + 1] == '.': # not integer
                        result += paragraph[index + 1]  # .

                        for temp_index in range(index + 2, len(paragraph)):
                            if not paragraph[temp_index].isnumeric():
                                index = temp_index
                                break

                            result += paragraph[temp_index]
                    else:   # integer
                        for temp_index in range(index + 1, len(paragraph)):
                            if not paragraph[temp_index].isnumeric():
                                index = temp_index
                                break

                            result += paragraph[temp_index]
                else:   # paragraph[index] == '0'
                    if paragraph[index + 1] == '.': # not integer
                        result += paragraph[index]  # 0
                        result += paragraph[index + 1]  # .

                        for temp_index in range(index + 2, len(paragraph)):
                            if not paragraph[temp_index].isnumeric():
                                index = temp_index
                                break

                            result += paragraph[temp_index]
                    else:
                        if paragraph[index + 1] != '0':
                            index += 2
                        else:   # paragraph[index + 1] == '0'
                            index += 1
            else:
                result += paragraph[index]
                index += 1

    return result


def check_odd_chars(string):
    before_odd_char_list = []

    index = 0

    while index < len(string):
        if string[index] == '○':
            if string[index + 1] != '○' and string[index] not in before_odd_char_list:
                before_odd_char_list.append(string[index + 1])
        elif string[index].isnumeric() and string[index] != '0':
            for temp_index in range(index + 1, len(string) - 1):
                if not (string[temp_index].isnumeric() or string[temp_index] == '.'):
                    index = temp_index
                    break
        elif string[index] == '0' and string[index + 1] != '.' and string[index + 1] not in before_odd_char_list:
            before_odd_char_list.append(string[index + 1])

        index += 1

    print(before_odd_char_list)


if __name__ == '__main__':
    logger.info('Begin to process train data.')
    process('train')
    logger.info('train data processing is complete.')

    logger.info('Begin to process valid data.')
    process('valid')
    logger.info('valid data processing is complete.')

    logger.info('Begin to process test data.')
    process('test')
    logger.info('test data processing is complete.')

    logger.info('Task completed.')