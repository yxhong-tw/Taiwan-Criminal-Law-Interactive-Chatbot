import os


# Get file list strings
def get_file_list_string(dataset_folder_path):
    string_list = []
    number_of_files = 0

    for file_name in os.listdir(dataset_folder_path):
        if file_name == 'README.md':
            continue

        string_list.append(f'\t- {file_name}\n')
        number_of_files += 1

    return string_list, number_of_files


# Traversal all node in this data
def traversal_all_nodes(nodes_list_strings, data, tab_num):
    if type(data) == dict:
        for item in data:
            nodes_list_strings.append(('\t' * tab_num + '- ' + item + '\n'))
            nodes_list_strings = traversal_all_nodes(nodes_list_strings, data[item], tab_num+1)

    return nodes_list_strings