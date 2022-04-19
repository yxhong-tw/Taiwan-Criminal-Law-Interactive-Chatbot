import simple_IO.formatter as form

formatter = {}
collate_fn = {}

def init_formatter(config, task_list, *args, **params):
    for task in task_list:
        formatter[task] = form.init_formatter(config, task, *args, **params)

        def serve_collate_fn(data):
            return formatter['serve'].process(data, config, 'serve')

        if task == 'serve':
            collate_fn[task] = serve_collate_fn
        else:
            print('task not found')
            raise NotImplementedError