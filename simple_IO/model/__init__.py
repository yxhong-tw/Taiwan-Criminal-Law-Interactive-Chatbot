from simple_IO.model.ljp.Bert import LJPBert


model_list = {
    "LJPBert": LJPBert
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
