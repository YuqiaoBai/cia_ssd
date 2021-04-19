
def main():
    # data loader convnet
    data_loader_convnet = None
    data_masking = None # write your own class/func

    # construct model
    cnn_model = None
    od_model = None
    # resume/ reload parems
    od_model.load(path)
    # optimizer
    opt = None

    # forward cnn
    for data in data_loader_convnet:
        data_conv_in = data["conv_in"] # 256x256x22
        data_clouds = data["clouds"] # 256x256x22
        out = cnn_model(data) # 256x256x20
        input_od = data_masking(data_clouds, out["mask"])
        pred = od_model(input_od)


