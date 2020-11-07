import numpy as np
import math, os, torch
import importlib


def checkdirctexist(dirct):
    if not os.path.exists(dirct):
        os.makedirs(dirct)


def PSNR_self(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred -gt)
    rmse = math.sqrt(np.mean(imdff.cpu().data[0].numpy() ** 2))
    if rmse == 0:
        return 100
    return 20.0 * math.log10(1.0/rmse)


def save_checkpoint(model, epoch, save_path):
    model_out_path = os.path.join(save_path, "model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model": model}
    # check path status
    if not os.path.exists("model/"):
        os.makedirs("model/")
    # save model
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def load_checkpoint(path, extra_params=None):
    dict = torch.load(path)
    print(dict["epoch"])
    print(dict["model"])
    # md = importlib.import_module(dict["model_file"])
    # if extra_params:
    #     model = eval("md.{}".format(dict["model_class"]))(**extra_params)
    # else:
    #     model = eval("md.{}".format(dict["model_class"]))()
    model = dict['model']
    return model


