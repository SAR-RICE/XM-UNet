import sys
import time

import cv2
import numpy as np
from PIL import Image

import torch
from models.XM_UNet import XM_UNet
from configs.config_setting import setting_config

from osgeo import gdal
import torch.nn.functional as F


def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "cannot open")
    return dataset



def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype, options=["COMPRESS=LZW"])
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


if __name__ == "__main__":
    model_cfg = setting_config.model_config
    model = XM_UNet(num_classes=model_cfg['num_classes'],
                            input_channels=model_cfg['input_channels'],
                            c_list=model_cfg['c_list'],
                            split_att=model_cfg['split_att'],
                            bridge=model_cfg['bridge'], )

    model = torch.nn.DataParallel(model.cuda(), device_ids=[0], output_device=[0])
    
    resume_model = ''
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    model.eval() 

    mode = "dir_predict"

    count = False
    name_classes = ["background", "rice"]

    # -------------------------------------------------------------------------#
    #   dir_origin_path      Specifies the folder path of the images to be used for prediction
    #   dir_save_path        Specifies the path to save the predicted images
    # -------------------------------------------------------------------------#
    dir_origin_path = ''
    dir_save_path = ''


    if mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.tif', '.tiff', '.hdr')):
                image_path = os.path.join(dir_origin_path, img_name)#[:-4])
                dataset = readTif(image_path)
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                band = dataset.RasterCount
                proj = dataset.GetProjection()
                geotrans = dataset.GetGeoTransform()
                gdal_array = dataset.ReadAsArray(0, 0, width, height)
                gdal_array = np.nan_to_num(gdal_array, posinf=0, neginf=0)
                image = np.rollaxis(gdal_array, 0, 3)
                result_img = np.full_like(image, 0)[:, :, 0]
                temp_img = result_img[:, :]
                temp_list = []
                temp_list.append(temp_img)
                result_img = np.array(temp_list)
                result_img = np.rollaxis(result_img, 0, 3)

                CropSize = 256
                RepetitionRate = 0.5  # 0.5
                for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                        sub_img = image[int(i * CropSize * (1 - RepetitionRate)): int(
                            i * CropSize * (1 - RepetitionRate)) + CropSize,
                                  int(j * CropSize * (1 - RepetitionRate)): int(
                                      j * CropSize * (1 - RepetitionRate)) + CropSize, :]
                        sub_img = torch.from_numpy(np.rollaxis(sub_img, 2, 0)).float().unsqueeze(0)
                        out = model(sub_img)
                        if type(out) is tuple:
                            out = out[0]
                        out = out.squeeze(1).cpu().detach().numpy()
                        sub_r_image = np.where(np.squeeze(out, axis=0) > 0.5, 1, 0) 
                        result_img[
                        int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                        int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize,
                        :] = sub_r_image.reshape(CropSize, CropSize, 1)#np.transpose(sub_r_image, (1, 2, 0))

                for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                    sub_img = image[int(i * CropSize * (1 - RepetitionRate)): int(
                        i * CropSize * (1 - RepetitionRate)) + CropSize, (width - CropSize): width, :]
                    sub_img = torch.from_numpy(np.rollaxis(sub_img, 2, 0)).float().unsqueeze(0)
                    out = model(sub_img)
                    if type(out) is tuple:
                        out = out[0]
                    out = out.squeeze(1).cpu().detach().numpy()
                    sub_r_image = np.where(np.squeeze(out, axis=0) > 0.5, 1, 0) 
                    result_img[int(i * CropSize * (1 - RepetitionRate)): int(
                        i * CropSize * (1 - RepetitionRate)) + CropSize, (width - CropSize): width, :] = sub_r_image.reshape(CropSize, CropSize, 1)#np.transpose(sub_r_image, (1, 2, 0))

                for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                    sub_img = image[(height - CropSize): height, int(j * CropSize * (1 - RepetitionRate)): int(
                        j * CropSize * (1 - RepetitionRate)) + CropSize, :]
                    sub_img = torch.from_numpy(np.rollaxis(sub_img, 2, 0)).float().unsqueeze(0)
                    out = model(sub_img)
                    if type(out) is tuple:
                        out = out[0]
                    out = out.squeeze(1).cpu().detach().numpy()
                    sub_r_image = np.where(np.squeeze(out, axis=0) > 0.5, 1, 0) 
                    result_img[(height - CropSize): height, int(j * CropSize * (1 - RepetitionRate)): int(
                        j * CropSize * (1 - RepetitionRate)) + CropSize, :] = sub_r_image.reshape(CropSize, CropSize, 1)#np.transpose(sub_r_image, (1, 2, 0))

                # 最后一块
                sub_img = image[(height - CropSize):height, (width - CropSize):width, :]
                sub_img = torch.from_numpy(np.rollaxis(sub_img, 2, 0)).float().unsqueeze(0)
                out = model(sub_img)
                if type(out) is tuple:
                    out = out[0]
                out = out.squeeze(1).cpu().detach().numpy()
                sub_r_image = np.where(np.squeeze(out, axis=0) > 0.5, 1, 0) 
                result_img[(height - CropSize):height, (width - CropSize):width, :] = sub_r_image.reshape(CropSize, CropSize, 1)#np.transpose(sub_r_image, (1, 2, 0))

                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                final_img = np.transpose(result_img, [2, 0, 1])
                t = np.max(final_img)
                writeTiff(final_img, geotrans, proj, os.path.join(dir_save_path, img_name[:-3]+'tif'))

    else:
        raise AssertionError("Please specify the correct mode: 'dir_predict'.")
