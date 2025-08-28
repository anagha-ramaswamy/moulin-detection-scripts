### Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
import rioxarray as rxr

import os
import gc
import re
import io
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from scipy.ndimage import zoom
from flipnslide.tiling import FlipnSlide

from google.cloud import storage
storage_client = storage.Client()



### Argument Parser

parser = argparse.ArgumentParser(description="Tile the image.")
parser.add_argument('--monochrome_path', required=True, help="Path to the monochrome tif")
parser.add_argument('--multiband_path', required=True, help="Path to the multiband tif")
parser.add_argument('--fracture_path', required=True, help="Path to the fracture mask tif")
parser.add_argument('--moulin_path', required=True, help="Path to the moulin mask tif")
parser.add_argument('--tile_size', required=True, help="Size for subtiles")
parser.add_argument('--save_path', required=True, help="Path to folder for saving the tiles")
args = parser.parse_args()



### Load the Imagery

print('Importing the tif files from the bucket...')

bucket_name = 'yao_scratch'

def bucket2numpy(geotiff_url, print_coords=True):
    '''
    convert the bucket index to a numpy array
    '''
    gcs_url = f'gs://{bucket_name}/{geotiff_url}'
    raster = rxr.open_rasterio(gcs_url, masked=True)
    image = raster.to_numpy()
    
    if print_coords==True:
        print(raster.rio.reproject("EPSG:4326").rio.bounds())
    
    return image

monochrome_image = bucket2numpy(args.monochrome_path, print_coords=False)
bands_image = bucket2numpy(args.multiband_path,print_coords=False)
fracture_mask = bucket2numpy(args.fracture_path,print_coords=False)
moulin_mask = bucket2numpy(args.moulin_path,print_coords=False)

# Set the nans for the moulin mask to match the fracture mask
moulin_mask[(moulin_mask != 1.) & (~np.isnan(fracture_mask))] = 0.

print(f'The image and mask shapes match: {monochrome_image.shape == fracture_mask.shape == moulin_mask.shape}')
gc.collect()



### Reproject the coordinates of the color bands to High Res

print('Reprojecting the multiband image onto the same grid as the monochrome image...')

scale_factor_rows = monochrome_image.shape[1] / bands_image.shape[1]  #scaling factor for latitude (rows)
scale_factor_cols = monochrome_image.shape[2] / bands_image.shape[2]  #scaling factor for longitude (columns)

upsampled_bands = zoom(bands_image, (1, scale_factor_rows, scale_factor_cols), order=1)  # order=1 for bilinear interpolation
del bands_image
gc.collect()



### Ensure that the number of nans are the same in each image

print('Ensuring that all the number of nans are equivalent in each image...')

for ii in range(len(upsampled_bands)):
    upsampled_bands[ii][np.isnan(monochrome_image[0])] = np.nan
    
fracture_mask[0][np.isnan(upsampled_bands[0])] = np.nan
moulin_mask[0][np.isnan(upsampled_bands[0])] = np.nan
monochrome_image[0][np.isnan(upsampled_bands[0])] = np.nan

print(f'The nan count is now equivalent: {np.all(np.equal(np.isnan(upsampled_bands[0]), np.isnan(monochrome_image[0])))}')
gc.collect()



### Ensure that the number of nans are the same in each image

print('Processing images into correct inputs...')

ndwi = (upsampled_bands[1,...]-upsampled_bands[4,...])/(upsampled_bands[1,...]+upsampled_bands[4,...])
blue_minus_green = upsampled_bands[1,...] - upsampled_bands[2,...]
overall_image = np.array([monochrome_image[0], ndwi, blue_minus_green]) #this is the order of the image indices

water_mask = ((ndwi > 0.19) & (blue_minus_green > 0.7)).astype(int).astype(float)
water_mask[np.isnan(fracture_mask[0])] = np.nan
overall_mask = np.array([moulin_mask[0], water_mask, fracture_mask[0]]) #this is the order of the mask indices

print(f'Check that this gives only 0.,1.,nan values: {np.unique(overall_mask)}')

del upsampled_bands
del monochrome_image
del ndwi
del blue_minus_green
del moulin_mask
del water_mask
del fracture_mask

gc.collect()



### Run FlipnSlide on the images

print('Normalizing images and processing into tiles using flipnslide...')

tile_size = int(args.tile_size)

def combo_scaler(x, range_max=1):
    median_x = np.nanmedian(x)
    iqr_x = np.nanpercentile(x,75) - np.nanpercentile(x,25)
    robust_x = ((x-median_x)/iqr_x)
    gc.collect()
    
    return ((robust_x - np.nanmin(robust_x)) / (np.nanmax(robust_x) - np.nanmin(robust_x))) * range_max

def norm_images(channels):
    '''
    Normalizes the images using a combo of robust scalar and min-max scalar
    '''
    
    n = len(channels)
    shape = channels.shape
    normed_data = np.zeros((n, shape[1], shape[2]))
    
    for ii in range(n):
        normed_data[ii,...] = combo_scaler(channels[ii,...])  
    gc.collect()
        
    return normed_data

def preprocess_data(img, mask_flag=False):
    
    #First normalize the image
    if mask_flag==False:
        norm_image = norm_images(img)
        del img
        gc.collect()
        
        #Then run flipnslide on the image
        tiled_image = FlipnSlide(tile_size=tile_size, data_type='array',
                                 verbose=True, image=norm_image)    
        del norm_image
        
    else:
        tiled_image = FlipnSlide(tile_size=tile_size, data_type='array',
                                 verbose=True, image=img) 
        del img
    
    gc.collect()
    
    #Only preserve tiles that don't have any nans (i.e. aren't near the image edge)
    data_idx = [i for i, arr in enumerate(tiled_image.tiles) if not np.isnan(arr).any()]
    
    return tiled_image.tiles[data_idx,...]

#norm the images and create image tiles and corresponding mask tiles
tiled_image= preprocess_data(overall_image)

def save2bucket(save_path, bucket_name, array, mask_flag=False):
    '''
    Save the array to the bucket
    '''
    
    if mask_flag==False:
        destination_blob = save_path + f'image_tiles_{tile_size}.npy'
    else:
        destination_blob = save_path + f'mask_tiles_{tile_size}.npy'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)

    blob.upload_from_file(buffer, content_type='application/octet-stream')

print("Saving image tiles...")
save2bucket(args.save_path, bucket_name, tiled_image)
del tiled_image
gc.collect()

print("Starting mask tiling...")
tiled_mask= preprocess_data(overall_mask, mask_flag=True)

print("Saving mask tiles...")
save2bucket(args.save_path, bucket_name, tiled_mask, mask_flag=True)
del tiled_mask
gc.collect()