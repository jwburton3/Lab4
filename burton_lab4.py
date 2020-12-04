import os
import fiona
import glob
import numpy as np
import rasterio
from scipy.spatial import distance, cKDTree

def window(x):

    """
   This function is generating moving window dimensions
    """
    return x[row - 5:row + 6,
            col - 4:col + 5]
#wind
ws80m_rs = rasterio.open('/Users/jonathanburton/Desktop/Fall2020/Geog5092/lab4/data/ws80m.tif')
ws80m = ws80m_rs.read(1)
ws80m = np.where(ws80m < 0, 0, ws80m)
wind_arr = np.zeros_like(ws80m, np.float32)
for row in range(5, ws80m.shape[0] - 5):
    for col in range(4, ws80m.shape[1] - 4):
        win = window(ws80m)
        wind_arr[row, col] = win.mean()
ws80m_arry = np.where(wind_arr > 8.5, 1, 0)
#protected
pro_areas_rs = rasterio.open('./data/protected_areas.tif')
pro_areas = pro_areas_rs.read(1)
pro_areas = np.where(pro_areas < 0, 0, pro_areas)
pro_arr = np.zeros_like(pro_areas)
for row in range(5, pro_areas.shape[0] - 5):
    for col in range(4, pro_areas.shape[1] - 4):
        win = window(pro_areas)
        pro_arr[row, col] = win.mean()
pro_arry = np.where(pro_arr < 0.05, 1, 0)
#slope
slope_rs = rasterio.open('./data/slope.tif')
slope = slope_rs.read(1)
slope = np.where(slope < 0, 0, slope)
slope_arr = np.zeros_like(slope)
for row in range(5, slope.shape[0] - 5):
    for col in range(4, slope.shape[1] - 4):
        win = window(slope)
        slope_arr[row, col] = win.mean()
slope_arry = np.where(slope_arr < 15, 1, 0)  
#urban
urban_areas_rs = rasterio.open('./data/urban_areas.tif')
urban_areas = urban_areas_rs.read(1)
urban_arr = np.zeros_like(urban_areas)
urban_areas = np.where(urban_areas < 0, 0, urban_areas)
for row in range(5, urban_areas.shape[0] - 5):
    for col in range(4, urban_areas.shape[1] - 4):
        win = window(urban_areas)
        urban_arr[row, col] = win.mean()
urban_area_arry = np.where(urban_arr == 0, 1, 0)
#water
water_rs = rasterio.open('./data/water_bodies.tif')
waters = water_rs.read(1)
waters = np.where(waters < 0, 0, waters)
water_arr = np.zeros_like(waters)
for row in range(5, waters.shape[0] - 5):
    for col in range(4, waters.shape[1] - 4):
        win = window(waters)
    water_arr[row, col] = win.mean()
wtr_bdy_arry = np.where(waters < 0.02, 1, 0)
suitability_array = np.zeros_like(wtr_bdy_arry)
suitability_array = wtr_bdy_arry + urban_area_arry + slope_arry + pro_arry + ws80m_arry
suitability_array = np.where(suitability_array == 5, 1, 0)
suitability_array = suitability_array.astype('float32')
final_sites_num = suitability_array.sum()
print('There are', final_sites_num, 'locations that meet the listed criteria')
with rasterio.open('./data/slope.tif') as dataset:
    with rasterio.open('./data/suitable_sites.tif' , 'w',
                          driver='GTiff',
                          height=suitability_array.shape[0],
                          width=suitability_array.shape[1],
                          count=1,
                          dtype=np.float32,
                          crs=dataset.crs,
                            transform=dataset.transform,
                          nodta=dataset.nodata
                          ) as out_dataset:
        out_dataset.write(suitability_array,1)
xs = []
ys = []
with open(r'./data/transmission_stations.txt') as coords:
    lines = coords.readlines()[1:]
    for l in lines:
        x,y = l.split(',')
        xs.append(float(x))
        ys.append(float(y))
        stations = np.vstack([xs,ys])
        stations = stations.T
with rasterio.open(r'./data/suitable_sites.tif') as file:
    bounds = file.bounds
    top_left = (bounds[0], bounds[3])
    bot_right = (bounds[2], bounds[1])
    cellSize = 1000
    x_coords = np.arange(top_left[0] + cellSize/2, bot_right[0], cellSize)
    y_coords = np.arange(bot_right[1] + cellSize/2, top_left[1], cellSize)
    x,y = np.meshgrid(x_coords, y_coords)
    cent_coords = np.c_[x.flatten(), y.flatten()]
suitable_cent_coords = []
for sx, sy in zip(cent_coords, suitability_array.flatten()):
        xxx = np.multiply(sx[0], sy)
        yyy = np.multiply(sx[1], sy)
        if xxx != 0 and yyy != 0:
            suitable_cent_coords.append([xxx, yyy])
dd, ii = cKDTree(stations).query(suitable_cent_coords)
print("The shortest distance is ", np.min(dd)/1000, 'km')
print('The longest distance is ', np.max(dd)/1000, 'km')