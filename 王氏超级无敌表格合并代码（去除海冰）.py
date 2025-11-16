#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据合并处理脚本（无空值版本）
用于提取nppv、thetao、chl、kd四个数据集的公共部分，并合并到同一个表格中
过滤掉所有包含空值的行
"""

import os
import pandas as pd
import re
import glob
from collections import defaultdict

# 数据目录
DATA_DIR = r"C:\Users\steve\Desktop\地图数据"
# 输出文件
OUTPUT_FILE = r"C:\Users\steve\Desktop\merged_data_no_na.xlsx"

def extract_location_and_variable(file_path):
    """
    从文件中提取位置信息和变量类型
    
    位置信息(location)的提取过程:
    1. 从CSV文件的元数据部分查找包含"Geometry: POINT"的行
    2. 使用正则表达式从该行提取经度(longitude)和纬度(latitude)坐标
    3. 将这对坐标作为元组(lon, lat)返回作为位置标识
    
    这些坐标随后会在main函数中被用于:
    - 对不同数据文件按位置分组
    - 生成格式为"Position[字母] time[数字]"的位置名称(如"PositionA time1")
    - 创建'处理位置'列，保存原始坐标信息
    """
    location = None
    variable = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Geometry: POINT" in line:
                match = re.search(r'POINT \(([0-9.-]+) ([0-9.-]+)\)', line)
                if match:
                    lon, lat = float(match.group(1)), float(match.group(2))
                    location = (lon, lat)
            
            # 提取变量名称
            if "Variable:" in line:
                if "nppv" in line.lower():
                    variable = "nppv"
                elif "thetao" in line.lower():
                    variable = "thetao"
                elif "chlorophyll" in line.lower() or "chl" in line.lower():
                    variable = "chl"
                elif "attenuation coefficient" in line.lower() or "kd" in line.lower():
                    variable = "kd"
                elif "so" in line.lower():
                    variable = "so"
                elif "spco2" in line.lower():
                    variable = "spco2"
            
            # 如果已经找到位置和变量，就可以退出循环
            if location and variable:
                break
                
    return location, variable

def read_data_file(file_path, variable_name):
    """
    读取数据文件，返回时间序列数据
    """
    
    if variable_name in ['nppv', 'thetao', 'chl', 'kd','so']:
        df = pd.read_csv(file_path,skiprows=8)
    else:
        df = pd.read_csv(file_path,skiprows=7)
    # 将时间列转换为日期时间格式，并确保时区为空
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # 确保列名正确
    if variable_name not in df.columns and len(df.columns) == 2:
        # 如果变量名不在列中，但有两列（时间和数据），则重命名第二列
        df = df.rename(columns={df.columns[1]: variable_name})
    
    return df

def main():
    # 按位置组织文件
    location_data = defaultdict(lambda: defaultdict(list))
    location_counter = {}  # 用于生成位置名称
    
    # 查找所有数据文件
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"找到 {len(all_files)} 个CSV文件")
    
    # 处理每个文件
    for file_path in all_files:
        location, variable = extract_location_and_variable(file_path)
        
        if location and variable and variable in ['nppv', 'thetao', 'chl', 'kd',"so","spco2"]:
            location_data[location][variable].append(file_path)
    
    print(f"共找到 {len(location_data)} 个不同位置的数据")
    
    # 创建结果DataFrame
    result_dfs = []
    position_index = 1
    
    # 处理每个位置的数据
    for location, variables in location_data.items():
        # 只处理有所有四种变量的位置
        if len(variables) == 6:
            lon, lat = location
            # 生成位置名称：使用"Position"前缀 + 字母(A-Z) + "time" + 数字
            # 例如：第一个位置为"PositionA time1"，第二个为"PositionB time2"，依此类推
            # chr(64+position_index)将数字转换为对应的大写字母：1->A, 2->B, ...
            location_name = f"Position{chr(64+position_index)} time{position_index}"
            position_index += 1
            
            print(f"处理位置 {location} 命名为 {location_name}")
            
            # 合并该位置的所有数据
            merged_data = pd.DataFrame()
            
            # 处理每种变量
            for variable, files in variables.items():
                # 合并同一变量的所有文件
                var_df = pd.DataFrame()
                
                for file in files:
                    df = read_data_file(file, variable)
                    if var_df.empty:
                        var_df = df
                    else:
                        var_df = pd.concat([var_df, df], ignore_index=True)
                
                # 去除重复的时间点
                var_df = var_df.drop_duplicates(subset=['time'])
                
                # 合并到总数据中
                if merged_data.empty:
                    merged_data = var_df
                else:
                    merged_data = pd.merge(merged_data, var_df[['time', variable]], on='time', how='outer')
            
            # 添加位置信息
            merged_data.insert(0, 'location', location_name)
            
            # 添加原始坐标信息（处理位置）
            merged_data['处理位置'] = f"({lon}, {lat})"
            
            result_dfs.append(merged_data)
    
    # 合并所有位置的数据
    if result_dfs:
        final_df = pd.concat(result_dfs, ignore_index=True)
        
        # 过滤掉包含空值的行
        final_df = final_df.dropna()
        
        print(f"过滤前数据行数: {len(final_df)}")
        print(f"过滤后数据行数: {len(final_df)}")
        
        # 按坐标（处理位置）和时间排序
        final_df = final_df.sort_values(['处理位置', 'time'])
        
        # 重新排列列，确保按照图片中的顺序，并将处理位置放在最后一列
        cols = ['location', 'time', 'nppv', 'thetao', 'kd', 'chl', "so","spco2",'处理位置']
        final_cols = [col for col in cols if col in final_df.columns]
        final_df = final_df[final_cols]
        
        # 保存到Excel文件
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"数据已保存到 {OUTPUT_FILE}")
    else:
        print("没有找到符合条件的数据")

if __name__ == "__main__":
    main()