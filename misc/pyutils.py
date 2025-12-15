import random
import glob
import numpy as np
import os

"""
这段代码提供了几个实用的函数，用于设置随机种子、创建目录以及获取文件路径列表。以下是每个函数的简要说明：

seed_random(seed=2020):

这个函数用于固定随机种子，确保实验的可重复性。
它设置了 random 模块和 NumPy 模块的随机种子，并且设置了环境变量 PYTHONHASHSEED。
mkdir(path):

这个函数用于创建一个目录，如果该目录不存在的话。
它接收一个路径作为参数，使用 os.path.exists 检查目录是否存在，如果不存在，则使用 os.makedirs 创建目录。
get_paths(image_folder_path, suffix='*.png'):

这个函数用于从指定的文件夹中获取所有符合特定后缀的文件路径。
它使用 glob.glob 来查找所有匹配的文件路径，并且使用 sorted 对结果进行排序。
get_paths_from_list(image_folder_path, list):

这个函数用于从指定的文件夹中获取列表中包含的文件路径。
它遍历提供的列表，将列表中的每个项与文件夹路径结合，然后返回一个排序后的路径列表。
请注意，这段代码中引用了 os 和 np（NumPy）模块，但是在代码片段中并没有显示导入这些模块的语句。在使用这段代码之前，需要确保导入了这些模块：

import os
import numpy as np
此外，seed_random 函数中的 np 应该是 NumPy 模块的别名，但在提供的代码片段中也没有显示导入 NumPy 模块。确保在使用 seed_random 函数之前，已经以 np 为别名导入了 NumPy 模块。

这些函数在数据预处理、文件管理和实验设置中非常有用，特别是在需要确保实验结果可重复性的情况下。
"""

def seed_random(seed=2020):
    # 加入以下随机种子，数据输入，随机扩充等保持一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_paths(image_folder_path, suffix='*.png'):
    """从文件夹中返回指定格式的文件
    :param image_folder_path: str
    :param suffix: str
    :return: list
    """
    paths = sorted(glob.glob(os.path.join(image_folder_path, suffix)))
    return paths


def get_paths_from_list(image_folder_path, list):
    """从image folder中找到list中的文件，返回path list"""
    out = []
    for item in list:
        path = os.path.join(image_folder_path,item)
        out.append(path)
    return sorted(out)
