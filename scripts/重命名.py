
# coding=utf-8
# 将png文件复制并重命名
# 使用os模块可以获取指定文件夹下所有文件名，有两个方法os.walk()和os.listdir().
# (1)os.walk可以用于遍历指定文件下所有的子目录、非目录子文件。


'''
#os.listdir()用于返回指定的文件夹下包含的文件或文件夹名字的列表，这个列表按字母顺序排序。

import os
filePath = 'C://myLearning//pythonLearning201712//carComments//01//'
os.listdir(filePath)
'''

import shutil

def copyfile(origin_filename, targetFile):
    shutil.copy(origin_filename, targetFile)


# 特定的文件后缀保存
def data_select(data_dir):  #
    import glob
    file_list = list(glob.glob(data_dir + '/*.png')) + list(
        glob.glob(data_dir + '/*.jpg'))  # get name list of all .png files
    data = []
    print(file_list)  # 得到文件的路径列表
    return file_list

if __name__ == '__main__':
    data_dir  = r'F:\all_datas\gray_origin'
    pic_all  = data_select(data_dir)
    print(pic_all)
    for i in pic_all:

        origin_filename = i
        # 复制1w张
        savefliename = (origin_filename.replace('.png','_mask.png')).replace('gray_origin','gray_origin\mask')
        targetFile = savefliename
        copyfile(origin_filename  ,savefliename)
