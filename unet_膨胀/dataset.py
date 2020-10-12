import torch.utils.data as data
import os
import PIL.Image as Image
# 你要采用的训练图片的后缀有哪些？
def data_select(data_dir):  #
    import  glob
    file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    data = []
    print(file_list) # 得到文件的路径列表
    return file_list



# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    # 原图片保存在gray_origin文件夹里，mask图在gray_origin\mask文件夹里

    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        all_pic = data_select(data_dir=root) # 提取所有图片的路径
        n =( len(all_pic) //1 ) # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整

        imgs = []
        for i in range(n):
            img = all_pic[i]  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = (all_pic[i]).replace('gray_origin',r'gray_origin\mask')
            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]