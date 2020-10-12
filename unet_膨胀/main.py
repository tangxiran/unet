import torch
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
import unet
from torch import optim
from dataset import LiverDataset
from torch.utils.data import DataLoader

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    # 保存位置
    torch.save(model.state_dict(), 'checkpoint//weights_%d.pth' % epoch)  # 返回模型的所有内容
    return model


# 训练模型
def train(in_channels=3,out_channels=1,loss_function='bce',train_data_dir=r"F:\all_datas\gray_origin"):
    train_data_dir = train_data_dir
    model = unet.UNet(in_channels, out_channels).to(device)
    batch_size = args.batch_size
    # 损失函数选择
    criterion = torch.nn.BCELoss()
    if loss_function == 'l1' or loss_function == 'L1':
        criterion = torch.nn.L1Loss()
    # criterion = torch.nn.L1Loss()
    if loss_function=='bce' or loss_function=='BCE':
        criterion = torch.nn.BCELoss()

    # 二分类交叉熵损失函数
    # 梯度下降
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    liver_dataset = LiverDataset(train_data_dir, transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # 报错4 改为0好一点
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion, optimizer, dataloader)


# 测试
def test(in_channels=3,out_channels=1):
    # inchanel 改为1 为灰度图
    model = unet.UNet(in_channels,out_channels)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    liver_dataset = LiverDataset(root=r"test\gray_origin", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size默认为1
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('action', type=str, help='train or test')  # 添加参数
    # batch改为几合适？看内存大小
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    parser.add_argument('--loss_function', type=str, default='bce',help='the loss function you can choose,select l1 or bce')
    parser.add_argument('--inchannels', type=int,default=3, help='pic channel ,select 1 or 3 ')

    args = parser.parse_args()

    # 图片是彩色还是灰度图，彩色选3
    inchannels= args.inchannels
    # 输入训练图片的路径
    train_data_dir = r"F:\all_datas\gray_origin"
    # 损失函数
    loss_function = args.loss_function
    # inchannels=3
    if args.action == 'train':
        train(in_channels = inchannels,out_channels=1,loss_function=loss_function,train_data_dir=train_data_dir)
    elif args.action == 'test':
        test(in_channels = inchannels,out_channels=1)