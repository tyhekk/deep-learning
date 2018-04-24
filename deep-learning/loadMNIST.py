"""
mnist_loader
"""

import pickle
import gzip
import struct
#--------------------------------  
#提取训练集的标签
def getTrainLabel(file):#'./train-labels-idx1-ubyte.gz'
    f = gzip.open(file,'rb')
    data_byte = f.read()
    f.close()
    MSB_first, = struct.unpack('>l',data_byte[:4])
    print('train label MSB first:',MSB_first)
    data_num, = struct.unpack('>l',data_byte[4:8])
    print('data number:',data_num)
    train_label = []
    for i in range(data_num):
        train_label.append(ord(data_byte[8 + i:9 + i]))
    #保存数据
    #with open('./train_label.pkl','wb') as fout:
    #    pickle.dump(train_label,fout)
    return train_label
#--------------------------------  
#提取训练集的图片数据
def getTrainData(file):#'./train-images-idx3-ubyte.gz'
    f = gzip.open(file,'rb')
    data_byte = f.read()
    f.close()
    MSB_first, = struct.unpack('>l',data_byte[:4])
    print('train data MSB first:',MSB_first)
    data_num, = struct.unpack('>l',data_byte[4:8])
    image_row, = struct.unpack('>l',data_byte[8:12])
    image_column, = struct.unpack('>l',data_byte[12:16])
    print('data number:',data_num)
    train_data = []
    for i in range(data_num):
        image_pixel = []
        for j in range(image_row*image_column):
            #print(16 + image_row*image_column*i + j)
            image_pixel.append(ord(data_byte[16 + image_row*image_column*i + j:17 + image_row*image_column*i + j]))
        train_data.append(image_pixel)
    #保存数据
    #with open('./train_data.pkl','wb') as fout:
    #    pickle.dump(train_data,fout)
    return train_data
#--------------------------------  
#提取测试集的标签
def getTestLabel(file):#'./t10k-labels-idx1-ubyte.gz'
    f = gzip.open(file,'rb')
    data_byte = f.read()
    f.close()
    MSB_first, = struct.unpack('>l',data_byte[:4])
    print('test label MSB first:',MSB_first)
    data_num, = struct.unpack('>l',data_byte[4:8])
    print('data number:',data_num)
    test_label = []
    for i in range(data_num):
        test_label.append(ord(data_byte[8 + i:9 + i]))
    #保存数据
    #with open('./test_label.pkl','wb') as fout:
    #    pickle.dump(test_label,fout)
    return test_label
#--------------------------------   
#提取测试集的图片数据
def getTestData(file):#'./t10k-images-idx3-ubyte.gz'
    f = gzip.open(file,'rb')
    data_byte = f.read()
    f.close()
    MSB_first, = struct.unpack('>l',data_byte[:4])
    print('test data MSB first:',MSB_first)
    data_num, = struct.unpack('>l',data_byte[4:8])
    image_row, = struct.unpack('>l',data_byte[8:12])
    image_column, = struct.unpack('>l',data_byte[12:16])
    print('data number:',data_num)
    test_data = []
    for i in range(data_num):
        image_pixel = []
        for j in range(image_row*image_column):
            #print(16 + image_row*image_column*i + j)
            image_pixel.append(ord(data_byte[16 + image_row*image_column*i + j:17 + image_row*image_column*i + j]))
        test_data.append(image_pixel)
    #保存数据
    #with open('./test_data.pkl','wb') as fout:
    #    pickle.dump(test_data,fout)
    return test_data





