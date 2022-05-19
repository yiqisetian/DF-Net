import os,shutil
import random
import json
import numpy as np
import torch

sourceDir = "/home/ping/ping/experiment/PointCloud/PF-Net-Point-Fractal-Network-master/dataset/shapenetcore_partanno_segmentation_benchmark_v0"
targetDir = "/home/ping/ping/experiment/PointCloud/PF-Net-Point-Fractal-Network-master/dataset/shapenet_pc"
listDir="/home/ping/ping/experiment/PointCloud/PF-Net-Point-Fractal-Network-master/dataset/shapenet16"
jsonDir="/home/ping/ping/experiment/PointCloud/PF-Net-Point-Fractal-Network-master/dataset/shapenetcore_partanno_segmentation_benchmark_v0/train_test_split"
pcUncropDir="/home/ping/ping/experiment/PointCloud/PoinTr-master/data/ShapeNet16/shapenet_pc_uncrop"
pcDir="/home/ping/ping/experiment/PointCloud/PoinTr-master/data/ShapeNet16/shapenet_pc"
list13="/home/ping/ping/experiment/PointCloud/PoinTr-master/data/ShapeNet16/shapenet13"

def CopyPCFile():
    filelist=[]
    for dir in os.listdir(sourceDir):
        taxonomy_dir=os.path.join(sourceDir,dir,'points')
        for file in os.listdir(taxonomy_dir):
            shutil.copyfile(os.path.join(taxonomy_dir,file),os.path.join(targetDir,dir+"-"+file))
            filelist.append(dir+'-'+file)

# trainId=random.sample(list(range(len(filelist))),int(len(filelist)*0.8))
def SplitDataset(testNum=2768):
    filelist=[]
    for dir in os.listdir(sourceDir):
        taxonomy_dir=os.path.join(sourceDir,dir,'points')
        for file in os.listdir(taxonomy_dir):
            filelist.append(dir+'-'+file)
    fileIndex=list(range(len(filelist)))
    testId = random.sample(fileIndex, testNum)
    trainId= list(set(fileIndex)-set(testId))
    random.shuffle(testId)
    random.shuffle(trainId)
    trainFile=open(os.path.join(listDir,'train.txt'),'a')
    for fileName in trainId:
        trainFile.write(str(filelist[fileName]))
        trainFile.write('\r\n')
    testFile=open(os.path.join(listDir,'test.txt'),'a')
    for fileName in testId:
        testFile.write(str(filelist[fileName]))
        testFile.write('\r\n')

def jsonToTxt():
    trainList=[]
    testList=[]
    trainFile=open(os.path.join(jsonDir,'shuffled_train_file_list.json'))
    trainSample=json.load(trainFile)
    for line in trainSample:
        taxonomy_id = line.split('/')[1]
        model_id = line.split('/')[2]
        sampleName=taxonomy_id+'-'+model_id+".pts"
        if taxonomy_id=="04099429" or taxonomy_id=="03261776" or taxonomy_id=="03624134":
            continue
        trainList.append(sampleName)

    # valFile = open(os.path.join(jsonDir, 'shuffled_val_file_list.json'))
    # valSample = json.load(valFile)
    # for line in valSample:
    #     taxonomy_id = line.split('/')[1]
    #     model_id = line.split('/')[2]
    #     sampleName = taxonomy_id + '-' + model_id + ".pts"
    #     if taxonomy_id=="04099429" or taxonomy_id=="03261776" or taxonomy_id=="03624134":
    #         continue
    #     trainList.append(sampleName)

    trainTxt = open(os.path.join(listDir, 'train.txt'), 'a')
    for fileName in trainList:
        trainTxt.write(fileName)
        trainTxt.write('\r\n')

    testFile = open(os.path.join(jsonDir, 'shuffled_test_file_list.json'))
    testSample = json.load(testFile)
    for line in testSample:
        taxonomy_id = line.split('/')[1]
        model_id = line.split('/')[2]
        sampleName = taxonomy_id + '-' + model_id + ".pts"
        if taxonomy_id=="04099429" or taxonomy_id=="03261776" or taxonomy_id=="03624134":
            continue
        testList.append(sampleName)
    testTxt = open(os.path.join(listDir, 'test.txt'), 'a')
    for fileName in testList:
        testTxt.write(fileName)
        testTxt.write('\r\n')

#randomType控制是否随机选择视角，0为随机，1-5为5个视角点
#遍列所有的原始点云文件，生成随机视角的2048个点的点云文件，16类
def random_center(pointNum=2048,randomType=0):
    choice = [[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0],[-1, 1, 0]]  # 设置5个点的视角
    i=0
    sNum=0
    fileNum=len(os.listdir(pcUncropDir))
    for pc in os.listdir(pcUncropDir):#遍历16类的点云文件
        points=np.loadtxt(os.path.join(pcUncropDir,pc)).astype(np.float32)
        points=pc_normalize(points)
        if randomType==0:
            index = random.sample(choice, 1)
            p_center = index[0]
        else:
            p_center=choice(randomType)
        points=points[np.random.choice(len(points),pointNum,replace=True)] #采样到2048个点
        if len(points)<pointNum:
            sNum=sNum+1
        distance_list = []
        for n in range(pointNum):
            distance_list.append(distance_squre(points[n], p_center))#计算距离视角点的差值
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])  # list[2048]，排序
        real_center=[]
        for sp in range(pointNum):
           real_center.append(points[distance_order[sp][0]])
        pts = open(os.path.join(pcDir, pc), 'a')
        for p in real_center:
            pts.write(str(p).replace('[','').replace(']',''))
            pts.write('\r\n')
        pts.close()
        i=i+1
        print(str(i)+"/"+str(fileNum)+"/"+str(sNum))

def distance_squre(p1,p2):
    p=p1-p2
    val=np.multiply(p,p,dtype=np.float32)
    val = np.sum(val)
    return val

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

if __name__=='__main__':
    #CopyPCFile()
    #SplitDataset()
    #jsonToTxt()
    random_center()