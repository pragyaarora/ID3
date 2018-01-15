from collections import namedtuple
import sys
import math
from Data import *

DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def InformationGain(data, f):
    n=0
    p=0
    count_l=[0,0]
    count_r=[0,0]
    for d in data:
        if d[f.feature] == f.value:
            if d[0] == POS_CLASS:
                count_l[0]+=1
                p += 1
            else:
                count_l[1]+=1
                n += 1
        else:
            if d[0] == POS_CLASS:
                count_r[0] +=1
                p += 1
            else:
                count_r[1] += 1
                n += 1


    p11= 0 if count_l[0]==0 else float(count_l[0])/(count_l[0]+count_l[1])
    p12= 0 if count_l[1]==0 else float(count_l[1])/(count_l[0]+count_l[1])
    p21= 0 if count_r[0]==0 else float(count_r[0])/(count_r[0]+count_r[1])
    p22= 0 if count_r[1]==0 else float(count_r[1])/(count_r[0]+count_r[1])

    p1=Entropy(p11,p12)
    p2=Entropy(p21,p22)

    par1=float(p)/(n+p)
    par2=float(n)/(n+p)

    return Entropy(par1,par2) - (float(count_l[0]+count_l[1])/(n+p))*p1 - (float(count_r[0]+count_r[1])/(n+p))*p2

def Entropy(p,q):
    if p==0 or q==0:
        return 0
    return -p*math.log(p,2)-q*math.log(q,2)


def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')        
        
def ID3(data, features, MIN_GAIN=0.1):
    #TODO: implement decision tree learning
    node = DtNode(FeatureVal('-','-'), (0,0), 0, None, None)

    if len(features) == 0:
        return node
    max = 0
    for f in features:
        ig = InformationGain(data,f)
        if ig>max:
            max=ig
            split_feature=f

    if max<=MIN_GAIN:
        pn = GetPosNeg(data)
        return DtNode(FeatureVal('-','-'), pn, 0, None, None)

    features.remove(split_feature)
    split_data = SplitData(data,split_feature)
    return DtNode(split_feature, GetPosNeg(data), max, ID3(split_data[0],features.copy(),MIN_GAIN), ID3(split_data[1],features.copy(),MIN_GAIN))

def GetPosNeg(data):
    p=0
    n=0
    for d in data:
        if d[0] == POS_CLASS:
            p+=1
        else:
            n+=1
    return (p, n)

def SplitData(data,f):
    left=[]
    right=[]
    for d in data:
        if d[f.feature] == f.value:
            left.append(d)
        else:
            right.append(d)

    return (left,right)

if __name__ == "__main__":
    train = MushroomData(sys.argv[1])
    dev = MushroomData(sys.argv[2])

    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    
    PrintTree(dTree)

    print Accuracy(dTree, dev.data)
