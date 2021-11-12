## YOLOv1  
大致流程:  
1.Resize成448\*448，图片分割得到7\*7网格(cell)  
2.CNN提取特征和预测：卷积部分负责提特征。全连接部分负责预测：a) 7\*7\*2=98个bounding box(bbox) 的坐标(cx,cy,w,h)和是否有物体的conﬁdence 。 b) 7\*7=49个cell所属20个物体的概率。  
3.过滤bbox（通过nms）  
 
网络最后的输出是 S×S×30 的数据块，yolov1是含有全连接层的，这个数据块可以通过reshape得到。也就是说，输出其实已经丢失了位置信息（在v2，v3中用全卷积网络，每个输出点都能有各自对应的感受野范围）。yolov1根据每张图像的目标label，编码出一个 S×S×(B\*5+20) 的数据块，然后让卷积网络去拟合这个target
### 1.S\*S 框
如果Ground Truth的中心落在某个单元（cell）内，则该单元负责该物体的检测  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-42-06.png)
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-42-47.png)  
我们将输出层Os\*s\*(c+B\*5) 看做一个三维矩阵，如果物体的中心落在第 (i,j) 个单元内，那么网络只优化一个 C+B\*5维的向量，即向量 O[i,j,:] 。 S 是一个超参数，在源码中 S=7,B是每个单元预测的bounding box的数量，B的个数同样是一个超参数，YOLO使用多个bounding box是为了每个cell计算top-B个可能的预测结果，这样做虽然牺牲了一些时间，但却提升了模型的检测精度。  

每个bounding box要预测5个值：bounding box(cx,cy,w,h)以及置信度P,定义confidence为Pr(object)\*IOU(pred,truth)。bbox的形状是任意猜测的，这也是后续yolov2进行优化的一个点。置信度P表示bounding box中物体为待检测物体的概率以及bounding box对该物体的覆盖程度的乘积 Pr(Object) * IOU(pred, truth)。其中(cx,cy)是bounding box相对于每个cell中心的相对位置， (w,h)是物体相对于整幅图的尺寸,范围均为[0,1]。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-13.png) 

同时，YOLO也预测检测物体为某一类C的条件概率：Pr(class(i)|object)  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-37.png)  

对于每一个单元，YOLO值计算一个分类概率，而与B的值无关。在测试时  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-52.png)  
### 2.输入层　　
YOLO作为一个统计检测算法，整幅图是直接输入网络的。因为检测需要更细粒度的图像特征，YOLO将图像Resize到了448\*448而不是物体分类中常用的224\*224的尺寸。需要注意的是YOLO并没有采用VGG中先将图像等比例缩放再裁剪的形式，而是直接将图片非等比例resize。所以YOLO的输出图片的尺寸并不是标准比例的。 
### 3.骨干架构：VGG,leaky-relu  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-44-32.png)  
然而现在的一些文章指出leaky ReLU并不是那么理想，现在尝试网络超参数时ReLU依旧是首选。　　
### 4.Loss function  
#### loss = classification loss + localization loss + confidence loss  
#### classification loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-44-53.png)
#### localization loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-00.png)  
对不同大小的bbox预测中，相比于大bbox预测偏一点，小box预测偏一点更不能忍受。而sum-square error loss中对同样的偏移loss是一样。 为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。 如下图：small bbox的横轴值较小，发生偏移时，反应到y轴上的loss（下图绿色）比big box(下图红色)要大。
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-06.png)

为了更加强调边界框的准确性，我们设定lambda(coord)=5(default)  
#### confidence loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-12.png)  
许多bbox不包含任何物体，这造成了类别不平衡问题，eg:我们训练的模型检测到背景的情况会比物体的情况多的多，为了解决这个问题，我们设定lambda(noobj)=0.5(default)  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-21.png)  
### 5.后处理　　　
测试样本时，有些物体会被多个单元检测到，NMS用于解决这个问题。 

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2013-49-58.png) 
### YOLO优点：　　
1.YOLO检测物体非常快 　
 
2.不像其他物体检测系统使用了滑窗或region proposal，分类器只能得到图像的局部信息。YOLO在训练和测试时都能够看到一整张图像的信息，因此YOLO在检测物体时能很好的利用上下文信息，从而不容易在背景上预测出错误的物体信息。和Fast-R-CNN相比，YOLO的背景错误不到Fast-R-CNN的一半    

3.YOLO可以学到物体的泛化特征.  
  
4.当YOLO在自然图像上做训练，在艺术作品上做测试时，YOLO表现的性能比DPM、R-CNN等之前的物体检测系统要好很多。因为YOLO可以学习到高度泛化的特征，从而迁移到其他领域。    

### YOLO的缺点
1.其精确检测的能力比不上Fast R-CNN更不要提和其更早提出的Faster R-CNN了。  

2.由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。虽然每个格子可以预测B个bounding box，但是最终只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。  

3.YOLO对小物体的检测，其为了提升速度而粗粒度划分单元而且每个单元的bounding box的功能过度重合导致模型的拟合能力有限，尤其是其很难覆盖到的小物体。YOLO检测小尺寸问题效果不好的另外一个原因是因为其只使用顶层的Feature Map，而顶层的Feature Map已经不会包含很多小尺寸的物体的特征了。

4.Faster R-CNN之后的算法均趋向于使用全卷积代替全连接层，但是YOLO依旧笨拙的使用了全连接不仅会使特征向量失去对于物体检测非常重要的位置信息，容易产生物体的定位错误，而且会产生大量的参数，影响算法的速度。  

## YOLOv2  
在yolov1的基础上进行改进　　

## YOLOv2     
### 1.BN替代Dropout   
神经网络学习过程本质就是为了学习数据分布,一旦训练数据与测试数据的分布不同,那么网络的泛化能力也大大降低;另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降),那么网络就要在每次迭代都去学习适应不同的分布,这样将会大大降低网络的训练速度。 对数据进行预处理（统一格式、均衡化、去噪等）能够大大提高训练速度，提升训练效果。

解决办法之一是对数据都要做一个归一化预处理。YOLOv2网络通过在每一个卷积层后添加batch normalization，极大的改善了收敛速度同时减少了对其regularization方法的依赖（舍弃了dropout优化后依然没有过拟合）
Batch Normalization和Dropout均有正则化的作用。但是Batch Normalization具有提升模型优化的作用，这点是Dropout不具备的。所以BN更适合用于数据量比较大的场景。  

### 2.High Resolution Classifier  
预训练分类模型采用了更高分辨率的图片  
YOLOv1先在ImageNet（224x224）分类数据集上预训练模型的主体部分（大部分目标检测算法），获得较好的分类效果，然后再训练网络的时候将网络的输入从224x224增加为448x448。但是直接切换分辨率，检测模型可能难以快速适应高分辨率。所以YOLOv2增加了在ImageNet数据集上使用448x448的输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。

### 3.Darknet-19网络框架　　
提出了新的分类网络darknet-19作为基础模型　　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-32.png)  
用3\*3的filter来提取特征，1\*1的filter减少output channels.  
最后的卷积层用3\*3的卷积层代替,filter=1024,然后用1\*1的卷积层将7\*7\*1024-->7\*7\*125 注:125 = (5\*(4+1+20)    
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-40.png)   
　　
使用1\*1的卷积是为了减少参数,Darknet-19进行了5次降采样，但是在最后一层卷积并没有添加池化层，目的是为了获得更高分辨率的Feature Map,在3\*3卷积中间添加了1\*1卷积，Feature Map之间的一层非线性变化提升了模型的表现能力 

### 4.Convolutional with anchor boxes  
YOLO采用全连接层来直接预测bounding boxes,yolov2去除了YOLO的全连接层，采用先验框（anchor boxes）来预测bounding boxes。    
首先，去除了一个pooling层来提高卷积层输出分辨率。然后，修改网络输入尺寸：考虑到很多情况下待检测物体的中心点容易出现在图像的中央，将448×448改为416，使特征图只有一个中心。物品（特别是大的物品）更有可能出现在图像中心。　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-47.png)  
使用416\*416经过5次降采样之后生成的Feature Map的尺寸是13\*13 ，这种奇数尺寸的Feature Map获得的中心点的特征向量更准确。其实这也和YOLOv1产生7\*7的理念是相同的；采用anchor boxes，提升了精确度。YOLO每张图片预测98个boxes，但是采用anchor boxes，每张图片可以预测超过1000个boxes

### 5.Dimension Clusters  
为了确定对训练数据具有最佳覆盖范围的前K个bbox，我们在训练数据上运行K-均值聚类，以找到前K个聚类的质心。由于我们是在处理边界框而不是点，因此无法使用规则的空间距离来测量数据点距离。所以我们使用IOU.聚类的目的是anchor boxes和临近的ground truth有更大的IOU值，这和anchor box的尺寸没有直接关系。自定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid),距离越小，IOU值越大。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-55.png)  
图中看出anchor个数选为5时，结果最佳。右侧表示anchor的形状  

### 6.Direct Location Prediciton　　
对anchor的偏移量进行预测，但是如果不限制我们的预测将再次随机化，pre-bbox很容易向任何方向偏移。因此，每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的offsets.yolov2预测了5个参数(tx,ty,tw,th,to),应用sigma函数来限制偏移范围。　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-04.png)　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-11.png)

### 7.Fine-Grained Features
5次maxpooling得到的13×13大小的feature map对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained Features）.YOLOv2提出了一种passthrough层来利用更精细的特征图。YOLOv2所利用的Fine-Grained Features是26×26大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为26×26×512的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个2×2的局部区域，然后将其转化为channel维度，对于26×26×512的特征图，经passthrough层处理之后就变成了13×13×2048的新特征图（特征图大小降低4倍，而channles增加4倍)，这样就可以与后面的13×13×1024特征图连接在一起形成13×13×3072的特征图，然后在此特征图基础上卷积做预测。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-18.png)  

### 8.Training   
分类任务训练  

作者采用ImageNet1000类数据集来训练分类模型。训练过程中，采用了 random crops, rotations, and hue, saturation, and exposure shifts等data augmentation方法。预训练后，作者采用高分辨率图像（448×448）对模型进行finetune。

检测任务训练  

作者将分类模型的最后一层卷积层去除，替换为三层卷积层（3×3,1024 filters），最后一层为1×1卷积层，filters数目为需要检测的数目。对于VOC数据集，我们需要预测5个boxes，每个boxes包含5个适应度值，每个boxes预测20类别。因此，输出为125（5*20+5*5） filters。最后还加入了passthough 层。  

## YOLOv3  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-35-10.png)  
CBL:conv + BN + LeakyRELU  
YOLO3主要的改进有：调整了网络结构使用了Darknet-53；利用多尺度特征进行对象检测；对象分类用Logistic取代了softmax.  
### Darknet-53  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-35-18.png)   
### anchor  
YOLOv2已经开始采用K-means聚类得到先验框的尺寸，YOLOv3延续了这种方法，为每种下采样尺度设定3种先验框，总共聚类出9种尺寸的先验框。在COCO数据集这9个先验框是：(10x13)，(16x30)，(33x23)，(30x61)，(62x45)，(59x119)，(116x90)，(156x198)，(373x326)。  

分配上，在最小的13\*13特征图上（有最大的感受野）应用较大的先验框(116x90)，(156x198)，(373x326)，适合检测较大的对象。中等的26\*26特征图上（中等感受野）应用中等的先验框(30x61)，(62x45)，(59x119)，适合检测中等大小的对象。较大的52\*52特征图上（较小的感受野）应用较小的先验框(10x13)，(16x30)，(33x23)，适合检测较小的对象。  
感受一下9种先验框的尺寸，下图中蓝色框为聚类得到的先验框。黄色框式ground truth，红框是对象中心点所在的网格.  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-35-32.png)
### sigmoid  
预测对象类别时不使用softmax，改成使用logistic的输出进行预测。这样能够支持多标签对象，这样每个类的输出仍是[0,1]之间的一个值，但是他们的和不再是1。只要置信度大于阈值，该锚点便被作为检测框输出。
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-35-40.png)  
对于一个416\*416的输入图像，YOLOv3在每个尺度的特征图的每个网格设置3个先验框，总共有 13\*13\*3 + 26\*26\*3 + 52\*52\*3 = 10647 个预测。每一个预测是一个(4+1+80)=85维向量，这个85维向量包含边框坐标（4个数值），边框置信度（1个数值），对象类别的概率（对于COCO数据集，有80种对象）。对比一下，YOLO2采用13\*13\*5 = 845个预测，YOLO3的尝试预测边框数量增加了10多倍，而且是在不同分辨率上进行，所以mAP以及对小物体的检测效果有一定的提升。

## YOLOv4  

1.用单卡就能完成检测训练过程.  

2.Bag-of-Freebies and Bag-of-Specials

Bag-of-Freebies:指目标检测器在不增加推理损耗的情况下达到更好的精度，这些方法称为只需转变训练策略或只增加训练量成本。也就是说数据增广、类标签平滑(Class label smoothing)、Focal Loss等这些不用改变网络结构的方法   
Bag-of-Special:用最新最先进的方法（网络模块）来魔改检测模型--插入模块是用来增强某些属性的，显著提高目标检测的准确性。比如SE模块等注意力机制模块，还有FPN等模块。  

3.除了在模型上进行魔改，作者还加了其他的技巧   

backbone主要是提取特征，去掉head也可以做分类任务  
Neck主要是对特征进行融合，这里有很多技巧在  
### 核心目标  
加快神经网络的运行速度，在生产系统中优化并行计算，而不是低计算量理论指标（BFLOP） 
作者实验对比了CSPResNext50、CSPDarknet53和EfficientNet-B3。从理论与实验角度表明：CSPDarkNet53更适合作为检测模型的Backbone  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-26.png)  
#### 总结:YOLOv4模型 = CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head  
### BoF和BoS的选择  
对于改进CNN通常使用一下方法:   
1.activation:ReLU, leaky-ReLU, (parametric-ReLU and SELU训练难度大,ReLU6专门量化网络的设计,没选用这3个激活函数), Swish,or **Mish**  
   
2.bbox regression loss:MSE, IoU->GIoU->DIoU->CIoU  

3.data augumentation:CutOut, **MixUp**, **CutMix**, **Mosaic**.   
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-36.png)
4.regulation:DropOut, DropPath, Spatial DropOut ,or **DropBlock**  
Dropout:整体随便扔    
Spatial Dropout: 按通道随机扔    
**DropBlock**:每个特征图按spatial块随机扔    
DropConnect:只在连接处随意扔，神经元不扔    

5.normalization:Batch Normalization (BN) , Cross-GPU Batch Normalization (CGBN or SyncBN) , Filter Response Normalization (FRN),or Cross-Iteration Batch Normalization (CBN)  

6.skip-connections:Residual connections, Weighted residual connections(from EfficientDet), Multi-input weighted residual connections, or Cross stage partial connections (CSP)  

对于以上方法，作者又进行了额外更改:  
**输入端的创新:**    
1.data augument:Mosaic,SAT(自我对抗训练)  

2.Class label smoothing(标签平滑):通常，将bbox的正确分类表示为类别[0,0,0,1,0,0，...]的独热编码，并基于该表示来计算损失函数。但是，当模型对预测值接近1.0,变得过分确定时，通常是错误的，overfit，并且以某种方式忽略了其他预测值的复杂性。按照这种直觉，在某种程度上重视这种不确定性对类标签进行编码是更合理的。当然，作者选择0.9，因此[0,0,0,0.9，0 ....]代表正确的类

3.SAT:通过转换输入图片来反应了漏洞.首先，图片通过正常的训练步骤，然后用对于模型来说根据最有害的loss值来修改图片，而不是反向传播更新权重，在后面的训练中，模型不得不强制面对最困难的例子并学习它。 
 
4.BN-->CBN-->**CmBN**  
CBN:  
(1)作者认为连续几次训练iteration中模型参数的变化是平滑的  
(2)作者将前几次iteration的BN参数保存起来，当前iteration的BN参数由当前batch数据求出的BN参数和保存的前几次的BN参数共同推算得出(Cross-Interation BN)  
(3)训练前期BN参数记忆长度短一些，后期训练稳定了可以保存更长时间的BN参数来参与推算，效果更好  
CmBN:  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-48.png)  

## yolov4框架  
CSPDarknet53借鉴CSPNet,CSPNet全称是Cross Stage Paritial Network，主要从网络结构设计的角度解决推理中从计算量很大的问题。CSPNet的作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-54.png)  

yolov4创新点:   

### 输入端创新

这里指的创新主要是训练时对输入端的改进，主要包括**Mosaic数据增强、cmBN、SAT自对抗训练**.  

Mosiac:  

a. 丰富数据库：随机使用4张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好  
b. 减少GPU：可能会有人说，随机缩放，普通的数据增强也可以做，但作者考虑到很多人可能只有一个GPU。  
因此Mosaic增强训练时，可以直接计算4张图片的数据，使得Mini-batch大小并不需要很大，一个GPU就可以达到比较好的效果  

CmBN:  

SAT自对抗训练:  

自对抗训练也是一种新的数据增强方法，可以一定程度上抵抗对抗攻击。其包括两个阶段，每个阶段进行一次前向传播和一次反向传播。  
第一阶段，CNN通过反向传播改变图片信息，而不是改变网络权值。通过这种方式，CNN可以进行对抗性攻击，改变原始图像，造成图像上没有目标的假象。  
第二阶段，对修改后的图像进行正常的目标检测。  

（2）BackBone主干网络：将各种新的方式结合起来，包括：**CSPDarknet53、Mish激活函数、Dropblock**  
（3）Neck：目标检测网络在BackBone和最后的输出层之间往往会插入一些层，比如Yolov4中的SPP模块、FPN+PAN结构  
（4）Prediction：输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数CIOU_Loss，以及预测框筛选的nms变为DIOU_nms  

### backbone主干网络创新   

**CSPDarknet网络结构**    
优点:  
增加CNN学习能力，使得在轻量化的同时保持准确性,降低计算瓶颈,降低内存成本  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-02.png)   
每个CSP模块前面的卷积核的大小都是3×3，因此可以起到下采样的作用。  
因为Backbone有5个CSP模块，输入图像是608\*608，所以特征图变化的规律是：608->304->152->76->38->19,经过5次CSP模块后得到19\*19大小的特征图。  
而且作者只在Backbone中采用了Mish激活函数，网络后面仍然采用Leaky_relu激活函数。  
backbone卷积层个数:  
每个CSPX中包含3+2×X个卷积层，因此整个主干网络Backbone中一共包含2+（3+2×1）+2+（3+2×2）+2+（3+2×8）+2+（3+2×8）+2+（3+2×4）+1=72  

**DropBlock**  
Dropout的方式会随机的删减丢弃一些信息，但Dropblock的研究者认为，卷积层对于这种随机丢弃并不敏感，因为卷积层通常是三层连用：卷积+激活+池化层，池化层本身就是对相邻单元起作用。而且即使随机丢弃，卷积层仍然可以从相邻的激活单元学习到相同的信息。因此，在全连接层上效果很好的Dropout在卷积层上效果并不好。所以Dropblock的研究者则干脆整个局部区域进行删减丢弃。

这种方式其实是借鉴2017年的cutout数据增强的方式，cutout是将输入图像的部分区域清零，而Dropblock则是将Cutout应用到每一个特征图。而且并不是用固定的归零比率，而是在训练时以一个小的比率开始，随着训练过程线性的增加这个比率。  
Dropblock的研究者与Cutout进行对比验证时，发现有几个特点：  

优点一：Dropblock的效果优于Cutout  
优点二：Cutout只能作用于输入层，而Dropblock则是将Cutout应用到网络中的每一个特征图上  
优点三：Dropblock可以定制各种组合，在训练的不同阶段可以修改删减的概率，从空间层面和时间层面，和Cutout相比都有更精细的改进。  

Yolov4中直接采用了更优的Dropblock，对网络的正则化过程进行了全面的升级改进  

**Mish**  

### Neck创新  
**SPP模块**  
采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合concat操作。   
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-09.png)   
**FPN+PAN**  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-15.png)  
**modified PANet**  
PANet融合的时候使用的方法是Addition,yolov4将加法改为了concatenate  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-22.png)  

### prediction  

输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数CIOU_Loss，以及预测框筛选的nms变为DIOU_nms  
IOU_Loss：主要考虑检测框和目标框重叠面积  

GIOU_Loss：在IOU的基础上，解决边界框不重合时的问题。  

DIOU_Loss：在IOU和GIOU的基础上，考虑边界框中心点距离的信息。  

CIOU_Loss：在DIOU的基础上，考虑边界框宽高比的尺度信息  

不过这里需要注意几点：
注意一：
Yolov3的FPN层输出的三个大小不一的特征图①②③直接进行预测
但Yolov4的FPN层，只使用最后的一个76×76特征图①，而经过两次PAN结构，输出预测的特征图②和③。
这里的不同也体现在cfg文件中，这一点有很多同学之前不太明白。
比如Yolov3.cfg中输入时608×608，最后的三个Yolo层中，
第一个Yolo层是最小的特征图19×19，mask=6,7,8，对应最大的anchor box  
第二个Yolo层是中等的特征图38×38，mask=3,4,5，对应中等的anchor box  
第三个Yolo层是最大的特征图76×76，mask=0,1,2，对应最小的anchor box  
而Yolov4.cfg则恰恰相反  
第一个Yolo层是最大的特征图76×76，mask=0,1,2，对应最小的anchor box  
第二个Yolo层是中等的特征图38×38，mask=3,4,5，对应中等的anchor box  
第三个Yolo层是最小的特征图19×19，mask=6,7,8，对应最大的anchor box   
其他基础操作：
1. Concat：张量拼接，维度会扩充，和Yolov3中的解释一样，对应于cfg文件中的route操作。
2. Add：张量相加，不会扩充维度，对应于cfg文件中的shortcut操作  
