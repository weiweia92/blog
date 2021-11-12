## 基于region proposal的目标检测算法
RCNN系列介绍  

## RCNN
### 1.RCNN的过程分4个阶段:  

候选区域提出阶段（Proposal):采用selective-search(是按照颜色和纹理不断合并得到候选区域的，候选区域的产生没有规律)方法，从一幅图像生成1K~2K个候选区域;  
特征提取：对每个候选区域，使用CNN进行特征提取;  
分类：每个候选区域的特征放入分类器SVM，得到该候选区域的分类结果;  
回归：候选区域的特征放入回归器，得到bbox的修正量.  

### 2.1候选区域提出阶段所产生的结果尺寸不同?
由于RCNN特征提取阶段采用的是AlexNet，其最后两层是全连接层fc6和fc7，所以必须保证输入的图片尺寸相同.  

而候选区域所产生的结果尺寸是不相同的。为此，论文中作者采用了多种方式对图片进行放缩（各向同性、各向异性、加padding），最后经过对比实验确定各向异性加padding的放缩方式效果最好.  

### 2.2 分类器SVM使用的是二分类?
论文中，单个SVM实现的是二分类，分类器阶段由多个SVM组合而成。比如总共有20种不同的物体（加1种背景），那么分类阶段必须要有21个SVM：第1个SVM的输出是该候选区域属于分类1的概率；第2个SVM的输出是该候选区域属于分类2的概率；……；第21个SVM的输出是该候选区域属于背景的概率.  

对21个SVM的输出结果进行排序，哪个输出最大，候选区域就属于哪一类。比如，对于某个候选区域，第21个SVM的输出最大，那么就将该候选区域标为背景。  

### 2.3 分类器的输入是?回归器的输入是?
分类器的输入是特征提取器AlexNet的fc6的输出结果，回归器的输入是特征提取器AlexNet的pool5的输出结果。

之所以这样取输入，是因为，分类器不依赖坐标信息，所以取fc6全连接层的结果是没有问题的。但是回归器依赖坐标信息（要输出坐标的修正量），必须取坐标信息还没有丢失前的层。而fc6全连接层已经丢失了坐标信息(空间结构信息)。  

### 2.4 正负样本选择?
我们理所当然的认为训练时是把region proposal阶段生成的候选区域都放入训练，这样的思路是错的。一张图片中，背景占了绝大多数地方，这样就导致训练用的正样本远远少于负样本，对训练不利。正确的做法是对所有候选区域进行随机采样，要求采样的结果中正样本有x张，负样本y张，且保证x与y在数值上相近。（对于一些问题，不大容易做到x:y = 1:1，但至少x与y应该在同一数量级下）

### 2.5 如何训练?
RCNN的网络架构，注定了它不能像其他网络那样进行端到端（end-to-end）的训练。

前面提到RCNN分为4个阶段：Proposal阶段、特征提取阶段、分类阶段、回归阶段。这4个阶段都是**相互独立训练的**。  

首先，特征提取器是AlexNet，将它的最后一层fc7进行改造，使得fc7能够输出分类结果。Proposal阶段对每张图片产生了1k~2k个候选区域，把这些图片依照正负样本比例喂给特征提取器，特征提取器fc7输出的分类结果与标签结果进行比对，完成特征提取器的训练。特征提取器的训练完成后，fc7层的使命也完成了，后面的分类器和回归器只会用到fc6、pool5的输出。

然后，Proposal和特征提取器已经训练完毕了。把它们的结果fc6，输入到分类器SVM中，SVM输出与标签结果比对，完成SVM的训练。

最后，回归器的训练也和SVM类似，只不过回归器取的是pool5的结果。

为什么不能同时进行上面3步的训练？因为特征提取器是CNN，分类器是SVM，回归器是脊回归器(Ridge Regression)表示该回归器加了L2正则化，不属于同一体系，无法共同训练。甚至在测试时，也需要把每一阶段的结果先保存到磁盘，再喂入下一阶段。这是非常麻烦的一件事。

## SPPNet
特点:  
1.一般CNN后接全连接层或者分类器，他们都需要固定的输入尺寸，因此不得不对输入数据进行crop或者warp，这些预处理会造成数据的丢失或几何的失真。SPP Net的第一个贡献就是将金字塔思想加入到CNN，实现了数据的多尺度输入。

如下图所示，在卷积层和全连接层之间加入了SPP layer。此时网络的输入可以是任意尺度的，在SPP layer中每一个pooling的filter会根据输入调整大小，而SPP的输出尺度始终是固定的。
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2017-07-10.png)

2.只对原图提取一次卷积特征
在R-CNN中，每个候选框先resize到统一大小，然后分别作为CNN的输入，这样是很低效的。  
所以SPP Net根据这个缺点做了优化：只对原图进行一次卷积得到整张图的feature map，然后找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer和之后的层。节省了大量的计算时间，比R-CNN有一百倍左右的提速。  

## FastRCNN
特征提取还是用的selective-search  
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2017-32-28.png)
### 1.单层sppnet的网络层，叫做ROI Pooling
这个网络层可以把不同大小的输入映射到一个固定尺度的特征向量，而我们知道，conv、pooling、relu等操作都不需要固定size的输入，因此，在原始图片上执行这些操作后，虽然输入图片size不同导致得到的feature map尺寸也不同，不能直接接到一个全连接层fc进行分类，但是可以加入这个神奇的ROI Pooling层，对每个region都提取一个固定维度的特征表示，再通过正常的softmax进行类型识别。另外，之前RCNN的处理流程是先提proposal，然后CNN提取特征，之后用SVM分类器，最后再做bbox regression，而在Fast-RCNN中，作者巧妙的把bbox regression放进了神经网络内部，与region分类和并成为了一个multi-task模型，实际实验也证明，这两个任务能够共享卷积特征，并相互促进。Fast-RCNN很重要的一个贡献是成功的让人们看到了Region Proposal+CNN这一框架实时检测的希望，原来多类检测真的可以在保证准确率的同时提升处理速度，也为后来的Faster-RCNN做下了铺垫。

### 2.R-CNN有一些相当大的缺点（把这些缺点都改掉了，就成了Fast R-CNN)  
大缺点：由于每一个候选框都要独自经过CNN，这使得花费的时间非常多。
解决：共享卷积层，现在不是每一个候选框都当做输入进入CNN了，而是输入一张完整的图片，在第五个卷积层再得到每个候选框的特征

原来的方法：许多候选框（比如两千个）-->CNN-->得到每个候选框的特征-->分类+回归
现在的方法：一张完整图片-->CNN-->得到每张候选框的特征-->分类+回归

所以容易看见，Fast RCNN相对于RCNN的提速原因就在于不像RCNN把每个候选区域给深度网络提特征，而是整张图提一次特征，再把候选框映射到conv5上，而SPP只需要计算一次特征，剩下的只需要在conv5层上操作就可以了。  

### 3.1 为什么叫Fast?
将特征提取器、分类器、回归器合并，使得训练过程不需要再将每阶段结果保存磁盘单独训练，可以一次性完成训练，加快了训练速度。这是Fast之一。

对整张图片进行特征提取，用ROI层处理候选区域的特征，使得原本每一个候选区域都要做一次特征提取，变为了现在一整张图片做一次特征提取。训练速度（8.8倍）和测试速度（146倍）都大大加快，这是Fast之二。

### 3.2 分类器和回归器的实现细节?
分类器应该都能想到，用的softmax代替SVM。  
回归器求出（x,y,w,h）4个量，分别代表定位框左上角的坐标xy、宽度w、高度h，损失函数用的是**Smooth-L1**  

## FasterRCNN
![]()
一种网络，四个损失函数  
- RPN calssification(anchor good.bad)
- RPN regression(anchor->propoasal)
- Fast R-CNN classification(over classes)
- Fast R-CNN regression(proposal ->box)

![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2018-37-31.png)
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2018-37-55.png)

## MaskRCNN 
### 1.相对于FasterRCNN进行的修改  
1）将 Roi Pooling 层替换成了 RoiAlign；

2）Faster-RCNN网络的最后分别是分类网络和回归网络两条路并行，Mask-RCNN则是再加一条Mask网络与它们并行,Mask网络的实现是FCN网络，这也是语义分割领域中非常经典的网络结构。

3) 特征提取网络改为ResNet101+FPN  

### 2.MaskRCNN技术要点
- 技术要点1 - 强化的基础网络

 通过 ResNeXt-101+FPN 用作特征提取网络，达到 state-of-the-art 的效果。

- 技术要点2 - ROIAlign

 采用 ROIAlign 替代 RoiPooling（改进池化操作）。引入了一个插值过程，先通过双线性插值到14\*14，再 pooling到7\*7，很大程度上解决了仅通过 Pooling 直接采样带来的 Misalignment 对齐问题。

PS： 虽然 Misalignment 在分类问题上影响并不大，但在 Pixel 级别的 Mask 上会存在较大误差。

- 技术要点3 - Loss Function

 每个 ROIAlign 对应 K * m^2 维度的输出。K 对应类别个数，即输出 K 个mask，m对应 池化分辨率（7*7）。Loss 函数定义：

 Lmask(Cls_k) = Sigmoid (Cls_k)，    平均二值交叉熵 （average binary cross-entropy）Loss，通过逐像素的 Sigmoid 计算得到。

     Why K个mask？通过对每个 Class 对应一个 Mask 可以有效避免类间竞争（其他 Class 不贡献 Loss ）。


## Summary

|            | 步骤                                  | 缺点 | 改进 |
| ----       | ----                                  | ---- | ---- |
| RCNN       | 1.selective-search提取region proposal<br>2.CNN提取特征<br>3.SVM分类<br>4.bbox回归|1.训练步骤繁琐(微调网络+训练SVM+训练bbox)<br>2.训练测试均速度慢<br>3.训练占空间|1.从DPM,HSC的34.3%直接提升到了66%(mAP)|　
| FastRCNN   |1.selective-search提取region proposal<br>2.CNN提取特征(ROI pooling)<br>3.softmax分类<br>4.多任务损失函数边框回归|1.依旧用selective-search提取region proposal,特征提取很耗时<br>2.无法满足实时应用，没有真正实现端对端训练测试<br>3.利用GPU但是区域建议方法时在CPU上实现|1.66.9%->70%,应用SPPNet启发single-sppnet(ROI)|
| FasterRCNN |1.Region proposal net提取region proposal<br>2.CNN提取特征<br>3.softmax分类<br>4.多任务损失函数边框回归|1.无法达到实时检测目标<br>2.获取region proposal,再对每个proposal分类计算量还是比较大|1.提高了检测精度和速度<br>2.真正实现了端对端的目标检测框架<br>3.生成region proposal仅需10ms |

