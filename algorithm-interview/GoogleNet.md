### GoogleNet
Inception  
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2016-14-37.png)
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2016-14-55.png)
Inception v3,v4  
![](https://github.com/weiweia92/pictures/blob/master/deeplearning/Screenshot%20from%202020-07-13%2016-20-30.png)

### ResNet
short connect可以有效缓解反向传播时由于深度过深导致的梯度消失现象，这使得网络加深之后性能不会变差  
ResNet的关键点是:(1)使用short connection，使训练深层网络更容易，并且重复堆叠相同的模块组合。(2)ResNet大量使用了批量归一层。(3)对于很深的网络(超过50层)，ResNet使用了更高效的瓶颈(bottleneck)结构,如下图.  
![]()

### ResNext

受精简而高效的Inception模块启发，ResNeXt将ResNet中非短路那一分支变为多个分支。和Inception不同的是，每个分支的结构都相同.ResNeXt的关键点是:(1) 沿用ResNet的短路连接，并且重复堆叠相同的模块组合。(2)多分支分别处理。(3)使用1×1卷积降低计算量。其综合了ResNet和Inception的优点。此外，ResNeXt巧妙地利用分组卷积进行实现。ResNeXt发现，增加分支数是比加深或加宽更有效地提升网络性能的方式。如图  
![]()
