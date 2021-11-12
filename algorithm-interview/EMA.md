### Exponential Moving Average(EMA)指数移动平均  

解决梯度下降算法中收敛过慢的问题。

在深度学习中，采用 SGD 或者其他的一些优化算法 `(Adam, Momentum)` 训练神经网络时，通常会使用一个叫 `ExponentialMovingAverage (EMA)` 的方法指数滑动平均，这个方法对模型的参数做平均，以求提高测试指标并增加模型鲁棒。  

### What is MovingAverage?  


### EMA定义  
指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），是一种给予近期数据更高权重的平均方法。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-16%2018-44-14.png)  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-16%2018-44-47.png)  
![]()
