## 1.Machine Learning and Neural Networks

**(a)** i.Because SGD uses mini-batches and not the entire dataset to calculate the gradient for updating the weights, the updates suffer from a significant amount of noise. According to the formula and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\beta_1) almost set to be 0.9, At this time, the gradient value for the moving average m is mainly affected by the moving average of the previous gradient, and the gradient calculated this time will be scaled to the original ![](https://latex.codecogs.com/png.image?\dpi{110}%201-\beta_1). Even if the gradient obtained in this calculation is very large (gradient explosion), this effect will also be reduced, thereby preventing major changes in the update. 

 &ensp;&ensp;&nbsp;ii.The model parameter with the smallest moving average gradient will get a larger update.
On the one hand, the update of the parameter with the smaller gradient is made larger to help it get out of the local optimum (saddle point); on the other hand, the update of the parameter with the larger gradient is made smaller to make the update more stable. Combining the above two aspects makes learning faster and more stable. By reducing the degree of gradient change, each gradient update is made more stable, so that the model learning is more stable and the convergence speed is faster, and this also slows down the update speed of the parameters of the larger gradient value to ensure its update stability. The division by the element-wise square root serves as a normalizer that treats all weights equally. To a large extend each individual weight is given an update of roughly equal magnitude irrespectively of the size of it's derivate.

**(b)** i.![](https://latex.codecogs.com/png.image?\dpi{110}%20\gamma%20=\frac{1}{1-p_{drop}})  

explain:

&ensp;&ensp;&nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_i(1-p_{drop})h_i%20=%20(1-p_{drop})E[h])    

&ensp;&ensp;&nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_i[h_{drop}]_i=\gamma%20\sum_i(1-p_{drop})h_i=\gamma(1-p_{drop})E[h]=E[h])

&ensp;&ensp;&nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\therefore%20\gamma%20=%20\frac{1}{1-p_{drop}})    

&ensp;&ensp;&nbsp;ii. One way to look at dropout regularization is as an ensemble learning method that combines many different weaker classifier. Each classifier is train to some extent separately and learns a different aspect of the problem. During evaluation we wouldn't be leveraging(利用) all the learned experience from the different classifiers. If we apply dropout during the evaluation period, the evaluation result will be random and will not reflect the true performance of the model, which violates the original intention of regularization. By disabling dropout during the evaluation period, the performance of the model and the effect of regularization can be observed to ensure that the parameters of the model are updated correctly.     

## 2.Neural Transition-Based Dependency Parsing

![](https://github.com/weiweia92/blog/blob/main/NLP/pic/Screen%20Shot%202021-07-14%20at%208.34.57%20PM.png)   

**(b)** A sentence with n words will be parsed in 2n steps. Every word in the buffer needs to be eventually put on the stack which takes n steps. Eventually each word has to be removed from the stack to form a dependency which takes another n steps.    

**(c)** Coding  

**(d)** Coding   

**(e)** Results    
![](https://github.com/weiweia92/blog/blob/main/NLP/pic/Screen%20Shot%202021-07-15%20at%204.55.43%20PM.png)
