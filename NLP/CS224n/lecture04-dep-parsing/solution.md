## 1.Machine Learning and Neural Networks

**(a)** i.Because SGD uses mini-batches and not the entire dataset to calculate the gradient for updating the weights, the updates suffer from a significant amount of noise. According to the formula and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\beta_1) almost set to be 0.9, At this time, the gradient value for the moving average m is mainly affected by the moving average of the previous gradient, and the gradient calculated this time will be scaled to the original ![](https://latex.codecogs.com/png.image?\dpi{110}%201-\beta_1). Even if the gradient obtained in this calculation is very large (gradient explosion), this effect will also be reduced, thereby preventing major changes in the update. 

 &ensp;&ensp;&nbsp;ii.The model parameter with the smallest moving average gradient will get a larger update.
On the one hand, the update of the parameter with the smaller gradient is made larger to help it get out of the local optimum (saddle point); on the other hand, the update of the parameter with the larger gradient is made smaller to make the update more stable. Combining the above two aspects makes learning faster and more stable. By reducing the degree of gradient change, each gradient update is made more stable, so that the model learning is more stable and the convergence speed is faster, and this also slows down the update speed of the parameters of the larger gradient value to ensure its update stability. The division by the element-wise square root serves as a normalizer that treats all weights equally. To a large extend each individual weight is given an update of roughly equal magnitude irrespectively of the size of it's derivate.

**(b)** i.
![](https://latex.codecogs.com/png.image?\dpi{110}%20\gamma%20=\frac{1}{1-p_{drop}})  

explain: ![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_i(1-p_{drop})h_i%20=%20(1-p_{drop})E[h])       
 &ensp;&ensp;&nbsp;![](https://latex.codecogs.com/png.image?\dpi{110}%20\sum_i[h_{drop}]_i=\gamma%20\sum_i(1-p_{drop})h_i=\gamma(1-p_{drop})E[h]=E[h])
