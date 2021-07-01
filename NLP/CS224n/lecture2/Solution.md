In word2vec, the conditional probability distribution is given by taking vector dot-products and applying
the softmax function:

![](https://latex.codecogs.com/png.image?\dpi{110}%20P(O=o|C%20=%20c)=\frac{exp(u^T_0v_0)}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)})   

Here,  ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_0) is the ‘outside’ vector representing outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20o), and ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c) is the ‘center’ vector representing center word ![](https://latex.codecogs.com/png.image?\dpi{110}%20c). To contain these parameters, we have two matrices, ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) . The columns of ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) are all the ‘outside’ vectors ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_w). The columns of ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) are all of the ‘center’ vectors ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_w). Both ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) contain a vector for every
![](https://latex.codecogs.com/png.image?\dpi{110}%20w%20\in%20Vocabulary) .   

Recall from lectures that, for a single pair of words c and o, the loss is given by:  

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,o,U)=-logP(O=o|C=c))&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;  (2)  

Another way to view this loss is as the cross-entropy between the true distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and the predicted
distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) Here, both ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) are vectors with length equal to the number of words in the vocabulary.Furthermore, the ![](https://latex.codecogs.com/png.image?\dpi{110}%20k^{th}) entry in these vectors indicates the conditional probability of the ![](https://latex.codecogs.com/png.image?\dpi{110}%20k^{th}) word being an ‘outside word’ for the given c. The true empirical distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) is a one-hot vector with a 1 for the true outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20o) , and 0 everywhere else. The predicted distribution ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) is the probability distribution  ![](https://latex.codecogs.com/png.image?\dpi{110}%20P(O|C=c))given by our model in equation (1).  

### (a) (3 points) 

Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss
between ![](https://latex.codecogs.com/png.image?\dpi{110}%20y) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\hat{y}) ; i.e.,show that 

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{w%20\in{Vocab}}y_w%20log(\hat{y}_w)=-log(\hat{y_o}))&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; &ensp;&ensp;&ensp;&ensp; &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(3)   

Your answer should be one line.  

Answer:  

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\sum_{w%20\in{Vocab}}y_w%20log(\hat{y}_w)=-y_olog(\hat{y_o})-\sum_{w%20\in%20Vocab,w\neq%20o}y_wlog(\hat{y_w})=-log(\hat{y_o}))
 
### (b) (5 points) 

Compute the partial derivative of &ensp; ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,%20o,%20U)) &ensp; with respect to ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c) . Please write your answer in terms of ![](https://latex.codecogs.com/png.image?\dpi{110}%20y,\hat{y}) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c).  

Answer:

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,o,U)=-log(P(O=o|C=c)))      
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-log(\frac{exp(u_o^Tv_c)}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)})=-u_o^Tv_c+log(\sum_{w\in%20Vocab}exp(u_w^T%20v_c)))
![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J_{naive-softmax(v_c,o,U)}}{\partial%20v_c}=-u_o+\frac{\partial%20(log(\sum_{w\in%20Vocab}exp(u_w^T%20v_c)))}{\partial%20v_c})     
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=u_o+\frac{1}{\sum_{w\in%20Vocab}exp(u_w^Tv_c)}\sum_{w\in%20Vocab}\frac{\partial%20(exp(u_w^T%20v_c))}{\partial%20v_c})        
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=-u_o+\frac{1}{\sum_{w\in%20Vocab}exp(u_w^T%20v_c)}\sum_{w\in%20Vocab}exp(u_w^T%20v_c)u_w)      
![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture2/img/Screen%20Shot%202021-07-01%20at%204.15.13%20PM.png)
### (c) (5 points) 

Compute the partial derivatives of ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,%20o,%20U)) with respect to each of the ‘outside’ word vectors, ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_w)’s. There will be two cases: when ![](https://latex.codecogs.com/png.image?\dpi{110}%20w=o), the true ‘outside’ word vector, and ![](https://latex.codecogs.com/png.image?\dpi{110}%20w\neq%20o), for all other words. Please write you answer in terms of ![](https://latex.codecogs.com/png.image?\dpi{110}%20y,\hat{y}) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20U)

Answer:

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J(v_c,o,U)}{\partial%20u_w}==\frac{\partial%20u_o^Tv_c}{\partial%20u_w}+\frac{\partial%20(log\sum_{w\in%20Vocab}exp(u_w^Tv_c))}{\partial%20u_w})   
当 ![](https://latex.codecogs.com/png.image?\dpi{110}%20w\neq%20o) 时   

![](https://latex.codecogs.com/png.image?\dpi{110}%20-\frac{\partial%20u_o^Tv_c}{\partial%20u_w}=0)    

![](https://latex.codecogs.com/png.image?\dpi{110}%20\therefore%20\frac{\partial%20J}{\partial%20u_w}=\frac{\partial%20(log\sum_{w%20\in%20Vocab}exp(u_w^Tv_c))}{\partial%20u_w}=\frac{1}{\sum_{w%20\in%20Vocab}exp(u_w^Tv_c)}\sum_{w%20\in%20Vocab}exp(u_w^Tv_c)v_c)    
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%200+p(O=w|C=c)v_c=\hat{y_w}v_c)

当![](https://latex.codecogs.com/png.image?\dpi{110}%20w=o)时  

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J(v_c,o,U)}{\partial%20u_w}=-v_c+p(O=o|C=c)v_c)    
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\hat{y_w}v_c-v_c=(\hat{y_w}-1)v_c)

Then:

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture2/img/Screen%20Shot%202021-07-01%20at%204.26.44%20PM.png)

### (d)

The sigmoid function is given by the follow Equation :

![](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{e^x+1})

Please compute the derivative of ![](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma(x)) with respect to ![](https://latex.codecogs.com/png.image?\dpi{110}%20x), where ![](https://latex.codecogs.com/png.image?\dpi{110}%20x) is a vector.     

Answer:

![](https://github.com/weiweia92/blog/blob/main/NLP/CS224n/lecture2/img/Screen%20Shot%202021-07-01%20at%204.30.16%20PM.png)

### (e)

Now we shall consider the Negative Sampling loss, which is an alternative to the Naive Softmax loss. Assume that ![](https://latex.codecogs.com/png.image?\dpi{110}%20K) negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as ![](https://latex.codecogs.com/png.image?\dpi{110}%20w_1,w_2,...,w_K) and their outside vectors as ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_1,...,u_K). Note that ![](https://latex.codecogs.com/png.image?\dpi{110}%20o%20\notin%20\{w_1,...,w_K%20\}) . For a center word ![](https://latex.codecogs.com/png.image?\dpi{110}%20c) and an outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20o), the negative sampling loss function is given by:  

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{neg-sample}(v_c,o,U)=-log(\sigma(u_o^Tv_c))-\sum_{k=1}^Klog(\sigma(-u_k^Tv_c)))    

for a sample  ![](https://latex.codecogs.com/png.image?\dpi{110}%20w_1,w_2,...,w_K) ,where ![](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma(\cdot)) is sigmoid function.

Please repeat parts b and c, computing the partial derivatives of ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{neg-sample}) respect to ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c),with respect to ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o), and with respect to a negative sample ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_k).Please write your answers in terms of the vectors ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o),![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_k), where ![](https://latex.codecogs.com/png.image?\dpi{110}%20k%20\in%20[1,K]). After you've done this, describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able to use your solution to part (d) to help compute the necessary gradients here.

Answer:

For ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c):

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J_{neg-sample}}{\partial%20v_c}=(\sigma(u_o^Tv_c)-1)u_o+\sum_{k=1}^{K}(1-\sigma(-u_k^Tv_c))u_k)     
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=(\sigma(u_o^Tv_c)-1)u_o+\sum_{k=1}^K\sigma(u_k^Tv_c)u_k)

For ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_o), remember:     

![](https://latex.codecogs.com/png.image?\dpi{110}%20o%20\notin%20\{w_1,...,w_K%20\})

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J_{neg-sample}}{\partial%20u_o}=-\frac{1}{\sigma(u_o^Tv_c)}\cdot%20\sigma(u_o^Tv_c)(1-\sigma(u_o^Tv_c))v_c)      
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=(\sigma(u_o^Tv_c)-1)v_c)   

For ![](https://latex.codecogs.com/png.image?\dpi{110}%20u_k):

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J_{neg-sample}}{\partial%20u_o}=-\frac{1}{\sigma(-u_k^Tv_c)}\cdot%20\[\sigma(-u_k^Tv_c)(1-\sigma(-u_k^Tv_c))%20\](-v_c))     
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=(1-\sigma(-u_k^Tv_c))v_c)     
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;![](https://latex.codecogs.com/png.image?\dpi{110}%20=\sigma(u_k^Tv_c)v_c)    

Why this loss function is much more efficient to compute than the naive-softmax loss?

For naive softmax loss function:

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J(v_c,o,U)}{\partial%20v_c}=U(\hat{y}-y))

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J(v_c,o,U)}{\partial%20U}=v_c(\hat{y}-y)^T)

For negative sampling loss function:

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J}{\partial%20v_c}=\sigma(-u_o^Tv_c)u_o+\sum_{k=1}^{K}\sigma(u_k^Tv_c)u_k)

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J}{\partial%20u_o}=\sigma(-u_o^Tv_c)v_c=(\sigma(u_o^Tv_c)-1)v_c)

![](https://latex.codecogs.com/png.image?\dpi{110}%20\frac{\partial%20J}{\partial%20u_k}=\sigma(u_k^Tv_c)v_c)&ensp;&ensp;&ensp;&ensp;&ensp;for all ![](https://latex.codecogs.com/png.image?\dpi{110}%20k=1,2,...,K)

从求得的偏导数中我们可以看出，原始的softmax函数每次对 ![](https://latex.codecogs.com/png.image?\dpi{110}%20v_c)进行反向传播时，需要与 output vector matrix 进行大量且复杂的矩阵运算，而负采样中的计算复杂度则不再与词表大小 ![](https://latex.codecogs.com/png.image?\dpi{110}%20V)有关，而是与采样数量 ![](https://latex.codecogs.com/png.image?\dpi{110}%20K) 有关。  

### (f)

Suppose the center word is ![](https://latex.codecogs.com/png.image?\dpi{110}%20c=w_t) and the context window is ![](https://latex.codecogs.com/png.image?\dpi{110}%20\[%20w_{t-m},...,w_{t-1},w_t,w_{t+1},...,w_{t+m}\]) ,where ![](https://latex.codecogs.com/png.image?\dpi{110}%20m) is the context window size. Recall that for the skip-gram version of**word2vec**,the total loss for the context window is 

![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)=\sum_{-m\leq%20j\leq%20m%20\%20j\neq%20o}J(v_c,w_{t+j},U))    

Here,![](https://latex.codecogs.com/png.image?\dpi{110}%20J(v_c,w_{t+j},U)) represents an arbitrary loss term for the center word ![](https://latex.codecogs.com/png.image?\dpi{110}%20c=w_t) and outside word ![](https://latex.codecogs.com/png.image?\dpi{110}%20w_{t+j}).![](https://latex.codecogs.com/png.image?\dpi{110}%20J(v_c,w_{t+j},U)) could be ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{naive-softmax}(v_c,w_{t+j},U)) or ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{neg-softmax}(v_c,w_{t+j},U)), depending on your implementation.   

Write down three partial derivatives:

![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)/\partial%20U)

![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)/\partial%20v_c)

![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J_{skip-gram}(v_c,w_{t-m},...,w_{t+m},U)/\partial%20v_w) when ![](https://latex.codecogs.com/png.image?\dpi{110}%20w%20\neq%20c)  

Write your answers in terms of ![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J(v_c,w_{t+j},U)/%20\partial%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J(v_c,w_{t+j},U)/%20\partial%20v_c) .This is very simple-each solution should be one line.

**Once you're done**:Given that you computed the derivatives of ![](https://latex.codecogs.com/png.image?\dpi{110}%20\partial%20J(v_c,w_{t+j},U)) with respect of all the model parameters ![](https://latex.codecogs.com/png.image?\dpi{110}%20U) and ![](https://latex.codecogs.com/png.image?\dpi{110}%20V) in parts a to c, you have now computed the derivatives of the full loss function ![](https://latex.codecogs.com/png.image?\dpi{110}%20J_{skip-gram}) with respect to all parameters, You're ready to implement **word2vec**.

Answer:

![]()
